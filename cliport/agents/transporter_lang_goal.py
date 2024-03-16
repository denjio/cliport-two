import numpy as np
from cliport.utils import utils
from cliport.agents.transporter import TransporterAgent, TransporterAgent_two, TransporterAgent_time,TransporterAgent_net
from cliport.models.streams.one_stream_attention_lang_fusion import OneStreamAttentionLangFusion
from cliport.models.streams.one_stream_transport_lang_fusion import OneStreamTransportLangFusion
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusion
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusion
# from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusion_two
# from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusion_time
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat_two, TwoStreamTransportLangFusionLat_net
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat_two
import matplotlib.pyplot as plt


class TwoStreamClipLingUNetTransporterAgent(TransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_lingunet'
        self.attention = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']
        lang_goal = inp['lang_goal']

        out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax)
        return out

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'p0': p0, 'lang_goal': lang_goal}
        out = self.trans_forward(inp, softmax=False)
        # print(out.shape)  # ([1, 36, 320, 160])
        # print(p1) # (110, 104)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        return loss, err

    def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)  # torch.Size([320, 160, 1])
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
        place_conf = self.trans_forward(place_inp)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
            'place': [p1_pix[0], p1_pix[1], p1_theta],
        }


class TwoStreamClipFilmLingUNetLatTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_film_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipLingUNetLatTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamRN50BertLingUNetTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'rn50_bert_lingunet'
        self.attention = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamUntrainedRN50BertLingUNetTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'untrained_rn50_bert_lingunet'
        self.attention = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamRN50BertLingUNetLatTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'rn50_bert_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class OriginalTransporterLangFusionAgent(TwoStreamClipLingUNetTransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'plain_resnet_lang'
        self.attention = OneStreamAttentionLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class ClipLingUNetTransporterAgent(TwoStreamClipLingUNetTransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'clip_lingunet'
        self.attention = OneStreamAttentionLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipLingUNetTransporterAgent_two(TransporterAgent_two):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_lingunet'
        self.attention = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out


    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        p00, p00_theta = frame['p00'], frame['p00_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        # print('out', out.shape)  # torch.Size([1, 51200])
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta, p00, p00_theta)

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p1 = inp['p0']
        p2 = inp['p00']
        l_goal = inp['lang_goal']

        out1, out2 = self.transport.forward(inp_img=inp_img, p1=p1, p2=p2, lang_goal=l_goal, softmax=softmax)

        return out1, out2

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        p00 = frame['p00']
        p11, p11_theta = frame['p11'], frame['p11_theta']
        lang_goal = frame['lang_goal']
        inp = {'inp_img': inp_img, 'p0': p0, 'p00': p00, 'lang_goal':lang_goal }
        out1, out2 = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out1, p0, p1, p1_theta, out2, p00, p11,
                                             p11_theta)
        return loss, err


    def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
            """Run inference and return best action given visual observations."""
            # Get heightmap from RGB-D images.
            img = self.test_ds.get_image(obs)
            lang_goal = info['lang_goal']

            # Attention model forward pass.
            pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
            pick_conf = self.attn_forward(pick_inp)  # torch.Size([320, 160, 1])
            pick_conf = pick_conf.detach().cpu().numpy()

            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            # print(argmax)
            # another pick
            p00_pix = argmax[2:4]
            p00_theta = argmax[4] * (2 * np.pi / pick_conf.shape[2])

            # Transport model forward pass.
            place_inp = {'inp_img': img, 'p0': p0_pix, 'p00': p00_pix, 'lang_goal': lang_goal}
            place_conf1, place_conf11 = self.trans_forward(place_inp)
            place_conf1 = place_conf1.permute(1, 2, 0)
            place_conf1 = place_conf1.detach().cpu().numpy()
            argmax = np.argmax(place_conf1)
            argmax = np.unravel_index(argmax, shape=place_conf1.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf1.shape[2])

            # another place
            place_conf11 = place_conf11.permute(1, 2, 0)
            place_conf11 = place_conf11.detach().cpu().numpy()
            argmax = np.argmax(place_conf11)
            argmax = np.unravel_index(argmax, shape=place_conf11.shape)
            p11_pix = argmax[:2]
            p11_theta = argmax[2] * (2 * np.pi / place_conf11.shape[2])

            # Pixels to end effector poses.
            hmap = img[:, :, 3]
            p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
            p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
            p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
            p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

            # 另一个点的坐标
            p00_xyz = utils.pix_to_xyz(p00_pix, hmap, self.bounds, self.pix_size)
            p11_xyz = utils.pix_to_xyz(p11_pix, hmap, self.bounds, self.pix_size)
            p00_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p00_theta))
            p11_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p11_theta))

            return {
                'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
                'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
                'pick': [p0_pix[0], p0_pix[1], p0_theta],
                'place': [p1_pix[0], p1_pix[1], p1_theta],

                'pose00': (np.asarray(p00_xyz), np.asarray(p00_xyzw)),
                'pose11': (np.asarray(p11_xyz), np.asarray(p11_xyzw)),
                'pick0': [p00_pix[0], p00_pix[1], p0_theta],
                'place1': [p11_pix[0], p11_pix[1], p11_theta],
            }


class TwoStreamClipLingUNetLatTransporterAgent_two(TwoStreamClipLingUNetTransporterAgent_two):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat_two(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat_two(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipLingUNetTransporterAgent_time(TransporterAgent_time):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_lingunet'
        self.attention = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

        self.attention_1 = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport_1 = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_forward_1(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention_1.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        p00, p00_theta = frame['p00'], frame['p00_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        out1 = self.attn_forward_1(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta, out1, p00, p00_theta)

    def trans_forward(self, inp, sfmax=True):
        inp_img = inp['inp_img']
        p = inp['p0']
        l_goal = inp['lang_goal']

        out = self.transport.forward(inp_img=inp_img, p=p, lang_goal=l_goal, softmax=sfmax)
        return out

    def trans_forward_1(self, inp, sfmax=True):
        inp_img = inp['inp_img']
        # print(inp)
        p = inp['p00']
        l_goal = inp['lang_goal']

        out = self.transport_1.forward(inp_img=inp_img, p=p, lang_goal=l_goal, softmax=sfmax)
        return out

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        p00 = frame['p00']
        p11, p11_theta = frame['p11'], frame['p11_theta']
        lang_goal = frame['lang_goal']
        out = None
        out1 = None
        inp = {'inp_img': inp_img, 'p0': p0, 'p00': p00, 'lang_goal': lang_goal}
        if p1_theta != None:
            out = self.trans_forward(inp, sfmax=False)
        if p11_theta != None:
            out1 = self.trans_forward_1(inp, sfmax=False)
        loss, err = self.transport_criterion_1(backprop, compute_err, inp, out, p0, p1, p1_theta, out1, p00, p11,
                                               p11_theta)
        return loss, err
        # if type(p00) == tuple:
        #     inp = {'inp_img': inp_img, 'p0': p00, 'lang_goal': lang_goal}
        #     out1 = self.trans_forward(inp, sfmax=False)
        #     err1, loss1 = self.transport_criterion(backprop, compute_err, inp, out1, p00, p11, p11_theta)
        #     for i in err1.keys():
        #         err[i] = err1[i]
        #     loss += loss1
        #     return loss, err
        # else:
        #     return loss, err


    def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)  # torch.Size([320, 160, 1])
        pick_conf = pick_conf.detach().cpu().numpy()

        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
        print(argmax)
        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
        place_conf = self.trans_forward(place_inp)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # 另一个点的pick
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward_1(pick_inp)  # torch.Size([320, 160, 1])
        pick_conf = pick_conf.detach().cpu().numpy()
        print(argmax)
        argmax = np.argmax(pick_conf)

        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p00_pix = argmax[:2]
        p00_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # 另一个点的place
        place_inp = {'inp_img': img, 'p0': p00_pix, 'lang_goal': lang_goal}
        place_conf11 = self.trans_forward_1(place_inp)
        place_conf11 = place_conf11.permute(1, 2, 0)
        place_conf11 = place_conf11.detach().cpu().numpy()
        argmax = np.argmax(place_conf11)
        argmax = np.unravel_index(argmax, shape=place_conf11.shape)
        p11_pix = argmax[:2]
        p11_theta = argmax[2] * (2 * np.pi / place_conf11.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        # 另一个点的坐标
        p00_xyz = utils.pix_to_xyz(p00_pix, hmap, self.bounds, self.pix_size)
        p11_xyz = utils.pix_to_xyz(p11_pix, hmap, self.bounds, self.pix_size)
        p00_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p00_theta))
        p11_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p11_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
            'place': [p1_pix[0], p1_pix[1], p1_theta],

            'pose00': (np.asarray(p00_xyz), np.asarray(p00_xyzw)),
            'pose11': (np.asarray(p11_xyz), np.asarray(p11_xyzw)),
            'pick0': [p00_pix[0], p00_pix[1], p0_theta],
            'place1': [p11_pix[0], p11_pix[1], p11_theta],
        }


class TwoStreamClipLingUNetLatTransporterAgent_time(TwoStreamClipLingUNetTransporterAgent_time):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

        self.attention_1 = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport_1 = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipLingUNetLatTransporterAgent_net(TransporterAgent_net):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat_two(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat_net(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        p00, p00_theta = frame['p00'], frame['p00_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        # print('out', out.shape)  # torch.Size([1, 51200])
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta, p00, p00_theta)

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p1 = inp['p0']
        p2 = inp['p00']
        l_goal = inp['lang_goal']

        out1, out2 = self.transport.forward(inp_img=inp_img, p1=p1, p2=p2, lang_goal=l_goal, softmax=softmax)

        return out1, out2

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        p00 = frame['p00']
        p11, p11_theta = frame['p11'], frame['p11_theta']
        lang_goal = frame['lang_goal']
        inp = {'inp_img': inp_img, 'p0': p0, 'p00': p00, 'lang_goal': lang_goal}
        out1, out2 = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out1, p0, p1, p1_theta, out2, p00, p11,
                                             p11_theta)
        return loss, err

    def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        print(img.shape)

        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        # 找到前两个最大值的索引
        max_indices = np.argpartition(-pick_conf.flatten(), 2)[:2]
        # 将扁平化后的索引转换为坐标
        max_coords = np.unravel_index(max_indices, pick_conf.shape)
        p0_pix = (max_coords[0][0], max_coords[1][0])
        p0_theta = max_coords[2][0] * (2 * np.pi / pick_conf.shape[2])
        p00_pix = (max_coords[0][1], max_coords[1][1])
        p00_theta = max_coords[2][1] * (2 * np.pi / pick_conf.shape[2])
        for i in range(7):
            if i == 6:
                plt.subplot(1, 7, i + 1)
                plt.imshow(pick_conf)
            else:
                plt.subplot(1, 7, i + 1)
                plt.imshow(img[:, :, i])
        plt.show()
        print(p0_pix, p00_pix)
        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'p00': p00_pix, 'lang_goal': lang_goal}
        place_conf1, place_conf2 = self.trans_forward(place_inp)
        place_conf1 = place_conf1.permute(1, 2, 0)
        place_conf1 = place_conf1.detach().cpu().numpy()
        argmax = np.argmax(place_conf1)
        argmax = np.unravel_index(argmax, shape=place_conf1.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf1.shape[2])

        place_conf2 = place_conf2.permute(1, 2, 0)
        place_conf2 = place_conf2.detach().cpu().numpy()
        argmax = np.argmax(place_conf1)
        argmax = np.unravel_index(argmax, shape=place_conf2.shape)
        p11_pix = argmax[:2]
        p11_theta = argmax[2] * (2 * np.pi / place_conf2.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))
        # 另一个点的坐标
        p00_xyz = utils.pix_to_xyz(p00_pix, hmap, self.bounds, self.pix_size)
        p11_xyz = utils.pix_to_xyz(p11_pix, hmap, self.bounds, self.pix_size)
        p00_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p00_theta))
        p11_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p11_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick0': [p0_pix[0], p0_pix[1], p0_theta],
            'place1': [p1_pix[0], p1_pix[1], p1_theta],

            'pose00': (np.asarray(p00_xyz), np.asarray(p00_xyzw)),
            'pose11': (np.asarray(p11_xyz), np.asarray(p11_xyzw)),
            'pick00': [p00_pix[0], p00_pix[1], p0_theta],
            'place11': [p11_pix[0], p11_pix[1], p11_theta],
        }
