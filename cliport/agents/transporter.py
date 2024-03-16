import os
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
from cliport.tasks import cameras
from cliport.utils import utils
from cliport.models.core.attention import Attention
from cliport.models.core.transport import Transport
from cliport.models.streams.two_stream_attention import TwoStreamAttention
from cliport.models.streams.two_stream_transport import TwoStreamTransport
from cliport.models.streams.two_stream_attention import TwoStreamAttentionLat
from cliport.models.streams.two_stream_transport import TwoStreamTransportLat


def window(inp, p0_pix):
    inp['inp_img'] = np.around(inp['inp_img'], 5)
    i = 0
    while p0_pix[0] + i < 319 and inp['inp_img'][p0_pix[0] + i, p0_pix[1], 5] == inp['inp_img'][
        p0_pix[0], p0_pix[1], 5]:
        # print(inp['inp_img'][p0_pix[0] + i, p0_pix[1], 5])
        i += 1
    y1 = p0_pix[0] + i

    i = 0
    while p0_pix[0] + i > 0 and inp['inp_img'][p0_pix[0] + i, p0_pix[1], 5] == inp['inp_img'][p0_pix[0], p0_pix[1], 5]:
        i -= 1
    y0 = p0_pix[0] + i

    i = 0
    while p0_pix[1] + i < 149 and inp['inp_img'][p0_pix[0], p0_pix[1] + i, 5] == inp['inp_img'][
        p0_pix[0], p0_pix[1], 5]:
        # print(inp['inp_img'][p0_pix[0], p0_pix[1] + i, 5])
        i += 1
    x1 = p0_pix[1] + i

    i = 0
    while p0_pix[1] + i > 0 and inp['inp_img'][p0_pix[0], p0_pix[1] + i, 5] == inp['inp_img'][p0_pix[0], p0_pix[1], 5]:
        i -= 1
    x0 = p0_pix[1] + i
    # print(y0, y1, x0, x1)
    inp['inp_img'][y0:y1, x0:x1, :] = 0
    return inp


class TransporterAgent(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = cfg['train']['n_rotations']

        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']

        self._build_model()
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr']),
            'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr'])
        }
        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

    def _build_model(self):
        self.attention = None
        self.transport = None
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']

        output = self.attention.forward(inp_img, softmax=softmax)
        return output

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']

        inp = {'inp_img': inp_img}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        # Get label.
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)

        # Backpropagate.
        if backprop:
            attn_optim = self._optimizers['attn']
            self.manual_backward(loss, attn_optim)
            attn_optim.step()
            attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(p) - p0_pix, ord=1),
                'theta': np.absolute((theta - p0_theta) % np.pi)
            }
        return loss, err

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']

        output = self.transport.forward(inp_img, p0, softmax=softmax)
        return output

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']

        inp = {'inp_img': inp_img, 'p0': p0}
        output = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, output, p0, p1, p1_theta)
        return loss, err

    def transport_criterion(self, backprop, compute_err, inp, output, p, q, theta):
        itheta = theta / (2 * np.pi / self.transport.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.transport.n_rotations

        # Get one-hot pixel label map.
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
        output = output.reshape(1, np.prod(output.shape))
        loss = self.cross_entropy_with_logits(output, label)
        if backprop:
            transport_optim = self._optimizers['trans']
            self.manual_backward(loss, transport_optim)
            transport_optim.step()
            transport_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            place_conf = self.trans_forward(inp)
            place_conf = place_conf.permute(1, 2, 0)
            place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(q) - p1_pix, ord=1),
                'theta': np.absolute((theta - p1_theta) % np.pi)
            }
        self.transport.iters += 1
        return err, loss

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()

        frame, _ = batch

        # print('frame', frame)
        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame)
        else:
            loss1, err1 = self.transport_training_step(frame)
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.trainer.global_step
        if (global_step + 1) in self.save_steps:
            self.trainer.run_evaluation()
            val_loss = self.trainer.callback_metrics['val_loss']
            steps = f'{global_step + 1:05d}'
            filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        utils.set_seed(self.trainer.current_epoch + 1)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)

        print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(total_attn_dist_err, total_attn_theta_err))
        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))

        return dict(
            val_loss=mean_val_total_loss,
            val_loss0=mean_val_loss0,
            mean_val_loss1=mean_val_loss1,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
        )

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)

        # Attention model forward pass.
        pick_inp = {'inp_img': img}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix}
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
            'pick': p0_pix,
            'place': p1_pix,
        }

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu,
                       using_native_amp, using_lbfgs):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'])
        self.to(device=self.device_type)


class TransporterAgent_two(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = cfg['train']['n_rotations']

        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']

        self._build_model()
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr']),
            'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr'])
        }
        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

    def _build_model(self):
        self.attention = None
        self.transport = None
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well

        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']

        output = self.attention.forward(inp_img, softmax=softmax)
        return output

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        p00, p00_theta = frame['p00'], frame['p00_theta']

        inp = {'inp_img': inp_img}
        out = self.attn_forward(inp, softmax=False)
        # print('out', out)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta, p00, p00_theta)

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta, p00, theta00):
        # Get label. label为概率图，正确位置概率为1，错误位置概率为0
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        # print(label_size)  # (320, 160, 1)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)
        # print('label', label.shape)  # torch.Size([1, 51200])
        # print(torch.unique(label))  # 0, 1
        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)

        if theta00 != None:
            # 另一个点的预测
            theta00_i = theta00 / (2 * np.pi / self.attention.n_rotations)
            theta00_i = np.int32(np.round(theta00_i)) % self.attention.n_rotations
            label1 = np.zeros(label_size)
            label1[p00[0], p00[1], theta00_i] = 1
            label1 = label1.transpose((2, 0, 1))
            label1 = label1.reshape(1, np.prod(label1.shape))
            label1 = torch.from_numpy(label1).to(dtype=torch.float, device=out.device)

            # Get loss.
            loss += self.cross_entropy_with_logits(out, label1)

            # Backpropagate.
            if backprop:
                attn_optim = self._optimizers['attn']
                self.manual_backward(loss, attn_optim)
                attn_optim.step()
                attn_optim.zero_grad()
        else:
            # Backpropagate.
            if backprop:
                attn_optim = self._optimizers['attn']
                self.manual_backward(loss, attn_optim)
                attn_optim.step()
                attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            # print(p, p00)
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            # print(argmax)
            # print(pick_conf.shape)
            # print(inp['inp_img'].shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
            # for i in range(7):
            #     if i == 6:
            #         plt.subplot(1, 7, i + 1)
            #         plt.imshow(pick_conf)
            #     else:
            #         plt.subplot(1, 7, i + 1)
            #         plt.imshow(inp['inp_img'][:, :, i])
            # plt.show()
            # 这里要加mask,避免预测同一个点
            inp_window = window(inp, p0_pix)
            # for i in range(7):
            #     if i == 6:
            #         plt.subplot(1, 7, i + 1)
            #         plt.imshow(pick_conf)
            #     else:
            #         plt.subplot(1, 7, i + 1)
            #         plt.imshow(inp_window['inp_img'][:, :, i])
            # plt.show()
            pick_conf = self.attn_forward(inp_window)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p00_pix = argmax[:2]
            p00_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
            print(p0_pix, p00_pix)
            err = {
                'dist0': np.linalg.norm(np.array(p) - p0_pix, ord=1),
                'theta0': np.absolute((theta - p0_theta) % np.pi),
                'dist00': np.linalg.norm(np.array(p00) - p00_pix, ord=1),
                'theta00': np.absolute((theta00 - p00_theta) % np.pi)
            }
        return loss, err

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
        inp = {'inp_img': inp_img, 'p0': p0}
        out1, out2 = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out1, p0, p1, p1_theta, out2, p00, p11,
                                             p11_theta)
        return loss, err

    def transport_criterion(self, backprop, compute_err, inp, out1, p0, p1, theta1, out2, p00, p11, theta11):
        itheta1 = theta1 / (2 * np.pi / self.transport.n_rotations)
        itheta1 = np.int32(np.round(itheta1)) % self.transport.n_rotations

        # Get one-hot pixel label map.
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
        label = np.zeros(label_size)
        label[p1[0], p1[1], itheta1] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out1.device)
        output = out1.reshape(1, np.prod(out1.shape))

        loss = self.cross_entropy_with_logits(output, label)
        if out2 != None:
            itheta11 = theta11 / (2 * np.pi / self.transport.n_rotations)
            itheta11 = np.int32(np.round(itheta11)) % self.transport.n_rotations

            # Get one-hot pixel label map.
            inp_img = inp['inp_img']
            label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
            label = np.zeros(label_size)
            label[p11[0], p11[1], itheta11] = 1

            # Get loss.
            label = label.transpose((2, 0, 1))
            label = label.reshape(1, np.prod(label.shape))
            label = torch.from_numpy(label).to(dtype=torch.float, device=out2.device)
            output = out2.reshape(1, np.prod(out2.shape))

            loss += self.cross_entropy_with_logits(output, label)
            if backprop:
                transport_optim = self._optimizers['trans']
                self.manual_backward(loss, transport_optim)
                transport_optim.step()
                transport_optim.zero_grad()
        else:
            if backprop:
                transport_optim = self._optimizers['trans']
                self.manual_backward(loss, transport_optim)
                transport_optim.step()
                transport_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            place_conf1, place_conf2 = self.trans_forward(inp)
            place_conf1 = place_conf1.permute(1, 2, 0)
            place_conf1 = place_conf1.detach().cpu().numpy()
            argmax = np.argmax(place_conf1)
            argmax = np.unravel_index(argmax, shape=place_conf1.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf1.shape[2])

            err = {
                'dist1': np.linalg.norm(np.array(p1) - p1_pix, ord=1),
                'theta1': np.absolute((theta1 - p1_theta) % np.pi),
            }
            if place_conf2 != None:
                place_conf2 = place_conf2.permute(1, 2, 0)
                place_conf2 = place_conf2.detach().cpu().numpy()
                argmax = np.argmax(place_conf2)
                argmax = np.unravel_index(argmax, shape=place_conf2.shape)
                p11_pix = argmax[:2]
                p11_theta = argmax[2] * (2 * np.pi / place_conf2.shape[2])
                err['dist11'] = np.linalg.norm(np.array(p11) - p11_pix, ord=1),
                err['theta11'] = np.absolute((theta11 - p11_theta) % np.pi),

        self.transport.iters += 1
        return err, loss

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()

        frame, _ = batch

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame)
        else:
            loss1, err1 = self.transport_training_step(frame)
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.trainer.global_step
        if (global_step + 1) in self.save_steps:
            self.trainer.run_evaluation()
            val_loss = self.trainer.callback_metrics['val_loss']
            steps = f'{global_step + 1:05d}'
            filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist0'],
            val_attn_theta_err=err0['theta0'],
            val_trans_dist_err=err1['dist1'],
            val_trans_theta_err=err1['theta1'],
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        utils.set_seed(self.trainer.current_epoch + 1)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)

        print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(total_attn_dist_err, total_attn_theta_err))
        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))

        return dict(
            val_loss=mean_val_total_loss,
            val_loss0=mean_val_loss0,
            mean_val_loss1=mean_val_loss1,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
        )

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)

        # Attention model forward pass.
        pick_inp = {'inp_img': img}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix}
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
            'pick': p0_pix,
            'place': p1_pix,
        }

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu,
                       using_native_amp, using_lbfgs):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'])
        self.to(device=self.device_type)


class TransporterAgent_net(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = cfg['train']['n_rotations']

        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']

        self._build_model()
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr']),
            'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr'])
        }
        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

    def _build_model(self):
        self.attention = None
        self.transport = None
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well

        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']

        output = self.attention.forward(inp_img, softmax=softmax)
        return output

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        p00, p00_theta = frame['p00'], frame['p00_theta']

        inp = {'inp_img': inp_img}
        out = self.attn_forward(inp, softmax=False)
        # print('out', out)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta, p00, p00_theta)

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta, p00, theta00):
        # Get label. label为概率图，正确位置概率为1，错误位置概率为0
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        # print(label_size)  # (320, 160, 1)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        # 另一个点的预测
        theta00_i = theta00 / (2 * np.pi / self.attention.n_rotations)
        theta00_i = np.int32(np.round(theta00_i)) % self.attention.n_rotations
        label[p00[0], p00[1], theta00_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)
        # print('label', label.shape)  # torch.Size([1, 51200])
        # print(torch.unique(label))  # 0, 1
        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)

        # Backpropagate.
        if backprop:
            attn_optim = self._optimizers['attn']
            self.manual_backward(loss, attn_optim)
            attn_optim.step()
            attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            # print(p, p00)
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            # argmax = np.argmax(pick_conf)
            # argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            # # # print(argmax)
            # # # print(pick_conf.shape)
            # # # print(inp['inp_img'].shape)
            # #
            # p0_pix = argmax[:2]
            # p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
            # 找到前两个最大值的索引
            max_indices = np.argpartition(-pick_conf.flatten(), 2)[:2]
            # 将扁平化后的索引转换为坐标
            max_coords = np.unravel_index(max_indices, pick_conf.shape)

            p0_pix = (max_coords[0][0], max_coords[1][0])
            p0_theta = max_coords[2][0] * (2 * np.pi / pick_conf.shape[2])
            p00_pix = (max_coords[0][1], max_coords[1][1])
            p00_theta = max_coords[2][1] * (2 * np.pi / pick_conf.shape[2])
            # for i in range(7):
            #     if i == 6:
            #         plt.subplot(1, 7, i + 1)
            #         plt.imshow(pick_conf)
            #     else:
            #         plt.subplot(1, 7, i + 1)
            #         plt.imshow(inp['inp_img'][:, :, i])
            # plt.show()
            # 这里要加mask,避免预测同一个点

            # for i in range(7):
            #     if i == 6:
            #         plt.subplot(1, 7, i + 1)
            #         plt.imshow(pick_conf)
            #     else:
            #         plt.subplot(1, 7, i + 1)
            #         plt.imshow(inp_window['inp_img'][:, :, i])
            # plt.show()
            # pick_conf = self.attn_forward(inp)
            # pick_conf = pick_conf.detach().cpu().numpy()
            # argmax = np.argmax(pick_conf)
            # argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            # p00_pix = argmax[:2]
            # p00_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
            print(p0_pix, p00_pix)
            err = {
                'dist0': np.linalg.norm(np.array(p) - p0_pix, ord=1),
                'theta0': np.absolute((theta - p0_theta) % np.pi),
                'dist00': np.linalg.norm(np.array(p00) - p00_pix, ord=1),
                'theta00': np.absolute((theta00 - p00_theta) % np.pi)
            }
        return loss, err

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
        inp = {'inp_img': inp_img, 'p0': p0}
        out1, out2 = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out1, p0, p1, p1_theta, out2, p00, p11,
                                             p11_theta)
        return loss, err

    def transport_criterion(self, backprop, compute_err, inp, out1, p0, p1, theta1, out2, p00, p11, theta11):
        itheta1 = theta1 / (2 * np.pi / self.transport.n_rotations)
        itheta1 = np.int32(np.round(itheta1)) % self.transport.n_rotations

        # Get one-hot pixel label map.
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
        label = np.zeros(label_size)
        label[p1[0], p1[1], itheta1] = 1
        itheta11 = theta11 / (2 * np.pi / self.transport.n_rotations)
        itheta11 = np.int32(np.round(itheta11)) % self.transport.n_rotations
        label[p11[0], p11[1], itheta11] = 1
        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out1.device)
        output = out1.reshape(1, np.prod(out1.shape))

        loss = self.cross_entropy_with_logits(output, label)

        if backprop:
            transport_optim = self._optimizers['trans']
            self.manual_backward(loss, transport_optim)
            transport_optim.step()
            transport_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            place_conf1, place_conf2 = self.trans_forward(inp)
            place_conf1 = place_conf1.permute(1, 2, 0)
            place_conf1 = place_conf1.detach().cpu().numpy()
            argmax = np.argmax(place_conf1)
            argmax = np.unravel_index(argmax, shape=place_conf1.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf1.shape[2])
            place_conf2 = place_conf2.permute(1, 2, 0)
            place_conf2 = place_conf2.detach().cpu().numpy()
            argmax = np.argmax(place_conf2)
            argmax = np.unravel_index(argmax, shape=place_conf2.shape)
            p11_pix = argmax[:2]
            p11_theta = argmax[2] * (2 * np.pi / place_conf2.shape[2])
            err = {
                'dist1': np.linalg.norm(np.array(p1) - p1_pix, ord=1),
                'theta1': np.absolute((theta1 - p1_theta) % np.pi),
                'dist11': np.linalg.norm(np.array(p11) - p11_pix, ord=1),
                'theta11': np.absolute((theta11 - p11_theta) % np.pi),
            }

        self.transport.iters += 1
        return err, loss

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()

        frame, _ = batch

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame)
        else:
            loss1, err1 = self.transport_training_step(frame)
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.trainer.global_step
        if (global_step + 1) in self.save_steps:
            self.trainer.run_evaluation()
            val_loss = self.trainer.callback_metrics['val_loss']
            steps = f'{global_step + 1:05d}'
            filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist0'] + err0['dist00'],
            val_attn_theta_err=err0['theta0'] + err0['theta00'],
            val_trans_dist_err=err1['dist1'] + err1['dist11'],
            val_trans_theta_err=err1['theta1'] + err1['dist11'],
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        utils.set_seed(self.trainer.current_epoch + 1)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)

        print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(total_attn_dist_err, total_attn_theta_err))
        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))

        return dict(
            val_loss=mean_val_total_loss,
            val_loss0=mean_val_loss0,
            mean_val_loss1=mean_val_loss1,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
        )

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        pick_inp = {'inp_img': img}
        # Attention model forward pass.
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

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'p00': p00_pix}
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

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu,
                       using_native_amp, using_lbfgs):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'])
        self.to(device=self.device_type)


class TransporterAgent_time(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = cfg['train']['n_rotations']

        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']

        self._build_model()
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr']),
            'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr']),
            'attn1': torch.optim.Adam(self.attention_1.parameters(), lr=self.cfg['train']['lr']),
            'trans1': torch.optim.Adam(self.transport_1.parameters(), lr=self.cfg['train']['lr'])
        }
        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

    def _build_model(self):
        self.attention = None
        self.transport = None
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well

        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']

        output = self.attention.forward(inp_img, softmax=softmax)
        return output

    def attn_forward_1(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention_1.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        p00, p00_theta = frame['p00'], frame['p00_theta']

        inp = {'inp_img': inp_img}
        out = self.attn_forward(inp, softmax=False)
        out1 = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta, out1, p00, p00_theta)

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta, out1, p00, theta00):
        loss = torch.tensor(0).cuda(0)
        loss1 = torch.tensor(0).cuda(0)
        if theta != None:
            # Get label.
            theta_i = theta / (2 * np.pi / self.attention.n_rotations)
            theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
            inp_img = inp['inp_img']
            label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
            label = np.zeros(label_size)
            label[p[0], p[1], theta_i] = 1
            label = label.transpose((2, 0, 1))
            label = label.reshape(1, np.prod(label.shape))
            label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)
            # Get loss.
            loss = self.cross_entropy_with_logits(out, label)
            # Backpropagate.
            if loss != 0 and backprop:
                attn_optim = self._optimizers['attn']
                self.manual_backward(loss, attn_optim)
                attn_optim.step()
                attn_optim.zero_grad()

        if theta00 != None:
            # 另一个点的预测
            theta00_i = theta00 / (2 * np.pi / self.attention_1.n_rotations)
            theta00_i = np.int32(np.round(theta00_i)) % self.attention_1.n_rotations
            inp_img = inp['inp_img']
            label_size = inp_img.shape[:2] + (self.attention_1.n_rotations,)
            label1 = np.zeros(label_size)
            label1[p00[0], p00[1], theta00_i] = 1
            label1 = label1.transpose((2, 0, 1))
            label1 = label1.reshape(1, np.prod(label1.shape))
            label1 = torch.from_numpy(label1).to(dtype=torch.float, device=out1.device)

            # Get loss. 后面可以考虑out1和out差异化
            loss1 = self.cross_entropy_with_logits(out1, label1)
            # Backpropagate.
            if loss1 != 0 and backprop:
                attn_optim1 = self._optimizers['attn1']
                self.manual_backward(loss1, attn_optim1)
                attn_optim1.step()
                attn_optim1.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            if loss != 0 and theta != None:
                pick_conf = self.attn_forward(inp)
                pick_conf = pick_conf.detach().cpu().numpy()
                argmax = np.argmax(pick_conf)
                argmax = np.unravel_index(argmax, shape=pick_conf.shape)
                p0_pix = argmax[:2]
                p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
                err['dist'] = np.linalg.norm(np.array(p) - p0_pix, ord=1)
                err['theta'] = np.absolute((theta - p0_theta) % np.pi)
            if loss1 != 0 and theta00 != None:
                pick_conf = self.attn_forward_1(inp)
                pick_conf = pick_conf.detach().cpu().numpy()
                argmax = np.argmax(pick_conf)
                argmax = np.unravel_index(argmax, shape=pick_conf.shape)
                p00_pix = argmax[:2]
                p00_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
                err['dist00'] = np.linalg.norm(np.array(p00) - p00_pix, ord=1)
                err['theta00'] = np.absolute((theta00 - p00_theta) % np.pi)
        return loss + loss1, err

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']

        output = self.transport.forward(inp_img, p0, softmax=softmax)
        return output

    def trans_forward_1(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']

        output = self.transport_1.forward(inp_img, p0, softmax=softmax)
        return output

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
            out = self.trans_forward(inp, sofmax=False)
        if p11_theta != None:
            out1 = self.trans_forward_1(inp, sofmax=False)
        loss, err = self.transport_criterion_1(backprop, compute_err, inp, out, p0, p1, p1_theta, out1, p00, p11,
                                               p11_theta)
        return loss, err

    # def transport_criterion(self, backprop, compute_err, inp, output, p, q, theta):
    #     itheta = theta / (2 * np.pi / self.transport.n_rotations)
    #     itheta = np.int32(np.round(itheta)) % self.transport.n_rotations
    #
    #     # Get one-hot pixel label map.
    #     inp_img = inp['inp_img']
    #     label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
    #     label = np.zeros(label_size)
    #     label[q[0], q[1], itheta] = 1
    #
    #     # Get loss.
    #     label = label.transpose((2, 0, 1))
    #     label = label.reshape(1, np.prod(label.shape))
    #     label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
    #     output = output.reshape(1, np.prod(output.shape))
    #
    #     loss = self.cross_entropy_with_logits(output, label)
    #
    #     if backprop:
    #         transport_optim = self._optimizers['trans']
    #         self.manual_backward(loss, transport_optim)
    #         transport_optim.step()
    #         transport_optim.zero_grad()
    #
    #     # Pixel and Rotation error (not used anywhere).
    #     err = {}
    #     if compute_err:
    #         place_conf = self.trans_forward(inp)
    #         place_conf = place_conf.permute(1, 2, 0)
    #         place_conf = place_conf.detach().cpu().numpy()
    #         argmax = np.argmax(place_conf)
    #         argmax = np.unravel_index(argmax, shape=place_conf.shape)
    #         p1_pix = argmax[:2]
    #         p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
    #
    #         err = {
    #             'dist': np.linalg.norm(np.array(q) - p1_pix, ord=1),
    #             'theta': np.absolute((theta - p1_theta) % np.pi),
    #         }
    #     self.transport.iters += 1
    #     return err, loss

    def transport_criterion_1(self, backprop, compute_err, inp, output, p, q, theta, output1, p1, q1, theta1):
        loss = torch.tensor(0).cuda(0)
        loss1 = torch.tensor(0).cuda(0)
        if theta != None:
            itheta = theta / (2 * np.pi / self.transport.n_rotations)
            itheta = np.int32(np.round(itheta)) % self.transport.n_rotations

            # Get one-hot pixel label map.
            inp_img = inp['inp_img']
            label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
            label = np.zeros(label_size)
            label[q[0], q[1], itheta] = 1

            # Get loss.
            label = label.transpose((2, 0, 1))
            label = label.reshape(1, np.prod(label.shape))
            label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
            output = output.reshape(1, np.prod(output.shape))
            loss = self.cross_entropy_with_logits(output, label)
            if loss != 0 and backprop:
                transport_optim = self._optimizers['trans']
                self.manual_backward(loss, transport_optim)
                transport_optim.step()
                transport_optim.zero_grad()

        if theta1 != None:
            itheta1 = theta1 / (2 * np.pi / self.transport_1.n_rotations)
            itheta1 = np.int32(np.round(itheta1)) % self.transport_1.n_rotations

            # Get one-hot pixel label map.
            inp_img = inp['inp_img']
            label_size = inp_img.shape[:2] + (self.transport_1.n_rotations,)
            label1 = np.zeros(label_size)
            label1[q[0], q[1], itheta1] = 1

            # Get loss.
            label1 = label1.transpose((2, 0, 1))
            label1 = label1.reshape(1, np.prod(label1.shape))
            label1 = torch.from_numpy(label1).to(dtype=torch.float, device=output1.device)
            output1 = output1.reshape(1, np.prod(output1.shape))
            loss1 = self.cross_entropy_with_logits(output1, label1)

            if loss1 != 0 and backprop:
                transport_optim1 = self._optimizers['trans1']
                self.manual_backward(loss1, transport_optim1)
                transport_optim1.step()
                transport_optim1.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            if loss != 0 and theta != None:
                place_conf = self.trans_forward(inp)
                place_conf = place_conf.permute(1, 2, 0)
                place_conf = place_conf.detach().cpu().numpy()
                argmax = np.argmax(place_conf)
                argmax = np.unravel_index(argmax, shape=place_conf.shape)
                p1_pix = argmax[:2]
                p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
                err['dist'] = np.linalg.norm(np.array(q) - p1_pix, ord=1),
                err['theta'] = np.absolute((theta - p1_theta) % np.pi),
            if loss1 != 0 and theta1 != None:
                place_conf = self.trans_forward_1(inp)
                place_conf = place_conf.permute(1, 2, 0)
                place_conf = place_conf.detach().cpu().numpy()
                argmax = np.argmax(place_conf)
                argmax = np.unravel_index(argmax, shape=place_conf.shape)
                p11_pix = argmax[:2]
                p11_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
                err['dist1'] = np.linalg.norm(np.array(q1) - p11_pix, ord=1),
                err['theta1'] = np.absolute((theta1 - p11_theta) % np.pi),

        self.transport.iters += 1
        return loss + loss1, err

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()
        frame, _ = batch  # 这里是dataset的_getitem_函数的返回 sample 和 goal
        # print('frame', frame.keys()) # dict_keys(['img', 'p0', 'p0_theta', 'p1', 'p1_theta', 'perturb_params',
        # 'p00', 'p00_theta', 'p11', 'p11_theta', 'perturb_params1', 'lang_goal'])
        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame)
        else:
            loss1, err1 = self.transport_training_step(frame)
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.trainer.global_step
        if (global_step + 1) in self.save_steps:
            self.trainer.run_evaluation()
            val_loss = self.trainer.callback_metrics['val_loss']
            steps = f'{global_step + 1:05d}'
            filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        utils.set_seed(self.trainer.current_epoch + 1)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)

        print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(total_attn_dist_err, total_attn_theta_err))
        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))

        return dict(
            val_loss=mean_val_total_loss,
            val_loss0=mean_val_loss0,
            mean_val_loss1=mean_val_loss1,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
        )

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)

        # Attention model forward pass.
        pick_inp = {'inp_img': img}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix}
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
            'pick': p0_pix,
            'place': p1_pix,
        }

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu,
                       using_native_amp, using_lbfgs):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'])
        self.to(device=self.device_type)


class OriginalTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'plain_resnet'
        self.attention = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class ClipUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'clip_unet'
        self.attention = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_unet'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipUNetLatTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_unet_lat'
        self.attention = TwoStreamAttentionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipWithoutSkipsTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        # TODO: lateral version
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_woskip'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamRN50BertUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        # TODO: lateral version
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'rn50_bert_unet'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
