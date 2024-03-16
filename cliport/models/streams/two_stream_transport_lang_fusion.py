import torch
import numpy as np

import cliport.models as models
import cliport.models.core.fusion as fusion
from cliport.models.core.transport import Transport


class TwoStreamTransportLangFusion(Transport):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l):
        logits = self.fusion_key(self.key_stream_one(in_tensor), self.key_stream_two(in_tensor, l))
        kernel = self.fusion_query(self.query_stream_one(crop), self.query_stream_two(crop, l))
        return logits, kernel

    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)
        # print(inp_img.shape, in_tensor.shape)  # (320, 160, 6) torch.Size([1, 384, 224, 6])
        # Rotation pivot.

        pv = np.array([p[0], p[1]]) + self.pad_size
        # print('pv', pv.shape)  # pv (2,)

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        # print(crop.shape)  # torch.Size([36, 6, 64, 64])
        logits, kernel = self.transport(in_tensor, crop, lang_goal)
        # print(logits.shape, kernel.shape)  # torch.Size([1, 3, 384, 224]) torch.Size([36, 3, 64, 64])
        # TODO(Mohit): Crop after network. Broken for now.
        # # Crop after network (for receptive field, and more elegant).
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor, lang_goal)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)
        # hcrop = self.pad_size
        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        return self.correlate(logits, kernel, softmax)


class TwoStreamTransportLangFusionLat(TwoStreamTransportLangFusion):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel


class TwoStreamTransportLangFusion_time(Transport):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l):
        logits = self.fusion_key(self.key_stream_one(in_tensor), self.key_stream_two(in_tensor, l))
        kernel = self.fusion_query(self.query_stream_one(crop), self.query_stream_two(crop, l))
        return logits, kernel

    def forward(self, inp_img, p, lang_goal=None, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)
        # print(inp_img.shape, in_tensor.shape)  # (320, 160, 6) torch.Size([1, 384, 224, 6])
        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size
        # print('pv', pv.shape)  # pv (2,)

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        logits, kernel = self.transport(in_tensor, crop, lang_goal)

        # print(logits.shape, kernel.shape)  # torch.Size([1, 3, 384, 224]) torch.Size([36, 3, 64, 64])
        # TODO(Mohit): Crop after network. Broken for now.
        # # Crop after network (for receptive field, and more elegant).
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor, lang_goal)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)
        # hcrop = self.pad_size
        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        return self.correlate(logits, kernel, softmax)


class TwoStreamTransportLangFusion_two(Transport):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_thr = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_for = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)

        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, crop1, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        query_out_one, query_lat_one = self.query_stream_thr(crop1)
        query_out_two = self.query_stream_for(crop1, query_lat_one, l)
        kernel1 = self.fusion_query(query_out_one, query_out_two)
        return logits, kernel, kernel1

    def transport_key(self, in_tensor, l):
        logits = self.fusion_key(self.key_stream_one(in_tensor), self.key_stream_two(in_tensor, l))
        return logits

    def transport_query_1(self, crop, l):
        kernel = self.fusion_query(self.query_stream_one(crop), self.query_stream_two(crop, l))
        return kernel

    def transport_query_2(self, crop, l):
        kernel = self.fusion_query(self.query_stream_thr(crop), self.query_stream_for(crop, l))
        return kernel

    def forward(self, inp_img, p1, p2, lang_goal=None, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)
        # print(inp_img.shape, in_tensor.shape)  # (320, 160, 6) torch.Size([1, 384, 224, 6])
        # Rotation pivot.
        pv1 = np.array([p1[0], p1[1]]) + self.pad_size
        # print('pv', pv.shape)  # pv (2,)
        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop_1 = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop_1 = self.rotator(crop_1, pivot=pv1)
        crop_1 = torch.cat(crop_1, dim=0)
        crop_1 = crop_1[:, :, pv1[0]-hcrop:pv1[0]+hcrop, pv1[1]-hcrop:pv1[1]+hcrop]

        pv2 = np.array([p2[0], p2[1]]) + self.pad_size
        crop_2 = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop_2 = self.rotator(crop_2, pivot=pv2)
        crop_2 = torch.cat(crop_2, dim=0)
        crop_2 = crop_2[:, :, pv2[0] - hcrop:pv2[0] + hcrop, pv2[1] - hcrop:pv2[1] + hcrop]
        logits, kernel_1, kernel_2 = self.transport(in_tensor, crop_1, crop_2, lang_goal)

        return self.correlate(logits, kernel_1, softmax), self.correlate(logits, kernel_2, softmax)

        # print(logits.shape, kernel.shape)  # torch.Size([1, 3, 384, 224]) torch.Size([36, 3, 64, 64])
        # TODO(Mohit): Crop after network. Broken for now.
        # # Crop after network (for receptive field, and more elegant).
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor, lang_goal)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)
        # hcrop = self.pad_size
        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]


class TwoStreamTransportLangFusionLat_two(TwoStreamTransportLangFusion_two):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, crop1, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        query_out_one, query_lat_one = self.query_stream_thr(crop1)
        query_out_two = self.query_stream_for(crop1, query_lat_one, l)
        kernel1 = self.fusion_query(query_out_one, query_out_two)
        return logits, kernel, kernel1

    def transport_key(self, in_tensor, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)
        return logits

    def transport_query_1(self, crop, l):
        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)
        return kernel

    def transport_query_2(self, crop, l):
        query_out_one, query_lat_one = self.query_stream_thr(crop)
        query_out_two = self.query_stream_for(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)
        return kernel


class TwoStreamTransportLangFusion_net(Transport):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)


        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, crop1, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop1)
        query_out_two = self.query_stream_two(crop1, query_lat_one, l)
        kernel1 = self.fusion_query(query_out_one, query_out_two)
        return logits, kernel, kernel1


    def forward(self, inp_img, p1, p2, lang_goal=None, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)
        # print(inp_img.shape, in_tensor.shape)  # (320, 160, 6) torch.Size([1, 384, 224, 6])
        # Rotation pivot.
        pv1 = np.array([p1[0], p1[1]]) + self.pad_size
        # print('pv', pv.shape)  # pv (2,)
        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop_1 = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop_1 = self.rotator(crop_1, pivot=pv1)
        crop_1 = torch.cat(crop_1, dim=0)
        crop_1 = crop_1[:, :, pv1[0]-hcrop:pv1[0]+hcrop, pv1[1]-hcrop:pv1[1]+hcrop]

        pv2 = np.array([p2[0], p2[1]]) + self.pad_size
        crop_2 = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop_2 = self.rotator(crop_2, pivot=pv2)
        crop_2 = torch.cat(crop_2, dim=0)
        crop_2 = crop_2[:, :, pv2[0] - hcrop:pv2[0] + hcrop, pv2[1] - hcrop:pv2[1] + hcrop]
        logits, kernel_1, kernel_2 = self.transport(in_tensor, crop_1, crop_2, lang_goal)

        return self.correlate(logits, kernel_1, softmax), self.correlate(logits, kernel_2, softmax)

        # print(logits.shape, kernel.shape)  # torch.Size([1, 3, 384, 224]) torch.Size([36, 3, 64, 64])
        # TODO(Mohit): Crop after network. Broken for now.
        # # Crop after network (for receptive field, and more elegant).
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor, lang_goal)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)
        # hcrop = self.pad_size
        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]


class TwoStreamTransportLangFusionLat_net(TwoStreamTransportLangFusion_net):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, crop1, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop1)
        query_out_two = self.query_stream_two(crop1, query_lat_one, l)
        kernel1 = self.fusion_query(query_out_one, query_out_two)
        return logits, kernel, kernel1

    def transport_key(self, in_tensor, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)
        return logits

    def transport_query_1(self, crop, l):
        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)
        return kernel

    def transport_query_2(self, crop, l):
        query_out_one, query_lat_one = self.query_stream_thr(crop)
        query_out_two = self.query_stream_for(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)
        return kernel

