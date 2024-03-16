import numpy as np
import torch
import torch.nn.functional as F

from cliport.models.core.attention import Attention
import cliport.models as models
import cliport.models.core.fusion as fusion

import matplotlib.pyplot as plt


class TwoStreamAttentionLangFusion(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # print(inp_img.shape)  # (320, 160, 6)
        in_data = np.pad(inp_img, self.padding, mode='constant')
        # print(in_data.shape) # (320, 320, 6)
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        # print(in_data.shape) (1, 320, 320, 6)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]
        # print('in_tens', in_tens.shape)  # (1, 320, 320, 6)

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2
        # print('pv', pv.shape, pv, in_data.shape[1:3])  # pv (2,), [160 160] (320, 320)

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        # print(in_tens.shape) # Size([1, 6, 320, 320])
        in_tens = self.rotator(in_tens, pivot=pv)
        # print(len(in_tens), in_tens[0].shape)  # 1, Size([1, 6, 320, 320])

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x, lang_goal)
            logits.append(lgts)
        # print(len(logits), logits[0].shape) # 1 torch.Size([1, 1, 320, 320])
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)

        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]
        # print(logits.shape, c0, c1)  # torch.Size([1, 1, 320, 160]) [ 0 80] [320 240]
        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        # print(output.shape, np.prod(logits.shape)) # torch.Size([1, 51200]) 51200
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        # print(output.shape)  # torch.Size([320, 160, 1])
        return output


class TwoStreamAttentionLangFusionLat(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def attend(self, x, l):
        # print(x.shape, l)  # [1, 6, 320, 320]) put the brown block on the lightest brown block
        x1, lat = self.attn_stream_one(x)
        # print(x1.shape, len(lat), lat[0].shape)  # torch.Size([1, 1, 320, 320]) 6 torch.Size([1, 1024, 20, 20])
        x2 = self.attn_stream_two(x, lat, l)
        # print(x2.shape)  # torch.Size([1, 1, 320, 320])
        x = self.fusion(x1, x2)
        # print(x.shape)  # torch.Size([1, 1, 320, 320])
        return x


class TwoStreamAttentionLangFusion_two(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # print(inp_img.shape)  # (320, 160, 6)
        in_data = np.pad(inp_img, self.padding, mode='constant')
        # print(in_data.shape) # (320, 320, 6)
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        # print(in_data.shape) (1, 320, 320, 6)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]
        # print('in_tens', in_tens.shape)  # (1, 320, 320, 6)

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2
        # print('pv', pv.shape, pv, in_data.shape[1:3])  # pv (2,), [160 160] (320, 320)

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        # print(in_tens.shape) # Size([1, 6, 320, 320])
        in_tens = self.rotator(in_tens, pivot=pv)
        # print(len(in_tens), in_tens[0].shape)  # 1, Size([1, 6, 320, 320])

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x, lang_goal)
            logits.append(lgts)
        # print(len(logits), logits[0].shape) # 1 torch.Size([1, 1, 320, 320])
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)

        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]
        # print(logits.shape, c0, c1)  # torch.Size([1, 1, 320, 160]) [ 0 80] [320 240]
        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        # print(output.shape, np.prod(logits.shape)) # torch.Size([1, 51200]) 51200
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        # print(output.shape)  # torch.Size([320, 160, 1])
        return output


class TwoStreamAttentionLangFusionLat_two(TwoStreamAttentionLangFusion_two):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def attend(self, x, l):
        # print(x.shape, l)  # [1, 6, 320, 320]) put the brown block on the lightest brown block
        x1, lat = self.attn_stream_one(x)
        # print(x1.shape, len(lat), lat[0].shape)  # torch.Size([1, 1, 320, 320]) 6 torch.Size([1, 1024, 20, 20])
        x2 = self.attn_stream_two(x, lat, l)
        # print(x2.shape)  # torch.Size([1, 1, 320, 320])
        x = self.fusion(x1, x2)
        # print(x.shape)  # torch.Size([1, 1, 320, 320])
        return x
