"""
Standalone HiFiGAN vocoder — no speechbrain dependency.
Adapted from speechbrain.lobes.models.HifiGAN (Apache 2.0 License).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

LRELU_SLOPE = 0.1


def _same_padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2


class WNConv1d(nn.Module):
    """Conv1d with weight normalization and same-padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, bias=True, weight_norm=True):
        super().__init__()
        padding = _same_padding(kernel_size, dilation) if stride == 1 else 0
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, padding=padding, bias=bias,
        )
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)


class WNConvTranspose1d(nn.Module):
    """ConvTranspose1d with weight normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, weight_norm=True):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
        )
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            WNConv1d(channels, channels, kernel_size, dilation=d)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            WNConv1d(channels, channels, kernel_size, dilation=1)
            for _ in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            c.remove_weight_norm()
        for c in self.convs2:
            c.remove_weight_norm()


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            WNConv1d(channels, channels, kernel_size, dilation=d)
            for d in dilation
        ])

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs:
            c.remove_weight_norm()


class HifiganGenerator(nn.Module):
    """HiFiGAN Generator with Multi-Receptive Field Fusion (MRF).

    Args:
        in_channels: mel spectrogram channels (80)
        out_channels: waveform channels (1)
        resblock_type: "1" or "2"
        resblock_dilation_sizes: dilation per resblock layer
        resblock_kernel_sizes: kernel sizes for resblocks
        upsample_kernel_sizes: kernel sizes for upsampling layers
        upsample_initial_channel: channels for first upsample (halved each layer)
        upsample_factors: stride per upsample layer
        inference_padding: padding at inference time
    """

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        resblock_type="1",
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        resblock_kernel_sizes=(3, 7, 11),
        upsample_kernel_sizes=(16, 16, 4, 4),
        upsample_initial_channel=512,
        upsample_factors=(8, 8, 2, 2),
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)

        self.conv_pre = WNConv1d(in_channels, upsample_initial_channel, 7)

        ResBlock = ResBlock1 if resblock_type == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(WNConvTranspose1d(
                upsample_initial_channel // (2 ** i),
                upsample_initial_channel // (2 ** (i + 1)),
                k, stride=u, padding=(k - u) // 2,
            ))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = WNConv1d(ch, out_channels, 7, bias=conv_post_bias)

    def forward(self, x, g=None):
        o = self.conv_pre(x)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = F.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


def load_hifigan_vocoder(ckpt_path, device="cpu"):
    """Load pretrained HiFiGAN from checkpoint.

    Args:
        ckpt_path: path to generator.ckpt file
        device: torch device

    Returns:
        HifiganGenerator model in eval mode
    """
    vocoder = HifiganGenerator(
        in_channels=80,
        out_channels=1,
        resblock_type="1",
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=512,
        upsample_factors=[8, 8, 2, 2],
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)
    return vocoder
