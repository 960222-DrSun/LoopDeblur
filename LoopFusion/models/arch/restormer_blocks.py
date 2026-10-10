"""Transformer building blocks (Restormer-style) used as the CDIMamba backbone.

Paper: "Learning Blur Directional Bias with Cross-Directional Interactive Mamba
for Image Deblurring" (ICASSP), Sec. 2.1 and Fig. 2.

CDIMamba adopts **Restormer as the backbone**: a head convolution extracts the
shallow feature ``F0``, Restormer blocks produce the multi-scale features
``{F1, F2, F3}``, each scale is refined by CDIMM followed by CDCEM, and a tail
convolution outputs the residual ``R`` added back to the blurry input.

This file is a compact, dependency-light re-implementation of the Restormer
blocks (MDTA + GDFN) so that the template runs without ``basicsr``'s optional
``restormer`` arch package.
"""

import numbers
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    """LayerNorm without learnable bias (as in Restormer)."""

    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        # x: (B, C, H, W) or (B, H*W, C)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        return x / torch.sqrt(var + 1e-6) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        return x * self.weight + self.bias


class LayerNorm(nn.Module):
    """LayerNorm over the channel dimension; supports (B,C,H,W) and (B,HW,C)."""

    def __init__(self, normalized_shape, bias=False):
        super().__init__()
        self.body = BiasFreeLayerNorm(normalized_shape) if not bias \
            else WithBiasLayerNorm(normalized_shape)

    def forward(self, x):
        if x.dim() == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        return self.body(x)


class FeedForward(nn.Module):
    """GDFN: gated-depthwise feed-forward network."""

    def __init__(self, dim, mult=4, bias=False):
        super().__init__()
        hidden = int(dim * mult)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(self.act(x1) * x2)


class Attention(nn.Module):
    """MDTA: multi-Dconv head transposed attention.

    Complexity is linear in ``H*W`` because the transposed attention is computed
    across the *channel* dimension rather than across spatial positions.
    """

    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return self.project_out(rearrange(out, 'b head c (h w) -> b (head c) h w',
                                          head=self.num_heads, h=h, w=w))


class RestormerBlock(nn.Module):
    """Pre-normalised MDTA + GDFN residual block."""

    def __init__(self, dim, num_heads=4, ffn_expansion=2.66, bias=False, **kwargs):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=bias)
        self.attn = Attention(dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm(dim, bias=bias)
        self.ffn = FeedForward(dim, mult=ffn_expansion, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


@ARCH_REGISTRY.register()
class Downsample(nn.Module):
    """Pixel-unshuffle based downsampling."""

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


@ARCH_REGISTRY.register()
class Upsample(nn.Module):
    """Pixel-shuffle based upsampling."""

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


@ARCH_REGISTRY.register()
class RestormerBackbone(nn.Module):
    """Multi-scale Restormer encoder-decoder producing ``{F1, F2, F3}``.

    Returns the list of encoder features at each level so that the CDIMamba
    network can insert CDIMM / CDCEM at every scale.
    """

    def __init__(self, dim=48, num_blocks=(4, 6, 6), num_heads=(1, 2, 4),
                 num_refinement_blocks=4, ffn_expansion=2.66):
        super().__init__()
        self.dim = dim
        self.num_levels = 3

        # ---- encoder ---------------------------------------------------------
        self.encoder_level1 = nn.Sequential(OrderedDict([
            (f'block{i + 1}', RestormerBlock(dim, num_heads[0], ffn_expansion))
            for i in range(num_blocks[0])
        ]))
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(OrderedDict([
            (f'block{i + 1}', RestormerBlock(dim * 2, num_heads[1], ffn_expansion))
            for i in range(num_blocks[1])
        ]))
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(OrderedDict([
            (f'block{i + 1}', RestormerBlock(dim * 4, num_heads[2], ffn_expansion))
            for i in range(num_blocks[2])
        ]))

        # ---- decoder ---------------------------------------------------------
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = nn.Sequential(OrderedDict([
            (f'block{i + 1}', RestormerBlock(dim * 2, num_heads[1], ffn_expansion))
            for i in range(num_blocks[1])
        ]))
        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 4, dim, 1, bias=False)
        self.decoder_level1 = nn.Sequential(OrderedDict([
            (f'block{i + 1}', RestormerBlock(dim, num_heads[0], ffn_expansion))
            for i in range(num_blocks[0])
        ]))

        self.refinement = nn.Sequential(OrderedDict([
            (f'block{i + 1}', RestormerBlock(dim, num_heads[0], ffn_expansion))
            for i in range(num_refinement_blocks)
        ]))
        self.head = nn.Conv2d(3, dim, 3, padding=1, bias=False)
        self.tail = nn.Conv2d(dim, 3, 3, padding=1, bias=False)

    def forward(self, x):
        """Returns the multi-scale feature list ``[F1, F2, F3]`` (F1 coarsest)."""
        f0 = self.head(x)
        e1 = self.encoder_level1(f0)          # H   x W   , dim
        e2 = self.encoder_level2(self.down1_2(e1))  # H/2 x W/2 , 2*dim
        e3 = self.encoder_level3(self.down2_3(e2))  # H/4 x W/4 , 4*dim
        return [e3, e2, e1]

    def decode(self, feats):
        """Decode ``[F3, F2, F1]`` (any order matching the encoder) to a residual."""
        e3, e2, e1 = feats
        d2 = self.decoder_level2(self.reduce_chan_level2(
            torch.cat([self.up3_2(e3), e2], dim=1)))
        d1 = self.decoder_level1(self.reduce_chan_level1(
            torch.cat([self.up2_1(d2), e1], dim=1)))
        return self.tail(self.refinement(d1))
