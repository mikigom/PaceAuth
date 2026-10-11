# Vendored from CVLFace (https://github.com/mk-minchul/CVLface, MIT License, Copyright (c) 2022 Minchul Kim) as shipped in
# HF minchul/cvlface_adaface_vit_base_kprpe_webface12m@daefd5012d369588bd214fbaf4cc6b1d286e7066:
# models/vit_kprpe/{__init__.py, vit.py, rpe_options.py}, models/vit_kprpe/RPE/KPRPE/{kprpe_shared.py,
# relative_keypoints.py, dist.py}. kprpe_shared.py/dist.py derive from Microsoft iRPE
# (https://github.com/microsoft/Cream/tree/main/iRPE, MIT License, Copyright (c) Microsoft Corporation).
# Paper: Kim, Su, Liu, Jain, Liu, "KeyPoint Relative Position Encoding for Face Recognition", CVPR 2024.
#
# Changes (inference only; parameter names unchanged so the released state_dict loads with strict=True):
#  - only the configuration of the released ViT-B model is kept: KPRPE_shared, rpe_on=k, mode=ctx, method=product,
#    ratio=1.9, ctx_type=rel_keypoint_splithead_unshared, num_keypoints=5, mask_ratio=0;
#  - the iRPE module on keys has no parameters in this mode; its bucket gather is inlined in Attention using the
#    same pure-PyTorch indexing upstream falls back to when the optional compiled rpe_ops extension is absent
#    (the auto-build/sys.exit() logic of RPE/__init__.py is dropped);
#  - removed training-only paths (masking, checkpointing, DropPath/Dropout, init).
import math

import torch
from torch import nn


def _piecewise_index(relative_position, alpha, beta, gamma):
    """iRPE piecewise index function (Eq. 18), long tensors in [-beta, beta]."""
    rp_abs = relative_position.abs()
    not_mask = ~(rp_abs <= alpha)
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha + torch.log(rp_abs_out / alpha) / math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(torch.long)
    idx = relative_position.clone()
    idx[not_mask] = y_out
    return idx


def rpe_gather_index(side: int, ratio: float = 1.9):
    """iRPE 'product' bucket ids for a side x side token grid, offset per query row so that
    ctx.flatten(2)[:, :, idx].view(B, H, L, L)[b, h, i, j] == ctx[b, h, i, bucket(i, j)]. -> (idx (L*L,), num_buckets)"""
    alpha, beta, gamma = 1 * ratio, 2 * ratio, 8 * ratio
    beta_int = int(beta)
    s = 2 * beta_int + 1
    pos = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), 2).view(-1, 2)
    diff = pos[:, None] - pos[None]  # (L, L, 2) (row, col) offsets
    bucket = ((_piecewise_index(diff[..., 0], alpha, beta, gamma) + beta_int) * s +
              _piecewise_index(diff[..., 1], alpha, beta, gamma) + beta_int)
    num_buckets = s * s
    offset = torch.arange(0, side * side * num_buckets, num_buckets).view(-1, 1)
    return (bucket + offset).flatten(), num_buckets


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rel_keypoints, rpe_index):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + rel_keypoints.flatten(2)[:, :, rpe_index].view(b, -1, n, n)  # KP-RPE on keys
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, rel_keypoints, rpe_index):
        x = x + self.attn(self.norm1(x), rel_keypoints, rpe_index)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformerWithKPRPE(nn.Module):
    """forward(x (B,3,112,112) in [-1,1] RGB, keypoints (B,5,2) x,y normalised to [0,1] of the input) -> (B, dim)."""

    def __init__(self, img_size=112, patch_size=8, num_classes=512, embed_dim=512, depth=24, num_heads=16,
                 mlp_ratio=3.0, num_keypoints=5, ratio=1.9):
        super().__init__()
        side = img_size // patch_size
        self.num_patches, self.embed_dim, self.depth, self.num_heads = side * side, embed_dim, depth, num_heads
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.feature = nn.Sequential(
            nn.Linear(embed_dim * self.num_patches, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, eps=2e-5),
            nn.Linear(embed_dim, num_classes, bias=False),
            nn.BatchNorm1d(num_classes, eps=2e-5),
        )
        rpe_index, self.num_buckets = rpe_gather_index(side, ratio)
        self.register_buffer("rpe_index", rpe_index, persistent=False)
        self.keypoint_linear = nn.Linear(2 * num_keypoints, self.num_buckets * num_heads * depth)
        # patch centres (x, y) in [0, 1], as relative_keypoints.make_rel_keypoints
        coord = torch.linspace(0, 1, side + 1)
        coord = (coord[:-1] + coord[1:]) / 2
        gy, gx = torch.meshgrid(coord, coord, indexing="ij")
        self.register_buffer("grid", torch.stack([gx, gy], -1).reshape(1, -1, 1, 2), persistent=False)

    def forward(self, x, keypoints):
        b = x.shape[0]
        x = self.patch_embed(x) + self.pos_embed
        rel = (self.grid.to(x.dtype) - keypoints.to(x.dtype).unsqueeze(1)).flatten(2)  # (B, N, 2K)
        rel = self.keypoint_linear(rel).view(b, self.num_patches, self.num_heads * self.depth, self.num_buckets)
        for blk, ctx in zip(self.blocks, rel.transpose(1, 2).chunk(self.depth, dim=1)):
            x = blk(x, ctx, self.rpe_index)
        x = self.norm(x.float())
        return self.feature(x.reshape(b, self.num_patches * self.embed_dim))


def vit_base_kprpe(num_features: int = 512) -> VisionTransformerWithKPRPE:
    """ViTKPRPEModel.from_config(name='base') of the released WebFace4M/12M models."""
    return VisionTransformerWithKPRPE(num_classes=num_features)
