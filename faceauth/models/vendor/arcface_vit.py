# Vendored from deepinsight/insightface, recognition/arcface_torch/backbones/vit.py and backbones/__init__.py
# (commit 81929474ec02e54e3655f6841bced800009c592d). InsightFace code is released under the MIT License
# (repository README: "The code of InsightFace is released under the MIT License"). The pretrained weights are
# NOT covered by that: "available for non-commercial research purposes only".
#
# Changes (inference only; parameter names unchanged so upstream state_dicts load with strict=True):
#  - removed training-only paths: random masking, gradient checkpointing, DropPath/Dropout, "bn" norm option, init;
#  - upstream hard-codes torch.cuda.amp.autocast(True) around qkv/proj/MLP, i.e. it runs mixed precision even when
#    the caller asks for fp32. Here those regions follow the caller's autocast state (fp32 stays fp32; an fp16
#    autocast context gives upstream's mixed precision). Upstream's fp32 attention core (autocast off, .float()) is kept.
import torch
from torch import nn


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

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        with torch.autocast(x.device.type, enabled=False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=112, patch_size=9, num_classes=512, embed_dim=768, depth=24, num_heads=8,
                 mlp_ratio=4.0):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
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
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # training-only; kept for strict loading

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x.float())
        return self.feature(x.reshape(x.shape[0], self.num_patches * self.embed_dim))


def vit_l_dp005_mask_005(num_features: int = 512) -> VisionTransformer:
    """backbones.get_model("vit_l_dp005_mask_005") in eval mode (drop_path 0.05 / mask_ratio 0.05 act only in training)."""
    return VisionTransformer(img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
                             num_heads=8)
