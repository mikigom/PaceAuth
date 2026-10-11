# Vendored from CVLFace (https://github.com/mk-minchul/CVLface, MIT License, Copyright (c) 2022 Minchul Kim) as shipped in
# HF minchul/cvlface_adaface_ir101_webface12m@54f602a0737bd1ee4a4e7e9fd089a485f397fefd: models/iresnet/model.py
# (Backbone / BasicBlockIR / IR_101, originally from the AdaFace repository, MIT License) and models/iresnet/__init__.py
# (IResNetModel: color_space RGB -> no channel flip; test transform ToTensor + Normalize(0.5, 0.5)).
# Paper: Kim, Jain, Liu, "AdaFace: Quality Adaptive Margin for Face Recognition", CVPR 2022.
#
# Changes (inference only; module indices and parameter names unchanged so the released state_dict loads with
# strict=True after stripping its "model.net." prefix):
#  - only the IR (BasicBlockIR, 112x112 input) path used by ir101 is kept; SE / bottleneck / 224 variants, flop
#    counting (fvcore) and weight init are dropped;
#  - Dropout(0.4) in output_layer is replaced by Identity (it is a no-op in eval mode; index 1 is kept).
from torch import nn


class BasicBlockIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                                                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class Backbone(nn.Module):
    """AdaFace/CVLFace IR backbone for 112x112 RGB input normalised to [-1, 1] -> (n, output_dim) raw features
    (BatchNorm1d without affine; not L2-normalised)."""

    def __init__(self, units=(3, 13, 30, 3), output_dim: int = 512):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        modules, in_c = [], 64
        for depth, n in zip((64, 128, 256, 512), units):
            modules.append(BasicBlockIR(in_c, depth, 2))
            modules += [BasicBlockIR(depth, depth, 1) for _ in range(n - 1)]
            in_c = depth
        self.body = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Identity(), nn.Flatten(),
                                          nn.Linear(512 * 7 * 7, output_dim), nn.BatchNorm1d(output_dim, affine=False))

    def forward(self, x):
        return self.output_layer(self.body(self.input_layer(x)))


def ir_101(output_dim: int = 512) -> Backbone:
    """IR_101 = Backbone(num_layers=100, mode='ir'): units (3, 13, 30, 3)."""
    return Backbone((3, 13, 30, 3), output_dim)
