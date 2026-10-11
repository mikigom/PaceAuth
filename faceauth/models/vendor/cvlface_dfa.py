# Vendored from the CVLFace DFA aligner (HF minchul/cvlface_DFA_mobilenet@8317e6dda53d91e7074979923144c2cc08906a33,
# aligners/differentiable_face_aligner/{__init__.py, aligner_helper.py, dfa/...}; also shipped as
# pretrained_model/aligner.pt in the KP-RPE repos). CVLFace: MIT License, Copyright (c) 2022 Minchul Kim.
# The RetinaFace mobilenet0.25 network (dfa/models/{net.py, retinaface.py}, dfa/layers/functions/prior_box.py) derives
# from biubug6/Pytorch_Retinaface (MIT License).
#
# Changes (inference only; parameter names unchanged): returns only the aggregated 5-point landmarks, normalised (x, y)
# in [0, 1] of the (square) input -- the `orig_pred_ldmks` the official KP-RPE eval pipeline
# (InferAlignerKeypointModelPipeline) feeds the recognition model. The differentiable warp, padding options and
# training code are dropped.
from itertools import product
from math import ceil

import torch
import torch.nn.functional as F
from timm.models.mlp_mixer import MixerBlock
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


def conv_bn(inp, oup, stride=1, leaky=0.0):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
                         nn.LeakyReLU(negative_slope=leaky, inplace=True))


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))


def conv_bn1x1(inp, oup, stride, leaky=0.0):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False), nn.BatchNorm2d(oup),
                         nn.LeakyReLU(negative_slope=leaky, inplace=True))


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        leaky = 0.1 if out_channel <= 64 else 0.0
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, x):
        c5_1 = self.conv5X5_1(x)
        return F.relu(torch.cat([self.conv3X3(x), self.conv5X5_2(c5_1), self.conv7x7_3(self.conv7X7_2(c5_1))], 1))


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, x):
        x = list(x.values())
        o1, o2, o3 = self.output1(x[0]), self.output2(x[1]), self.output3(x[2])
        o2 = self.merge2(o2 + F.interpolate(o3, size=o2.shape[2:], mode="nearest"))
        o1 = self.merge1(o1 + F.interpolate(o2, size=o1.shape[2:], mode="nearest"))
        return [o1, o2, o3]


class MobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(conv_bn(3, 8, 2, leaky=0.1), conv_dw(8, 16, 1), conv_dw(16, 32, 2),
                                    conv_dw(32, 32, 1), conv_dw(32, 64, 2), conv_dw(64, 64, 1))
        self.stage2 = nn.Sequential(conv_dw(64, 128, 2), *[conv_dw(128, 128, 1) for _ in range(5)])
        self.stage3 = nn.Sequential(conv_dw(128, 256, 2), conv_dw(256, 256, 1))


def _heads(out, ch, num_anchors=2, fpn_num=3):
    return nn.ModuleList([nn.ModuleDict({"conv1x1": nn.Conv2d(ch, num_anchors * out, 1)}) for _ in range(fpn_num)])


def _priors(size=160, min_sizes=((64, 80), (96, 112), (128, 144)), steps=(8, 16, 32)):
    """PriorBox.forward: (num_priors, 4) normalised (cx, cy, w, h)."""
    anchors = []
    for k, step in enumerate(steps):
        f = ceil(size / step)
        for i, j in product(range(f), range(f)):
            for m in min_sizes[k]:
                anchors += [(j + 0.5) * step / size, (i + 0.5) * step / size, m / size, m / size]
    return torch.tensor(anchors).view(-1, 4)


class DFALandmarks(nn.Module):
    """RetinaFace mobilenet0.25 + MLP-Mixer aggregator over all priors (CVLFace DFA, input 160, no padding).

    forward(x (B,3,H,H) RGB in [-1,1]) -> (B,5,2) landmarks, (x, y) normalised to [0,1] of the input image.
    """

    def __init__(self, input_size: int = 160, variances=(0.1, 0.2)):
        super().__init__()
        self.input_size, self.variances = input_size, variances
        self.body = IntermediateLayerGetter(MobileNetV1(), {"stage1": 1, "stage2": 2, "stage3": 3})
        self.fpn = FPN([64, 128, 256], 64)
        self.ssh1, self.ssh2, self.ssh3 = SSH(64, 64), SSH(64, 64), SSH(64, 64)
        self.ClassHead, self.BboxHead, self.LandmarkHead = _heads(2, 64), _heads(4, 64), _heads(10, 64)
        priors = _priors(input_size)
        self.aggregator = nn.Sequential(*[MixerBlock(16, len(priors)) for _ in range(3)], nn.Linear(16, 1))
        self.register_buffer("priors", priors, persistent=False)

    @staticmethod
    def _cat(heads, feats, out):
        return torch.cat([h["conv1x1"](f).permute(0, 2, 3, 1).reshape(f.shape[0], -1, out) for h, f in
                          zip(heads, feats)], 1)

    def forward(self, x):
        if x.shape[2] != x.shape[3]:
            raise ValueError("DFA landmarks need a square input")
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=True)
        x = x.flip(1)  # the network takes BGR
        fpn = self.fpn(self.body(x))
        feats = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        loc, cls, ldm = self._cat(self.BboxHead, feats, 4), self._cat(self.ClassHead, feats, 2), \
            self._cat(self.LandmarkHead, feats, 10)
        p = self.priors.unsqueeze(0).expand(x.shape[0], -1, -1)
        v0, v1 = self.variances
        boxes = torch.cat((p[..., :2] + loc[..., :2] * v0 * p[..., 2:], p[..., 2:] * torch.exp(loc[..., 2:] * v1)), -1)
        boxes = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1)
        ldmk = torch.cat([p[..., :2] + ldm[..., 2 * i:2 * i + 2] * v0 * p[..., 2:] for i in range(5)], -1)
        combined = torch.cat([boxes, cls, ldmk], 2)  # raw class logits, as upstream
        weight = F.softmax(self.aggregator(combined), dim=1)
        return torch.sum(weight * combined, dim=1)[:, 6:].view(-1, 5, 2)
