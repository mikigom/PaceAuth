# Minimal inference code for 6DRepNet and 6DRepNet360 (parameter names unchanged, upstream checkpoints load strictly).
# Vendored from:
#   https://github.com/thohemp/6DRepNet     @464b2ba  sixdrepnet/model.py, sixdrepnet/utils.py, sixdrepnet/backbone/repvgg.py
#   https://github.com/thohemp/6DRepNet360  @b06e4fe  sixdrepnet360/test.py (SixDRepNet360), sixdrepnet360/utils.py
#   (RepVGG architecture: https://github.com/DingXiaoH/RepVGG)
# Changes: RepVGG blocks only in their re-parameterised deploy form (what the released 6DRepNet checkpoint contains:
# one 3x3 conv + bias + ReLU per block; B1g2 has no SE); SixDRepNet360's hand-written ResNet-50 replaced by
# torchvision's ResNet with the same Bottleneck/[3,4,6,3] layout (AdaptiveAvgPool == AvgPool2d(7) at 224 input);
# utils rewritten with torch.nn.functional / torch.where (same maths, same eps).
#
# MIT License — Copyright (c) 2022 Thorsten Hempel (6DRepNet); Copyright (c) 2023 Thorsten Hempel (6DRepNet360);
# Copyright (c) 2020 DingXiaoH (RepVGG).
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import torch
import torch.nn.functional as F
import torchvision
from torch import nn


def rotmat_from_6d(p: torch.Tensor) -> torch.Tensor:
    """utils.compute_rotation_matrix_from_ortho6d: (n,6) -> (n,3,3) with columns x, y, z (Gram-Schmidt)."""
    x = F.normalize(p[:, 0:3], dim=1, eps=1e-8)
    z = F.normalize(torch.cross(x, p[:, 3:6], dim=1), dim=1, eps=1e-8)
    y = torch.cross(z, x, dim=1)
    return torch.stack((x, y, z), dim=2)


def euler_from_rotmat(R: torch.Tensor) -> torch.Tensor:
    """utils.compute_euler_angles_from_rotation_matrices: (n,3,3) -> (n,3) radians (x=pitch, y=yaw, z=roll),
    inverse of R = Rz(roll) @ Ry(yaw) @ Rx(pitch) (utils.get_R), in the 300W-LP/AFLW2000 label convention."""
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = sy < 1e-6
    x = torch.where(singular, torch.atan2(-R[:, 1, 2], R[:, 1, 1]), torch.atan2(R[:, 2, 1], R[:, 2, 2]))
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.where(singular, torch.zeros_like(sy), torch.atan2(R[:, 1, 0], R[:, 0, 0]))
    return torch.stack((x, y, z), dim=1)


class RepVGGBlock(nn.Module):
    """Deploy form of repvgg.RepVGGBlock (after switch_to_deploy)."""

    def __init__(self, cin: int, cout: int, stride: int, groups: int):
        super().__init__()
        self.rbr_reparam = nn.Conv2d(cin, cout, 3, stride, 1, groups=groups, bias=True)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        return self.nonlinearity(self.rbr_reparam(x))


class SixDRepNet(nn.Module):
    """model.SixDRepNet(backbone_name='RepVGG-B1g2', deploy=True): num_blocks [4,6,16,1], widths [128,256,512,2048],
    groups=2 on block indices 2,4,..,26 (g2_map), + linear 6D head -> rotation matrix."""

    def __init__(self):
        super().__init__()
        self.layer0 = RepVGGBlock(3, 64, 2, 1)
        cin, idx, stages = 64, 1, []
        for cout, n in ((128, 4), (256, 6), (512, 16), (2048, 1)):
            blocks = []
            for i in range(n):
                blocks.append(RepVGGBlock(cin, cout, 2 if i == 0 else 1, 2 if idx % 2 == 0 else 1))
                cin, idx = cout, idx + 1
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear_reg = nn.Linear(2048, 6)

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x)))))
        return rotmat_from_6d(self.linear_reg(torch.flatten(self.gap(x), 1)))


class SixDRepNet360(torchvision.models.ResNet):
    """test.SixDRepNet360(Bottleneck, [3,4,6,3]): ResNet-50 + linear 6D head -> rotation matrix."""

    def __init__(self):
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        del self.fc
        self.linear_reg = nn.Linear(2048, 6)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return rotmat_from_6d(self.linear_reg(torch.flatten(self.avgpool(x), 1)))


ARCHS = {"6drepnet_repvgg_b1g2": SixDRepNet, "6drepnet360_resnet50": SixDRepNet360}
