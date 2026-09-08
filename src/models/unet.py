"""Small U-Net for watermark segmentation. No external arch dependencies.

Encoder: ResNet34 from torchvision (ImageNet weights on Kaggle, random init
fallback offline). Decoder: plain up-conv blocks with skip connections.
Input 256x256 RGB, output single-channel mask logits.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class WatermarkUNet(nn.Module):
    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT if pretrained_encoder else None
            enc = resnet34(weights=weights)
        except Exception:
            from torchvision.models import resnet34
            enc = resnet34(weights=None)
        self.stem_first = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.stem_pool = enc.maxpool
        self.e1, self.e2, self.e3, self.e4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.d1 = DoubleConv(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d0 = DoubleConv(32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.stem_first(x)  # /2
        s0 = self.stem_pool(c1)  # /4
        s1 = self.e1(s0)     # /4
        s2 = self.e2(s1)     # /8
        s3 = self.e3(s2)     # /16
        s4 = self.e4(s3)     # /32
        d = self.d4(torch.cat([self.up4(s4), s3], 1))
        d = self.d3(torch.cat([self.up3(d), s2], 1))
        d = self.d2(torch.cat([self.up2(d), s1], 1))
        d = self.d1(torch.cat([self.up1(d), c1], 1))
        d = self.d0(self.up0(d))
        return self.out(d)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)
