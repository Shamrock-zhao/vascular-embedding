
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, batch_norm=False):
        super(UNet, self).__init__()
        self.inc = input_block(n_channels, 64, batch_norm)
        self.down1 = down_block(64, 128, batch_norm)
        self.down2 = down_block(128, 256, batch_norm)
        self.down3 = down_block(256, 512, batch_norm)
        self.down4 = down_block(512, 512, batch_norm)
        self.up1 = up_block(1024, 256, batch_norm)
        self.up2 = up_block(512, 128, batch_norm)
        self.up3 = up_block(256, 64, batch_norm)
        self.up4 = up_block(128, 64, batch_norm)
        self.outc = output_block(64, n_classes, batch_norm)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x