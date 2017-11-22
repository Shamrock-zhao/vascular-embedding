import torch.nn as nn
import torch.nn.functional as F
from learning.models.utils import *

class unet(nn.Module):


    def __init__(self, n_classes=2, is_batchnorm=True, in_channels=3, is_deconv=True, dropout=0.2):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.dropout = dropout

        filters = [32, 64, 128]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, self.dropout)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, self.dropout)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, self.dropout)

        # upsampling
        self.up_concat2 = unetUp(filters[2] + filters[1], filters[1], self.is_deconv, self.dropout)
        self.up_concat1 = unetUp(filters[1] + filters[0], filters[0], self.is_deconv, 0.0)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)



    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)

        up2 = self.up_concat2(conv2, conv3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final



