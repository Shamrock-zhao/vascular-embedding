
# Blocks of the U-net

import torch
import torch.nn as nn
import torch.nn.functional as F



class double_conv(nn.Module):
    '''
    Two convolutions with 3x3 filters, followed by max pooling
    and with optional batch normalization
    '''

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(double_conv, self).__init__()

        if batch_norm:
            # CONV2D - BN - ReLU - CONV2D - BN - ReLU
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        else:
            # CONV2D - ReLU - CONV2D - ReLU
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x



class input_block(nn.Module):
    '''
    Input block of the U-net. Two subsequent convolutions
    '''

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(input_block, self).__init__()
        self.conv = double_conv(in_channels, out_channels, batch_norm)

    def forward(self, x):
        x = self.conv(x)
        return x



class down_block(nn.Module):
    '''
    Downsampling block of the U-net. We take a single input, which is
    the feature map of the previous layer, and we perform two subsequent
    convolutions on it
    '''

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(down_block, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels, batch_norm)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x



class up_block(nn.Module):
    '''
    Upsampling block of the U-net. We take two inputs: the output of the
    previous convolution and the current level convolution. Then we
    upsample the previous convolution to match current resolution and
    convolve on the concatenation of their feature maps.
    '''

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(up_block, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = double_conv(in_channels, out_channels, batch_norm)

    def forward(self, x1, x2):
        # Upsampling2D
        x1 = self.up(x1)
        # Cropping2D
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        # Concatenate both inputs
        x = torch.cat([x2, x1], dim=1)
        # Convolve in them
        x = self.conv(x)
        return x
        


class output_block(nn.Module):
    '''
    Output block. Convolution with 1x1 filters
    '''

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(output_block, self).__init__()
        self.conv = double_conv(in_channels, out_channels, batch_norm)

    def forward(self, x):
        x = self.conv(x)
        return x