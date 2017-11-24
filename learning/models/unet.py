
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from learning.models.utils import *


class unet(nn.Module):


    def __init__(self, n_classes=2, is_batchnorm=True, in_channels=3, is_deconv=True, dropout=0.2, patch_size=64):
        super(unet, self).__init__()
        
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.dropout = dropout
        self.patch_size=patch_size

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


    def predict_from_full_image(self, image):
        
        # initialize an empty segmentation
        segmentation_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        segmentation = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        # initialize the pad
        pad = int(self.patch_size/2)

        # loop for every patch in the image    
        for i in range(pad, image.shape[0] - pad, self.patch_size):
            for j in range(pad, image.shape[1] - pad, self.patch_size):
                
                # get current patch
                current_patch = image[i-pad:i+pad, j-pad:j+pad, :]
                current_patch = (current_patch - np.mean(current_patch)) / np.std(current_patch) # normalize by its own mean and standard deviation

                current_patch = torch.from_numpy(current_patch).float()
                current_patch = torch.unsqueeze(current_patch, 0)
                current_patch = current_patch.permute(0, 3, 1, 2)

                if torch.cuda.is_available():
                    current_patch = Variable(current_patch, volatile=True).cuda()
                else:
                    current_patch = Variable(current_patch, volatile=True)

                # get the score map and assign to the position in the array
                scores = self.forward(current_patch)
                segmentation[i-pad:i+pad, j-pad:j+pad] = scores.data.max(1)[1].cpu().numpy()

                m = nn.Softmax2d()
                segmentation_scores[i-pad:i+pad, j-pad:j+pad] = m(scores).data[0][1].cpu().numpy()

        return segmentation_scores, segmentation
