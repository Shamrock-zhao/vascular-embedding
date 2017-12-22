
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable
from learning.models.utils import *
from skimage import filters


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
        
        # initialize the pad
        sub_pad = self.patch_size // 4
        pad = self.patch_size // 2
        
        # sub_pad + (image extended a little bit to fit) + sub_pad
        size_x = math.ceil(image.shape[0] / self.patch_size) * self.patch_size + 2 * pad + (image.shape[0] % self.patch_size)
        size_y = math.ceil(image.shape[1] / self.patch_size) * self.patch_size + 2 * pad + (image.shape[1] % self.patch_size)
        
        # initialize matrices for the segmentations and the padded image
        segmentation_scores = np.zeros((size_x, size_y), dtype=np.float32)
        unary_potentials = np.zeros((2, size_x, size_y), dtype=np.float32)
        padded_image = np.zeros((size_x, size_y, 3), dtype=np.uint8)
        # pad the image
        padded_image[pad:image.shape[0]+pad, pad:image.shape[1]+pad, :] = image

        m = nn.Softmax2d()

        increment = self.patch_size - 2 * sub_pad
        for i in range(pad, padded_image.shape[0] - pad, increment):
            for j in range(pad, padded_image.shape[1] - pad, increment):

                # get current patch
                current_patch = np.asarray(padded_image[i-pad:i+pad, j-pad:j+pad, :], dtype=np.float32)
                # normalize by the image mean and standard deviation
                current_patch = (current_patch - np.mean(current_patch)) / (np.std(current_patch) + 0.000001)
                # prepare data for pytorch
                current_patch = torch.from_numpy(current_patch).float()
                current_patch = torch.unsqueeze(current_patch, 0)
                current_patch = current_patch.permute(0, 3, 1, 2)
                # use CUDA if possible
                if torch.cuda.is_available():
                    current_patch = Variable(current_patch, volatile=True).cuda()
                else:
                    current_patch = Variable(current_patch, volatile=True)
                # get the scores doing a forward pass
                scores = self.forward(current_patch)
                # get the scores and the unary potentials
                scores_patch = m(scores).data[0][1].cpu().numpy()
                up_patch = m(scores).data[0].cpu().numpy()
                # assign the inner part
                segmentation_scores[i-sub_pad:i+sub_pad, j-sub_pad:j+sub_pad] = scores_patch[sub_pad:self.patch_size-sub_pad, sub_pad:self.patch_size-sub_pad]
                unary_potentials[:,i-sub_pad:i+sub_pad, j-sub_pad:j+sub_pad] = up_patch[:, sub_pad:self.patch_size-sub_pad, sub_pad:self.patch_size-sub_pad]

        # unpad the segmentations
        segmentation_scores = segmentation_scores[pad:image.shape[0]+pad, pad:image.shape[1]+pad]
        val = filters.threshold_otsu(segmentation_scores)
        segmentation = np.asarray(segmentation_scores > val, dtype=np.float32)
        unary_potentials = unary_potentials[:, pad:image.shape[0]+pad, pad:image.shape[1]+pad]

        return segmentation_scores, segmentation, unary_potentials


class deeper_unet(nn.Module):
    

    def __init__(self, n_classes=2, is_batchnorm=True, in_channels=3, is_deconv=True, dropout=0.2, patch_size=64):
        super(deeper_unet, self).__init__()
        
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.dropout = dropout
        self.patch_size=patch_size

        filters = [32, 32, 64, 64, 128, 128]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, self.dropout)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, self.dropout)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, self.dropout)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, self.dropout)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm, self.dropout)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = unetConv2(filters[4], filters[5], self.is_batchnorm, self.dropout)

        # upsampling
        self.up_concat5 = unetUp(filters[5] + filters[4], filters[4], self.is_deconv, self.dropout)
        self.up_concat4 = unetUp(filters[4] + filters[3], filters[3], self.is_deconv, self.dropout)
        self.up_concat3 = unetUp(filters[3] + filters[2], filters[2], self.is_deconv, self.dropout)
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
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        conv5 = self.conv5(maxpool4)
        maxpool5 = self.maxpool5(conv5)
        
        conv6 = self.conv6(maxpool5)

        up5 = self.up_concat5(conv5, conv6)
        up4 = self.up_concat4(conv4, up5)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final


    def predict_from_full_image(self, image):
            
        # initialize the pad
        sub_pad = self.patch_size // 4
        pad = self.patch_size // 2
        
        # sub_pad + (image extended a little bit to fit) + sub_pad
        size_x = math.ceil(image.shape[0] / self.patch_size) * self.patch_size + 2 * pad + (image.shape[0] % self.patch_size)
        size_y = math.ceil(image.shape[1] / self.patch_size) * self.patch_size + 2 * pad + (image.shape[1] % self.patch_size)
        
        # initialize matrices for the segmentations and the padded image
        segmentation_scores = np.zeros((size_x, size_y), dtype=np.float32)
        unary_potentials = np.zeros((2, size_x, size_y), dtype=np.float32)
        padded_image = np.zeros((size_x, size_y, 3), dtype=np.uint8)
        # pad the image
        padded_image[pad:image.shape[0]+pad, pad:image.shape[1]+pad, :] = image

        m = nn.Softmax2d()

        increment = self.patch_size - 2 * sub_pad
        for i in range(pad, padded_image.shape[0] - pad, increment):
            for j in range(pad, padded_image.shape[1] - pad, increment):

                # get current patch
                current_patch = np.asarray(padded_image[i-pad:i+pad, j-pad:j+pad, :], dtype=np.float32)
                # normalize by the image mean and standard deviation
                current_patch = (current_patch - np.mean(current_patch)) / (np.std(current_patch) + 0.000001)
                # prepare data for pytorch
                current_patch = torch.from_numpy(current_patch).float()
                current_patch = torch.unsqueeze(current_patch, 0)
                current_patch = current_patch.permute(0, 3, 1, 2)
                # use CUDA if possible
                if torch.cuda.is_available():
                    current_patch = Variable(current_patch, volatile=True).cuda()
                else:
                    current_patch = Variable(current_patch, volatile=True)
                # get the scores doing a forward pass
                scores = self.forward(current_patch)
                # get the scores and the unary potentials
                scores_patch = m(scores).data[0][1].cpu().numpy()
                up_patch = m(scores).data[0].cpu().numpy()
                # assign the inner part
                segmentation_scores[i-sub_pad:i+sub_pad, j-sub_pad:j+sub_pad] = scores_patch[sub_pad:self.patch_size-sub_pad, sub_pad:self.patch_size-sub_pad]
                unary_potentials[:,i-sub_pad:i+sub_pad, j-sub_pad:j+sub_pad] = up_patch[:, sub_pad:self.patch_size-sub_pad, sub_pad:self.patch_size-sub_pad]

        # unpad the segmentations
        segmentation_scores = segmentation_scores[pad:image.shape[0]+pad, pad:image.shape[1]+pad]
        val = filters.threshold_otsu(segmentation_scores)
        segmentation = np.asarray(segmentation_scores > val, dtype=np.float32)
        unary_potentials = unary_potentials[:, pad:image.shape[0]+pad, pad:image.shape[1]+pad]

        return segmentation_scores, segmentation, unary_potentials