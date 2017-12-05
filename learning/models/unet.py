
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        
        # initialize the pad
        pad = int(self.patch_size/2)

        mean_image = np.mean(image)
        std_image = np.std(image)
        
        size_x = math.ceil(image.shape[0] / self.patch_size) * self.patch_size
        size_y = math.ceil(image.shape[1] / self.patch_size) * self.patch_size
        
        # initialize matrices for the segmentations and the padded image
        segmentation_scores = np.zeros((size_x, size_y), dtype=np.float32)
        segmentation = np.zeros(segmentation_scores.shape, dtype=np.float32)
        unary_potentials = np.zeros((2, size_x, size_y), dtype=np.float32)
        padded_image = np.zeros((size_x, size_y, 3), dtype=np.uint8)
        # pad the image
        padded_image[0:image.shape[0], 0:image.shape[1], :] = image

        # loop for every patch in the image    
        #increment = self.patch_size
        increment = self.patch_size
        m = nn.Softmax2d()
        for i in range(pad, image.shape[0] - pad, increment):
            for j in range(pad, image.shape[1] - pad, increment):
                
                # get current patch
                current_patch = np.asarray(padded_image[i-pad:i+pad, j-pad:j+pad, :], dtype=np.float32)
                # normalize by the image mean and standard deviation
                current_patch = (current_patch - np.mean(current_patch)) / (np.std(current_patch) + 0.000001)
                #print(str(i) + '-' + str(j))
                #print(current_patch.shape)

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

                segmentation_scores[i-pad:i+pad, j-pad:j+pad] = m(scores).data[0][1].cpu().numpy()

                unary_potentials[:,i-pad:i+pad, j-pad:j+pad] = m(scores).data[0].cpu().numpy()

        # unpad the segmentations
        segmentation_scores = segmentation_scores[0:image.shape[0], 0:image.shape[1]]
        segmentation = segmentation[0:image.shape[0], 0:image.shape[1]]
        unary_potentials = unary_potentials[:,0:image.shape[0], 0:image.shape[1]]

        return segmentation_scores, segmentation, unary_potentials



    def efficient_prediction_from_full_image(self, image):
        
        # initialize the pad
        pad = int(self.patch_size/2)

        mean_image = np.mean(image)
        std_image = np.std(image)
        
        size_x = math.ceil(image.shape[0] / self.patch_size) * self.patch_size
        size_y = math.ceil(image.shape[1] / self.patch_size) * self.patch_size
        
        # initialize matrices for the segmentations and the padded image
        segmentation_scores = np.zeros((size_x, size_y), dtype=np.float32)
        segmentation = np.zeros(segmentation_scores.shape, dtype=np.float32)
        unary_potentials = np.zeros((size_x, size_y, 2), dtype=np.float32)
        padded_image = np.zeros((size_x, size_y, 3), dtype=np.uint8)
        # pad the image
        padded_image[0:image.shape[0], 0:image.shape[1], :] = image

        length_patch_block = 1024
        patch_block = np.empty((self.patch_size, self.patch_size, image.shape[2], length_patch_block), dtype=np.float32)
        pos_i = np.empty((length_patch_block), dtype=int)
        pos_j = np.empty((length_patch_block), dtype=int)
        filling_iterator = 0
        last_pos_i = pad
        last_pos_j = pad

        for i in range(pad, padded_image.shape[0] - pad, 1):
            for j in range(pad, padded_image.shape[1] - pad, 1):

                if filling_iterator < length_patch_block:

                    current_patch = np.asarray(padded_image[i-pad:i+pad, j-pad:j+pad, :], dtype=np.float32)
                    patch_block[:,:,:,filling_iterator] = (current_patch - np.mean(current_patch)) / (np.std(current_patch) + 0.000001)
                    pos_i[filling_iterator] = i
                    pos_j[filling_iterator] = j
                    filling_iterator = filling_iterator + 1

                elif filling_iterator == length_patch_block:

                    current_patch = torch.from_numpy(patch_block).float()
                    current_patch = current_patch.permute(3,2,0,1)
                    if torch.cuda.is_available():
                        current_patch = Variable(current_patch, volatile=True).cuda()
                    else:
                        current_patch = Variable(current_patch, volatile=True)    
                    
                    scores = self.forward(current_patch)
                    
                    m = nn.Softmax2d()
                    up = m(scores).data.cpu().permute(2,3,1,0).numpy()
                    scores = up[:,:,1,:]
                    segm = scores > 0.5

                    for iterator in range(0, length_patch_block):
                        sub_i = pos_i[iterator]
                        sub_j = pos_j[iterator]
                        segmentation_scores[sub_i-pad:sub_i+pad, sub_j-pad:sub_j+pad] = scores[:,:,iterator]
                        segmentation[sub_i-pad:sub_i+pad, sub_j-pad:sub_j+pad] = segm[:,:,iterator]
                        unary_potentials[sub_i-pad:sub_i+pad, sub_j-pad:sub_j+pad, :] = up[:,:,:,iterator]

                    last_pos_i = i
                    last_pos_j = j
                    filling_iterator = 0
                    patch_block = np.empty((self.patch_size, self.patch_size, image.shape[2], length_patch_block), dtype=np.float32)

        if filling_iterator < length_patch_block:
            
            current_patch = torch.from_numpy(patch_block).float()
            current_patch = current_patch.permute(3,2,0,1)
            if torch.cuda.is_available():
                current_patch = Variable(current_patch, volatile=True).cuda()
            else:
                current_patch = Variable(current_patch, volatile=True)    
            scores = self.forward(current_patch)
            
            m = nn.Softmax2d()
            up = np.asarray(m(scores).data.cpu().permute(2,3,1,0).numpy(), dtype=np.float32)
            scores = up[:,:,1,:]
            segm = scores > 0.5

            iterator = 0
            for iterator in range(0, filling_iterator):
                sub_i = pos_i[iterator]
                sub_j = pos_j[iterator]
                segmentation_scores[sub_i-pad:sub_i+pad, sub_j-pad:sub_j+pad] = scores[:,:,iterator]
                segmentation[sub_i-pad:sub_i+pad, sub_j-pad:sub_j+pad] = segm[:,:,iterator]
                unary_potentials[sub_i-pad:sub_i+pad, sub_j-pad:sub_j+pad, :] = up[:,:,:,iterator]
            
        segmentation_scores = segmentation_scores[0:image.shape[0], 0:image.shape[1]]
        segmentation = segmentation[0:image.shape[0], 0:image.shape[1]]
        unary_potentials = unary_potentials[0:image.shape[0], 0:image.shape[1], :]

        return segmentation_scores, segmentation, unary_potentials