import torch
import torch.nn as nn
import torch.nn.functional as F



class unetConv2(nn.Module):
    
    def __init__(self, in_size, out_size, is_batchnorm, dropout):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU())
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU())
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        if not (self.drop is None):
            outputs = self.drop(outputs)
        outputs = self.conv2(outputs)
        return outputs



class unetUp(nn.Module):
    
    def __init__(self, in_size, out_size, is_deconv, dropout):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False, dropout)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))



class flatten(nn.Module):
    
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class vascularEmbedding(nn.Module):
    
    def __init__(self):
        super(vascularEmbedding, self).__init__()

        # to flatten the outputs of the convolutional layer
        self.toReconstruct = flatten()
        # to map the flattened layer to a linear layer
        self.encoder1 = nn.Sequential(nn.Linear(16*16*128, 2048), nn.ReLU())
        # this layer creates the encoding
        self.fullyEncoded = nn.Sequential(nn.Linear(2048, 128), nn.ReLU())
        # this restores the 2048 features
        self.decoder1 = nn.Sequential(nn.Linear(128, 2048), nn.ReLU())
        # this restores the original output of conv3
        self.decoderOutput = nn.Sequential(nn.Linear(2048, 16*16*128), nn.ReLU())

    def forward(self, conv3_output):
        
        # flatten the input
        smallFlatConv = self.toReconstruct(conv3_output)
        # encode it
        encoded = self.encoder1(smallFlatConv)
        encoded = self.fullyEncoded(encoded)
        # decode it
        decoded = self.decoder1(encoded)
        decoded = self.decoderOutput(decoded)
        # reshape
        return decoded.view((-1, 128, 16, 16))



