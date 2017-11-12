import torch
from losses import dice_coeff
import numpy as np
from torch.autograd import Variable
from data_visualization import plot_img_mask
import matplotlib.pyplot as plt
import torch.nn.functional as F
#from crf import dense_crf


def eval_net(net, dataset, gpu=False):
    tot = 0
    quantity = 0

    for i, b in enumerate(dataset):
        X = b[0]
        y = b[1]

        X = torch.FloatTensor(X).unsqueeze(0)
        y = torch.ByteTensor(y).unsqueeze(0)

        if gpu:
            X = Variable(X, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
        else:
            X = Variable(X, volatile=True)
            y = Variable(y, volatile=True)

        y_pred = net(X)
        y_pred = (F.sigmoid(y_pred) > 0.5).float()
        #y_pred.data = y_pred.data.squeeze(0)

        dice = dice_coeff(y_pred, y.float()).data[0]
        tot += dice
        quantity += 1
    
    return tot / quantity