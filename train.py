import sys
import torch
import visdom
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import matplotlib.pyplot as plt

from os import path
from configparser import ConfigParser

from torch.autograd import Variable
from torch.utils import data

from learning.models import get_model
from learning.loader import get_loader
from learning.loss import cross_entropy2d
from learning.metrics import scores



def train(config_file, load_weights=False):

    # read the configuration file to identify the experiment configuration
    config = ConfigParser()
    config.read(config_file)
    experiment_name = config['experiment']['name']

    # retrieve input data path and output path
    data_path = config['folders']['data-path']
    output_path = path.join(config['folders']['output-path'], experiment_name)
    # prepare folder for checkpoints
    dir_checkpoints = path.join(output_path, 'checkpoints')
    # get type of patch sampling and color preprocessing
    type_of_sampling = 'patches_' + config['experiment']['sampling-strategy']
    type_of_patch = type_of_sampling + '_' + config['experiment']['image-preprocessing']
    # and prepare image and labels path
    img_path = path.join(data_path, 'training', type_of_patch)
    labels_path = path.join(data_path, 'training', type_of_sampling + '_labels')

    # setup data loader
    data_loader = get_loader('vessel')
    loader = data_loader(img_path, labels_path, 'training', 
                        float(config['training']['validation-ratio']), 
                        parse_boolean(config['training']['augmented']))
    n_classes = loader.n_classes
    train_loader = data.DataLoader(loader, batch_size=int(config['training']['batch-size']), num_workers=4, shuffle=True)

    # setup visdom for visualization
    vis = visdom.Visdom()
    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss - ' + experiment_name,
                                     legend=['Loss']))

    # setup model
    model = get_model(config['architecture']['architecture'], 
                      int(config['architecture']['num-channels']),
                      int(config['architecture']['num-classes']),
                      parse_boolean(config['architecture']['batch-norm']))

    # initialize the optimizer
    if config['training']['optimizer']=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr = float(config['training']['learning-rate']), 
                                    momentum = float(config['training']['momentum']), 
                                    weight_decay = float(config['training']['weight-decay']))
    else:
        raise 'Optimizer {} not available'.format(config['training']['optimizer'])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        #test_image, test_segmap = loader[0]
        #test_image = torch.from_numpy(test_image)
        #test_segmap = torch.from_numpy(test_segmap)
        #test_image = Variable(test_image.unsqueeze(0).cuda(0))
    #else:
    #    test_image, test_segmap = loader[0]
    #    test_image = Variable(test_image.unsqueeze(0))

    # ---------------------------
    # TRAINING ------------------
    # ---------------------------
    n_epochs = int(config['training']['epochs'])
    for epoch in range(0, n_epochs):
        
        # for each batch
        for i, (images, labels) in enumerate(train_loader):
            
            # turn the batch of images and labels to a cuda variable if available
            images = images.float()
            images = images.permute(0, 3, 1, 2)

            if torch.cuda.is_available():
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
            else:
                images = Variable(images)
                labels = Variable(labels)

            
            #images = images.float()
            #labels = labels.float()

            # clear the gradients
            optimizer.zero_grad()
            # forward pass of the batch through the model
            outputs = model(images)
            # computation of the cross entropy loss
            loss = cross_entropy2d(outputs, labels)
            # backward pass
            loss.backward()
            # gradient update
            optimizer.step()

            # plot on screen
            vis.line(
                X=torch.ones((1, 1)).cpu() * i,
                Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                win=loss_window,
                update='append')

            # print loss every 20 iterations
            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, n_epochs, loss.data[0]))

        # test_output = model(test_image)
        # predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        # target = loader.decode_segmap(test_segmap.numpy())

        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))

        torch.save(model, path.join(dir_checkpoints, "{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch)))



def parse_boolean(input_string):
    return input_string.upper()=='TRUE'



import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="configuration file", type=str)
    parser.add_argument("--load", help="load precomputed weights", type=str, default='False')

    args = parser.parse_args()

    # call the main function
    train(args.config_file, parse_boolean(args.load))