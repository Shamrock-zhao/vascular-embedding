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

from os import path, makedirs
from configparser import ConfigParser

from torch.autograd import Variable
from torch.utils import data

from learning.models import get_model
from learning.loader import get_loader
from learning.loss import cross_entropy2d
from learning.metrics import jaccard_index
from learning.plots import VisdomLinePlotter



def train(config_file, load_weights=False):

    # read the configuration file to identify the experiment configuration
    config = ConfigParser()
    config.read(config_file)
    experiment_name = config['experiment']['name']

    # retrieve input data path and output path
    data_path = config['folders']['data-path']
    output_path = path.join(config['folders']['output-path'], experiment_name)
    if not path.exists(output_path):
        makedirs(output_path)
    # prepare folder for checkpoints
    dir_checkpoints = path.join(output_path, 'checkpoints')
    if not path.exists(dir_checkpoints):
        makedirs(dir_checkpoints)

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
    # losses per epoch
    plotter = VisdomLinePlotter(env_name=config['experiment']['name'])

    # setup model
    model = get_model(config['architecture']['architecture'], 
                      int(config['architecture']['num-channels']),
                      int(config['architecture']['num-classes']),
                      parse_boolean(config['architecture']['batch-norm']),
                      float(config['architecture']['dropout']))

    # initialize the optimizer
    if config['training']['optimizer']=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr = float(config['training']['learning-rate']), 
                                    momentum = float(config['training']['momentum']), 
                                    weight_decay = float(config['training']['weight-decay']))
    else:
        raise "Optimizer {} not available".format(config['training']['optimizer'])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # ---------------------------
    # TRAINING ------------------
    # ---------------------------
    n_epochs = int(config['training']['epochs'])
    epoch_size = len(loader) // int(config['training']['batch-size'])
    current_epoch_loss = 0.0

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

            # accumulate loss to print per epoch
            current_epoch_loss += loss.data[0]

            # plot on screen
            plotter.plot('training-minibatch', 'loss', i, loss.data[0])
            
            # print loss every 20 iterations
            if (i+1) % 200 == 0:
                percentage = i / (loader.n_training_samples / int(config['training']['batch-size']))
                print("Epoch [%d/%d] - Progress: %.4f - Loss: %.4f" % (epoch+1, n_epochs, percentage, loss.data[0]))

        # Compute the mean epoch loss
        current_epoch_loss = current_epoch_loss / epoch_size

        # switch to the validation set
        loader.split = 'validation'
        # Run validation
        mean_val_loss, mean_val_jaccard_index = validate(loader, model, config)

        # plot values
        plotter.plot('loss', 'train', epoch+1, current_epoch_loss)
        plotter.plot('loss', 'validation', epoch+1, mean_val_loss)
        plotter.plot('jaccard', 'validation', epoch+1, mean_val_jaccard_index)

        # restart current_epoch_loss
        current_epoch_loss = 0.0
        # switch back to training
        loader.split = 'training'

        # save current checkpoint
        torch.save(model, path.join(dir_checkpoints, "{}_{}.pkl".format(experiment_name, epoch)))



def validate(loader, model, config):
    
    # set model for evaluation
    model.eval()
    # initialize the validation loader
    validation_loader = data.DataLoader(loader, batch_size=int(config['training']['batch-size']), num_workers=4, shuffle=False)

    mean_jaccard_index = 0.0
    mean_loss = 0.0
    n_iterations = len(loader) // int(config['training']['batch-size'])

    # iterate for each batch of validation samples
    for i, (images, labels) in enumerate(validation_loader):
        
        # turn the batch of images and labels to a cuda variable if available
        images = images.float()
        images = images.permute(0, 3, 1, 2)

        if torch.cuda.is_available():
            images = Variable(images, volatile=True).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images, volatile=True)
            labels = Variable(labels)

        # forward pass of the batch of images
        scores = model(images)
        # evaluate the cross entropy loss
        loss = cross_entropy2d(scores, labels)

        # get predictions and ground truth
        pred = scores.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()

        # sum up all the jaccard indices
        for gt_, pred_ in zip(gt, pred):
            mean_jaccard_index += jaccard_index(gt_, pred_)
        # and the loss function
        mean_loss += loss.data[0]
    
    # Compute average loss
    mean_loss = mean_loss / n_iterations
    # Compute average Jaccard
    mean_jaccard_index = mean_jaccard_index / len(loader)

    return mean_loss, mean_jaccard_index



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