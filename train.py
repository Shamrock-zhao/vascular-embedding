import sys
import torch
import visdom
import argparse
import numpy as np
import warnings
import ntpath

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import matplotlib.pyplot as plt

from os import path, makedirs, listdir
from configparser import ConfigParser
from ntpath import basename
from scipy import misc
from glob import glob

from torch.autograd import Variable
from torch.utils import data

from data_preparation.util.image_processing import preprocess
from data_preparation.util.files_processing import natural_key

from learning.models import get_model
from learning.loader import get_loader
from learning.loss import cross_entropy2d
from learning.metrics import dice_index
from learning.plots import VisdomLinePlotter



def train(config_file, load_weights=False):

    # read the configuration file to identify the experiment configuration
    config = ConfigParser()
    config.read(config_file)
    experiment_name = config['experiment']['name']

    # retrieve input data path and output path
    data_path = config['folders']['data-path']
    output_path = path.join(config['folders']['output-path'], basename(config_file))
    if not path.exists(output_path):
        makedirs(output_path)
    # prepare folder for checkpoints
    dir_checkpoints = path.join(output_path, 'checkpoints')
    if not path.exists(dir_checkpoints):
        makedirs(dir_checkpoints)

    # setup data loader for training and validation
    data_loader = get_loader('vessel')
    t_loader = data_loader(data_path, 'training', 
                        config['experiment']['sampling-strategy'],
                        config['experiment']['image-preprocessing'], 
                        parse_boolean(config['training']['augmented']))
    train_loader = data.DataLoader(t_loader, batch_size=int(config['training']['batch-size']), num_workers=4, shuffle=True)
    v_loader = data_loader(data_path, 'validation', 
                        config['experiment']['sampling-strategy'],
                        config['experiment']['image-preprocessing'], 
                        parse_boolean(config['training']['augmented']))
    val_loader = data.DataLoader(v_loader, batch_size=int(config['training']['batch-size']), num_workers=4, shuffle=True)

    # open a validation image to show its progress during training
    val_image_name = listdir(path.join(data_path, 'validation', 'images'))[0]
    val_image_mask = listdir(path.join(data_path, 'validation', 'masks'))[0]
    validation_image = np.asarray(misc.imread(path.join(data_path, 'validation', 'images', val_image_name)), dtype=np.uint8)
    validation_fov = np.asarray(misc.imread(path.join(data_path, 'validation', 'masks', val_image_mask)), dtype=np.uint8) // 255
    # preprocess the image according to the model
    validation_image = preprocess(validation_image, validation_fov, config['experiment']['image-preprocessing'])   

    # setup visdom for visualization
    vis = visdom.Visdom()
    # losses per epoch
    plotter = VisdomLinePlotter(env_name=config['experiment']['name'],
                                val_image_name=val_image_name)

    # setup model
    model = get_model(config['architecture']['architecture'], 
                        int(config['architecture']['num-channels']),
                        int(config['architecture']['num-classes']),
                        parse_boolean(config['architecture']['batch-norm']),
                        float(config['architecture']['dropout']))
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # load pretrained weights if possible
    if load_weights:
        checkpoints_filenames = sorted(glob(path.join(dir_checkpoints, '*.pkl')), key=natural_key)
        if len(checkpoints_filenames) > 0:
            model.load_state_dict(torch.load(checkpoints_filenames[-1]))
            checkpoint_name = ntpath.basename(checkpoints_filenames[-1])
            first_epoch = int(((checkpoint_name.split('_'))[-1].split('.'))[0]) + 1
        else:
            warnings.warn('Unable to find pretrained models in {}. Starting from 0.'.format(dir_checkpoints))
            first_epoch = 0
    else:
        first_epoch = 0

    # initialize the optimizer
    if config['training']['optimizer']=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr = float(config['training']['learning-rate']), 
                                    momentum = float(config['training']['momentum']), 
                                    weight_decay = float(config['training']['weight-decay']))
    else:
        raise "Optimizer {} not available".format(config['training']['optimizer'])


    # ---------------------------
    # TRAINING ------------------
    # ---------------------------
    n_epochs = int(config['training']['epochs'])
    epoch_size = len(t_loader) // int(config['training']['batch-size'])
    current_epoch_loss = 0.0

    for epoch in range(first_epoch, n_epochs):
        
        model.train()

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
            plotter.plot('minibatch loss', str(epoch+1), i, loss.data[0], 'minibatch')
            
            # print loss every 20 iterations
            if (i+1) % 200 == 0:
                percentage = i / (len(t_loader) / int(config['training']['batch-size']))
                print("Epoch [%d/%d] - Progress: %.4f - Loss: %.4f" % (epoch+1, n_epochs, percentage, loss.data[0]))

        # Compute the mean epoch loss
        current_epoch_loss = current_epoch_loss / epoch_size

        # Run validation
        mean_val_loss, mean_val_dice = validate(v_loader, val_loader, model, config)

        # evaluate the validation image on the current model
        validation_image_scores, _ = model.module.predict_from_full_image(validation_image)

        # plot values
        plotter.plot('loss', 'train', epoch+1, current_epoch_loss)
        plotter.plot('loss', 'validation', epoch+1, mean_val_loss)
        plotter.plot('dice', 'validation', epoch+1, mean_val_dice)
        plotter.display_image(validation_image_scores, epoch)

        # restart current_epoch_loss
        current_epoch_loss = 0.0

        # save current checkpoint
        torch.save(model.state_dict(), path.join(dir_checkpoints, "{}_{}.pkl".format(experiment_name, epoch)))

    # save the final model
    torch.save(model, path.join(dir_checkpoints, "model.pkl"))



def validate(loader, validation_loader, model, config):
    
    # set model for evaluation
    model.eval()

    mean_val_dice = 0.0
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
            mean_val_dice += dice_index(gt_, pred_)
        # and the loss function
        mean_loss += loss.data[0]
    
    # Compute average loss
    mean_loss = mean_loss / n_iterations
    # Compute average Jaccard
    mean_val_dice = mean_val_dice / len(loader)

    return mean_loss, mean_val_dice



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