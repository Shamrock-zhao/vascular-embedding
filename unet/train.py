import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

from load import get_ids, encode_training_validation_data, get_imgs_and_masks
from data_visualization import *
from utils import batch
from losses import DiceLoss
from evaluation import eval_net
from unet_model import UNet
from torch.autograd import Variable
from torch import optim
from optparse import OptionParser
import sys
import os

from configparser import ConfigParser
from os import path, makedirs, listdir, rename


def train_net(net, data_path, output_path, config):
    
    # configure input/output folders
    type_of_sampling = 'patches_' + config['experiment']['sampling-strategy']
    type_of_patch = type_of_sampling + '_' + config['experiment']['image-preprocessing']
    
    dir_training_img = path.join(data_path, 'training', type_of_patch)
    dir_training_labels = path.join(data_path, 'training', type_of_sampling + '_labels')
    dir_validation_img = path.join(data_path, 'validation', type_of_patch)
    dir_validation_labels = path.join(data_path, 'validation', type_of_sampling + '_labels')
    dir_checkpoints = path.join(output_path, 'checkpoints')
    
    # initialize output folders if they does not exist
    if not path.exists(output_path):
        makedirs(output_path)
    if not path.exists(dir_checkpoints):
        makedirs(dir_checkpoints)

    # setup cuda is necessary
    use_gpu = parse_boolean(config['technical']['gpu'])
    if use_gpu:
        net.cuda()
        cudnn.benchmark = True

    # parse training parameters
    n_epochs = int(config['training']['epochs'])
    batch_size = int(config['training']['batch-size'])
    lr = float(config['training']['learning-rate'])

    #encode training and validation data
    training_data_filenames = get_ids(dir_training_img)
    validation_data_filenames = get_ids(dir_validation_img)
    iddataset = encode_training_validation_data(training_data_filenames, validation_data_filenames)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        CUDA: {}
    '''.format(n_epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(use_gpu)))

    # get the number of training samples
    N_train = len(iddataset['train'])

    # get the training and validation sets
    train = get_imgs_and_masks(iddataset['train'], dir_training_img, dir_training_labels)
    val = get_imgs_and_masks(iddataset['val'], dir_validation_img, dir_validation_labels)

    #setup the optimizer and the loss functions
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.BCELoss()

    # train!
    for epoch in range(n_epochs):
        
        print('Starting epoch {}/{}.'.format(epoch+1, n_epochs))
        train = get_imgs_and_masks(iddataset['train'], dir_training_img, dir_training_labels)
        val = get_imgs_and_masks(iddataset['val'], dir_validation_img, dir_validation_labels)

        epoch_loss = 0

        if 1:
            val_dice = eval_net(net, val, use_gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        for i, b in enumerate(batch(train, batch_size)):
            X = np.array([i[0] for i in b])
            y = np.array([i[1] for i in b])

            X = torch.FloatTensor(X)
            y = torch.ByteTensor(y)

            if use_gpu:
                X = Variable(X).cuda()
                y = Variable(y).cuda()
            else:
                X = Variable(X)
                y = Variable(y)

            y_pred = net(X)
            probs = F.sigmoid(y_pred)
            probs_flat = probs.view(-1)

            y_flat = y.view(-1)

            loss = criterion(probs_flat, y_flat.float())
            epoch_loss += loss.data[0]

            print('{0:.4f} --- loss: {1:.6f}'.format(i*batch_size/N_train,
                                                     loss.data[0]))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss/i))

        torch.save(net.state_dict(),
                    dir_checkpoints + 'CP{}.pth'.format(epoch+1))
        print('Checkpoint {} saved !'.format(epoch+1))




def parse_boolean(input_string):
    return input_string.upper()=='TRUE'



def train(config_file, load_weights=False):
    
    # read the configuration file to identify the experiment configuration
    config = ConfigParser()
    config.read(config_file)
    experiment_name = config['experiment']['name']

    data_path = config['folders']['data-path']
    output_path = path.join(config['folders']['output-path'], experiment_name)

    # parse data parameters
    num_channels = int(config['architecture']['num-channels'])
    num_classes = int(config['architecture']['num-classes'])
    batch_norm = parse_boolean(config['architecture']['batch-norm'])

    # parse training parameters
    n_epochs = int(config['training']['epochs'])
    batch_size = int(config['training']['batch-size'])
    lr = float(config['training']['learning-rate'])

    # initialize the u-net
    net = UNet(num_channels, num_classes, batch_norm)

    # run training and save intermediate model if interrupted
    try:
        train_net(net, data_path, output_path, config)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), path.join(output_path, 'INTERRUPTED.pth'))
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



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