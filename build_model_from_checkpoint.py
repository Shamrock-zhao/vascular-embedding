
import torch

from configparser import ConfigParser
from os import path
from ntpath import basename
from learning.models import get_model


def build_model_from_checkpoint(checkpoint_filename, config_file):

    # parse parameters
    config = ConfigParser()
    config.read(config_file)

    # setup model
    model = get_model(config['architecture']['architecture'], 
                      int(config['architecture']['num-channels']),
                      int(config['architecture']['num-classes']),
                      config['architecture']['batch-norm'].upper()=='TRUE',
                      float(config['architecture']['dropout']))
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # load weights
    model.load_state_dict(torch.load(checkpoint_filename))

    # return the packed model
    return model
    

    
import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_filename", help="full path to the checkpoint file", type=str)
    parser.add_argument("config_file", help="full path to the configuration file associated to this model", type=str)
    parser.add_argument("output_path", help="path to save the model file", type=str)

    args = parser.parse_args()

    # call the main function
    model = build_model_from_checkpoint(args.checkpoint_filename, args.config_file)

    # save the model
    torch.save(model, path.join(args.output_path, basename(args.checkpoint_filename) + '_full.pkl'))