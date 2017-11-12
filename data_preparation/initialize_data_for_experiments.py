

from configparser import ConfigParser
from util.experiments_preparation import prepare_all_data_together
from os import path, listdir, makedirs
import numpy as np


def prepare_data_for_experiments(data_path, experiment=0):
    """ Prepare data for experiments, grouping patches in separate folders """

    if experiment==0 or experiment==1:
        prepare_all_data_together(data_path, ['DRIVE', 'STARE', 'CHASEDB1', 'HRF'])




import sys
import argparse

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="data set path", type=str, default='data')
    parser.add_argument("--experiment", help="id of the experiment", type=int, default=0)

    args = parser.parse_args()

    # call the main function
    prepare_data_for_experiments(args.data_path, args.experiment)
