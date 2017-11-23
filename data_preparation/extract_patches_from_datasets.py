

from configparser import ConfigParser
from util.patch_processing import extract_random_patches_from_dataset
from os import path, listdir, makedirs
import numpy as np


def extract_patches_from_datasets(patch_size=64, num_patches=200000, overwrite=False):
    '''
    Prepares data for experiments, creating pickle files with training data for different configurations.
    '''

    # Set a random seed for reproducibility
    np.random.seed(7)

    datasets = ['DRIVE', 'STARE', 'CHASEDB1', 'HRF']

    # For each database
    for i in range(0, len(datasets)):
        # Process training data
        if overwrite or not path.exists(path.join('data', datasets[i], 'training', 'patches_guided-by-labels_labels')):
            print('Extracting patches from {} training set...'.format(datasets[i]))
            extract_random_patches_from_dataset(path.join('data', datasets[i], 'training'), 
                                                patch_size=patch_size, num_patches=num_patches)
        else:
            print('{} training set precomputed.'.format(datasets[i]))
        # Process validation data
        if overwrite or not path.exists(path.join('data', datasets[i], 'validation', 'patches_guided-by-labels_labels')):
            print('Extracting patches from {} validation set...'.format(datasets[i]))
            extract_random_patches_from_dataset(path.join('data', datasets[i], 'validation'), 
                                                patch_size=patch_size, num_patches=int(num_patches * 0.05))
        else:
            print('{} validation set precomputed.'.format(datasets[i]))



import sys
import argparse

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", help="size of the patch", type=int, default=64)
    parser.add_argument("--num_patches", help="number of patches to extract from each image", type=int, default=200000)
    parser.add_argument("--overwrite", help="overwrite existing folders", type=str, default='False')

    args = parser.parse_args()
    args.overwrite = args.overwrite.upper()=='TRUE'

    # call the main function
    extract_patches_from_datasets(args.patch_size, args.num_patches, args.overwrite)