

from configparser import ConfigParser
from util.patch_processing import extract_random_patches_from_dataset
from os import path, listdir, makedirs
import numpy as np


def extract_patches_from_datasets(patch_size=64, num_patches=1000):
    '''
    Prepares data for experiments, creating pickle files with training data for different configurations.
    '''

    # Set a random seed for reproducibility
    np.random.seed(7)

    print('Extracting patches from DRIVE training set...')
    extract_random_patches_from_dataset(path.join('data', 'DRIVE', 'training'), 
                                        patch_size=patch_size, num_patches=num_patches)
    extract_random_patches_from_dataset(path.join('data', 'DRIVE', 'validation'), 
                                        patch_size=patch_size, num_patches=num_patches)

    print('Extracting patches from STARE training set...')
    extract_random_patches_from_dataset(path.join('data', 'STARE', 'training'), 
                                        patch_size=patch_size, num_patches=num_patches)
    extract_random_patches_from_dataset(path.join('data', 'STARE', 'validation'), 
                                        patch_size=patch_size, num_patches=num_patches)

    print('Extracting patches from CHASEDB1 training set...')
    extract_random_patches_from_dataset(path.join('data', 'CHASEDB1', 'training'), 
                                        patch_size=patch_size, num_patches=num_patches)
    extract_random_patches_from_dataset(path.join('data', 'CHASEDB1', 'validation'), 
                                        patch_size=patch_size, num_patches=num_patches)

    print('Extracting patches from HRF training set...')
    extract_random_patches_from_dataset(path.join('data', 'HRF', 'training'), 
                                        patch_size=patch_size, num_patches=num_patches)
    extract_random_patches_from_dataset(path.join('data', 'HRF', 'validation'), 
                                        patch_size=patch_size, num_patches=num_patches)                                    



import sys
import argparse

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", help="size of the patch", type=int, default=64)
    parser.add_argument("--num_patches", help="number of patches to extract from each image", type=int, default=1000)

    args = parser.parse_args()

    # call the main function
    extract_patches_from_datasets(args.patch_size, args.num_patches)