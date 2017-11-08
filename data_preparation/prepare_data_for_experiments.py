

from configparser import ConfigParser
from util.patch_processing import extract_patches_from_training_datasets
import numpy as np



def prepare_data_for_experiments(patch_size=64, num_patches=1000, color=True, overwrite=True):

    # Set a random seed for reproducibility
    np.random.seed(7)

    # Extract patches from all the data sets
    print('Extract patches from all the data sets...')
    extract_patches_from_training_datasets('DRIVE', 
                                           patch_size=patch_size, 
                                           num_patches=num_patches, 
                                           color=color,
                                           overwrite=overwrite)   
    extract_patches_from_training_datasets('STARE', 
                                           patch_size=patch_size, 
                                           num_patches=num_patches, 
                                           color=color,
                                           overwrite=overwrite)   
    extract_patches_from_training_datasets('CHASEDB1', 
                                           patch_size=patch_size, 
                                           num_patches=num_patches, 
                                           color=color,
                                           overwrite=overwrite)   
    extract_patches_from_training_datasets('HRF', 
                                           patch_size=patch_size, 
                                           num_patches=num_patches, 
                                           color=color,
                                           overwrite=overwrite)   



import sys
import argparse

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", help="size of the patch", type=int, default=64)
    parser.add_argument("--num_patches", help="number of patches to extract from each image", type=int, default=1000)
    parser.add_argument("--color", help="a bool value indicating if patches have to be in color", type=str, default='True')
    parser.add_argument("--overwrite", help="a bool value indicating if patches have to be overwrite", type=str, default='True')

    args = parser.parse_args()
    args.color = (args.color.upper()=='TRUE')
    args.overwrite = (args.overwrite.upper()=='TRUE')

    # call the main function
    prepare_data_for_experiments(args.patch_size, args.num_patches, args.color, args.overwrite)