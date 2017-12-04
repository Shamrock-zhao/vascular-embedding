

from os import makedirs, path
from util.organize_data import organize_drive, organize_stare, organize_chasedb, organize_hrf, organize_iostar, organize_drhagis



def print_dataset_was_ready(dataset_name):
    print('{0} data set seems to be already prepared. Skipping...'.format(dataset_name))



def organize_datasets(data_path='../data'):

    # Check if tmp exists
    if not path.exists('../tmp'):
        makedirs('../tmp')

    # Prepare DRIVE if the folder does not exist
    if not path.exists(path.join(data_path, 'DRIVE')):
        print('Preparing DRIVE...')
        organize_drive()
    else:
        print_dataset_was_ready('DRIVE')   

    # Prepare STARE if the folder does not exist
    if not path.exists(path.join(data_path, 'STARE')):
        print('Preparing STARE...')
        organize_stare()
    else:
        print_dataset_was_ready('STARE')

    # Prepare CHASEDB1 if the folder does not exist
    if not path.exists(path.join(data_path, 'CHASEDB1')):
        print('Preparing CHASEDB1...')
        organize_chasedb()
    else:
        print_dataset_was_ready('CHASEDB1')      

    # Prepare HRF if the folder does not exist
    if not path.exists(path.join(data_path, 'HRF')):
        print('Preparing HRF...')
        organize_hrf()
    else:
        print_dataset_was_ready('HRF') 

    # Prepare IOSTAR if the folder does not exist
    if not path.exists(path.join(data_path, 'IOSTAR')):
        print('Preparing IOSTAR...')
        organize_iostar()
    else:
        print_dataset_was_ready('IOSTAR') 
    
    # Prepare DRHAGIS if the folder does not exist
    if not path.exists(path.join(data_path, 'DRHAGIS')):
        print('Preparing DRHAGIS...')
        organize_drhagis()
    else:
        print_dataset_was_ready('DRHAGIS')     


import sys
import argparse

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="data set path", type=str, default='../data')

    args = parser.parse_args()

    # call the main function
    organize_datasets(args.data_path)