

from os import makedirs, path
from util.organize_data import organize_drive, organize_stare, organize_chasedb, organize_hrf, organize_iostar, organize_drhagis



def print_dataset_was_ready(dataset_name):
    print('{0} data set seems to be already prepared. Skipping...'.format(dataset_name))



def setup_data(patch_size=64):

    # Check if tmp exists
    if not path.exists('tmp'):
        makedirs('tmp')

    # Prepare DRIVE if the folder does not exist
    if not path.exists('data/DRIVE'):
        print('Preparing DRIVE...')
        organize_drive()
    else:
        print_dataset_was_ready('DRIVE')

    # Prepare STARE if the folder does not exist
    if not path.exists('data/STARE'):
        print('Preparing STARE...')
        organize_stare()
    else:
        print_dataset_was_ready('STARE')

    # Prepare CHASEDB1 if the folder does not exist
    if not path.exists('data/CHASEDB1'):
        print('Preparing CHASEDB1...')
        organize_chasedb()
    else:
        print_dataset_was_ready('CHASEDB1')        

    # Prepare HRF if the folder does not exist
    if not path.exists('data/HRF'):
        print('Preparing HRF...')
        organize_hrf()
    else:
        print_dataset_was_ready('HRF') 
    
    # Prepare IOSTAR if the folder does not exist
    if not path.exists('data/IOSTAR'):
        print('Preparing IOSTAR...')
        organize_iostar()
    else:
        print_dataset_was_ready('IOSTAR') 
    
    # Prepare DRHAGIS if the folder does not exist
    if not path.exists('data/DRHAGIS'):
        print('Preparing DRHAGIS...')
        organize_drhagis()
    else:
        print_dataset_was_ready('DRHAGIS') 

    
    # Extract random patches from each training set
    #extract_random_patches()
    


import sys

if __name__ == '__main__':

    # call the main function
    setup_data()