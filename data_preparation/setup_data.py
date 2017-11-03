

from os import makedirs, path
from util.organize_data import organize_drive, organize_stare, organize_chasedb, organize_hrf, organize_iostar



def setup_data():

    # Check if tmp exists
    if not path.exists('tmp'):
        makedirs('tmp')

    print('Preparing DRIVE...')
    organize_drive()
    print('Preparing STARE...')
    organize_stare()
    print('Preparing CHASEDB...')
    organize_chasedb()
    print('Preparing HRF...')
    organize_hrf()
    print('Preparing IOSTAR...')
    organize_iostar()
    


import sys

if __name__ == '__main__':

    # call the main function
    setup_data()