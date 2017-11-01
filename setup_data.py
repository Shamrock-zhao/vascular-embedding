
from data_preparation import organize_drive, organize_stare, organize_hrf, organize_chasedb
from os import makedirs, path


def setup_data():

    # Check if tmp exists
    if not path.exists('tmp'):
        makedirs('tmp')

    print('Preparing DRIVE...')
    organize_drive.organize_drive()
    print('Preparing STARE...')
    organize_stare.organize_stare()
    print('Preparing CHASEDB...')
    organize_chasedb.organize_chasedb()
    print('Preparing HRF...')
    organize_hrf.organize_hrf()
    


import sys

if __name__ == '__main__':

    # call the main function
    setup_data()