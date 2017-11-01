
import urllib.request # for downloading files
from os import path, makedirs, listdir, rename
from glob import glob
import zipfile
from shutil import copyfile, rmtree, move
from numpy import invert
from scipy import misc
from util.files_processing import natural_key, unzip_file








def move_files(filenames, source_folder, destination_folder):
    # Move files from source to destination folder
    for i in range(0, len(filenames)):
        move(path.join(source_folder, filenames[i]), path.join(destination_folder, filenames[i]))     




def organize_drive():

    # Check if tmp exists
    if not path.exists('tmp'):
        makedirs('tmp')

    # Download images from the known link to tmp
    data_path = path.join('tmp', 'DRIVE.zip')
    if not path.exists(data_path):
        raise ValueError('Download the DRIVE database from https://www.isi.uu.nl/Research/Databases/DRIVE/index.php and save the file in ./tmp.')
    else:
        print('DRIVE.zip file exists. Continuing processing...')

    # Unzip files
    unzip_file('tmp', 'DRIVE.zip', 'tmp')

    # Rename folders
    # Training set
    rename('tmp/DRIVE/training/1st_manual', 'tmp/DRIVE/training/labels')
    rename('tmp/DRIVE/training/mask', 'tmp/DRIVE/training/masks')
    # Test set
    rename('tmp/DRIVE/test/1st_manual', 'tmp/DRIVE/test/labels')
    rename('tmp/DRIVE/test/2nd_manual', 'tmp/DRIVE/test/labels2')
    rename('tmp/DRIVE/test/mask', 'tmp/DRIVE/test/masks')

    # Move images to the validation set
    makedirs('tmp/DRIVE/validation/images')
    image_filenames = sorted(listdir('tmp/DRIVE/training/images'), key=natural_key)
    move_files(image_filenames[-5:], 'tmp/DRIVE/training/images', 'tmp/DRIVE/validation/images')

    # Move labels to the validation set
    makedirs('tmp/DRIVE/validation/labels')
    image_filenames = sorted(listdir('tmp/DRIVE/training/labels'), key=natural_key)
    move_files(image_filenames[-5:], 'tmp/DRIVE/training/labels', 'tmp/DRIVE/validation/labels')

    # Move masks to the validation set
    makedirs('tmp/DRIVE/validation/masks')
    image_filenames = sorted(listdir('tmp/DRIVE/training/masks'), key=natural_key)
    move_files(image_filenames[-5:], 'tmp/DRIVE/training/masks', 'tmp/DRIVE/validation/masks')

    # Done! Now move the folder
    move('tmp/DRIVE', 'data/DRIVE')    



import sys

if __name__ == '__main__':

    # call the main function
    organize_drive()