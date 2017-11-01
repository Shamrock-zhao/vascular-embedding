'''
# Ugly hack to allow absolute import from the root folder
# whatever its name is. Please forgive the heresy.
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "core"
    from core.preprocess.extract_patches import generate_random_patches
'''

import urllib.request # for downloading files
from os import path, makedirs, listdir
import zipfile
from shutil import copyfile, rmtree
from numpy import invert
from scipy import misc

import tarfile, sys

from util.files_processing import natural_key, untar_file, ungz_file



# URLs to download images
URLs = ['http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar', 
        'http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar',
        'http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar']
URLS_FILENAMES = ['stare-images.tar', 'labels-ah.tar', 'labels-vk.tar']

# Training data paths
TRAINING_IMAGES_DATA_PATH = 'data/STARE/train/images'
TRAINING_GT_DATA_PATH = 'data/STARE/train/labels'
TRAINING_GT2_DATA_PATH = 'data/STARE/train/labels2'
# Validation data paths
VALIDATION_IMAGES_DATA_PATH = 'data/STARE/validation/images'
VALIDATION_GT_DATA_PATH = 'data/STARE/validation/labels'
VALIDATION_GT2_DATA_PATH = 'data/STARE/validation/labels2'
# Test data paths
TEST_IMAGES_DATA_PATH = 'data/STARE/test/images/'
TEST_GT_DATA_PATH = 'data/STARE/test/labels/'
TEST_GT2_DATA_PATH = 'data/STARE/test/labels2/'



def ungz_files(root_path, gz_filenames, data_path):
    ''' 
    Uncompressed all the gz files in a folder
    '''
    # Create folders
    if not path.exists(data_path):
        makedirs(data_path)
    # Ungz files    
    for i in range(0, len(gz_filenames)):
        ungz_file(root_path, gz_filenames[i], data_path)





def organize_stare():

    # Check if tmp exists
    if not path.exists('tmp'):
        makedirs('tmp')

    # Download images from the known links to tmp
    for i in range(0, len(URLs)):
        data_path = path.join('tmp', URLS_FILENAMES[i])
        if not path.exists(data_path):
            print('Downloading data from ' + URLs[i])
            urllib.request.urlretrieve(URLs[i], data_path)
        else:
            print(URLS_FILENAMES[i] + ' already exists. Skipping download.')

    # Check if STARE folders exist
    if not path.exists('tmp/STARE/images'):
        makedirs('tmp/STARE/images')
    if not path.exists('tmp/STARE/labels'):
        makedirs('tmp/STARE/labels')
    if not path.exists('tmp/STARE/labels2'):
        makedirs('tmp/STARE/labels2')        

    # Untar images in tmp/images
    print('Extracting images...')
    untar_file('tmp', URLS_FILENAMES[0], 'tmp/STARE/images')
    # Untar images in tmp/gt
    print('Extracting labels...')
    untar_file('tmp', URLS_FILENAMES[1], 'tmp/STARE/labels')
    # Untar images in tmp/gt
    print('Extracting the other labels...')
    untar_file('tmp', URLS_FILENAMES[2], 'tmp/STARE/labels2')    


    # Generate training/test images
    print('Copying images...')
    # 1. Get image names
    image_filenames = sorted(listdir('tmp/STARE/images'), key=natural_key)
    # 2. Copy training images
    ungz_files('tmp/STARE/images', image_filenames[:7], TRAINING_IMAGES_DATA_PATH)
    # 3. Copy validation images
    ungz_files('tmp/STARE/images', image_filenames[7:10], VALIDATION_IMAGES_DATA_PATH)
    # 4. Copy test images
    ungz_files('tmp/STARE/images', image_filenames[-10:], TEST_IMAGES_DATA_PATH)

    # Generate training/test labels
    print('Copying labels...')
    # 1. Get image names
    gt_filenames = sorted(listdir('tmp/STARE/labels'), key=natural_key)
    # 2. Copy training labels
    ungz_files('tmp/STARE/labels', gt_filenames[:7], TRAINING_GT_DATA_PATH)
    # 3. Copy validation labels
    ungz_files('tmp/STARE/labels', gt_filenames[7:10], VALIDATION_GT_DATA_PATH)
    # 4. Copy test labels
    ungz_files('tmp/STARE/labels', gt_filenames[-10:], TEST_GT_DATA_PATH)

    # Generate training/test labels2
    print('Copying the other labels...')
    # 1. Get image names
    gt_filenames = sorted(listdir('tmp/STARE/labels2'), key=natural_key)
    # 2. Copy training labels2
    ungz_files('tmp/STARE/labels2', gt_filenames[:7], TRAINING_GT2_DATA_PATH)
    # 3. Copy validation labels2
    ungz_files('tmp/STARE/labels2', gt_filenames[7:10], VALIDATION_GT2_DATA_PATH)
    # 4. Copy test labels2
    ungz_files('tmp/STARE/labels2', gt_filenames[-10:], TEST_GT2_DATA_PATH)
    
    # Remove useless folders
    rmtree('tmp/STARE/')



import sys

if __name__ == '__main__':

    # call the main function
    organize_stare()