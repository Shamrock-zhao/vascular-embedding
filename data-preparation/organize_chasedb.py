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
from glob import glob
import zipfile
from shutil import copyfile, rmtree
from numpy import invert
from scipy import misc
import re




# URL to download images
URL = 'https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip'

# Training data paths
TRAINING_IMAGES_DATA_PATH = 'data/CHASEDB1/train/images'
TRAINING_GT_DATA_PATH = 'data/CHASEDB1/train/labels'
TRAINING_GT2_DATA_PATH = 'data/CHASEDB1/train/labels2'
# Validation data paths
VALIDATION_IMAGES_DATA_PATH = 'data/CHASEDB1/validation/images'
VALIDATION_GT_DATA_PATH = 'data/CHASEDB1/validation/labels'
VALIDATION_GT2_DATA_PATH = 'data/CHASEDB1/validation/labels2'
# Test data paths
TEST_IMAGES_DATA_PATH = 'data/CHASEDB1/test/images/'
TEST_GT_DATA_PATH = 'data/CHASEDB1/test/labels/'
TEST_GT2_DATA_PATH = 'data/CHASEDB1/test/labels2/'






def unzip_file(root_path, zip_filename, data_path):
    zip_ref = zipfile.ZipFile(path.join(root_path, zip_filename), 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()



def copy_images(root_folder, filenames, data_path):
    # Create the folder if it doesnt exist
    if not path.exists(data_path):
        makedirs(data_path)
    # Copy images in filenames to data_path
    for i in range(0, len(filenames)):
        current_file = path.basename(filenames[i])
        if current_file[-3:]=='JPG':
            target_filename = current_file[:-3] + 'jpg'
        else:
            target_filename = current_file
        copyfile(path.join(root_folder, current_file), path.join(data_path, target_filename))            



def copy_labels(root_folder, filenames, data_path):
    # Create folders
    if not path.exists(data_path):
        makedirs(data_path)
    # Copy the images
    for i in range(0, len(filenames)):
        current_filename = path.basename(filenames[i])
        # Open the image
        labels = (misc.imread(path.join(root_folder, current_filename)) / 255).astype('int32')
        # Save the image as a .png file
        misc.imsave(path.join(data_path, current_filename[:-3] + 'png'), labels)



def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def organize_chasedb():

    # Check if tmp exists
    if not path.exists('tmp'):
        makedirs('tmp')

    # Download images from the known link to tmp
    data_path = path.join('tmp', 'CHASEDB1.zip')
    if not path.exists(data_path):
        print('Downloading data from ' + URL)
        urllib.request.urlretrieve(URL, data_path)
    else:
        print('CHASEDB1.zip file already exists. Skipping download.')

    # Check if CHASEDB1 folders exist
    if not path.exists('tmp/CHASEDB1/images'):
        makedirs('tmp/CHASEDB1/images')
    if not path.exists('tmp/CHASEDB1/labels'):
        makedirs('tmp/CHASEDB1/labels')
    if not path.exists('tmp/CHASEDB1/labels2'):
        makedirs('tmp/CHASEDB1/labels2')        

    # Unzip files in tmp/CHASEDB1
    print('Unzipping images...')
    unzip_file('tmp', 'CHASEDB1.zip', 'tmp/CHASEDB1')


    # Move images
    # 1. Get image filenames
    image_filenames = sorted(glob('tmp/CHASEDB1/*.jpg'), key=natural_key)
    # 2. Copy the first 20 images as test set
    copy_images('tmp/CHASEDB1', image_filenames[:20], TEST_IMAGES_DATA_PATH)
    # 3. Copy the last 2 images as validation set
    copy_images('tmp/CHASEDB1', image_filenames[-2:], VALIDATION_IMAGES_DATA_PATH)
    # 4. Copy the images from 20 to 26 as training set
    copy_images('tmp/CHASEDB1', image_filenames[20:26], TRAINING_IMAGES_DATA_PATH)


    # Move first observer labels
    # 1. Get labels filenames (first observer)
    labels_filenames = sorted(glob('tmp/CHASEDB1/*1stHO.png'), key=natural_key)
    # 2. Copy the first 20 images as test set
    copy_labels('tmp/CHASEDB1', labels_filenames[:20], TEST_GT_DATA_PATH)
    # 3. Copy the last 2 images as validation set
    copy_labels('tmp/CHASEDB1', labels_filenames[-2:], VALIDATION_GT_DATA_PATH)
    # 4. Copy the images from 20 to 26 as training set
    copy_labels('tmp/CHASEDB1', labels_filenames[20:26], TRAINING_GT_DATA_PATH)

    # Move second observer labels
    # 1. Get labels filenames (first observer)
    labels2_filenames = sorted(glob('tmp/CHASEDB1/*2ndHO.png'), key=natural_key)
    # 2. Copy the first 20 images as test set
    copy_labels('tmp/CHASEDB1', labels2_filenames[:20], TEST_GT2_DATA_PATH)
    # 3. Copy the last 2 images as validation set
    copy_labels('tmp/CHASEDB1', labels2_filenames[-2:], VALIDATION_GT2_DATA_PATH)
    # 4. Copy the images from 20 to 26 as training set
    copy_labels('tmp/CHASEDB1', labels2_filenames[20:26], TRAINING_GT2_DATA_PATH)

    # Remove useless folders
    rmtree('tmp/CHASEDB1/')





import sys

if __name__ == '__main__':

    # call the main function
    organize_chasedb()