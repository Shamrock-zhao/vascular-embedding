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
from shutil import copyfile, rmtree
from numpy import invert
from scipy import misc
from data_preparation.util.files_processing import natural_key, unzip_file





# URLs to download images
URLs = ['https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip', 
        'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma.zip',
        'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy.zip',
        'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip',
        'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma_manualsegm.zip',
        'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy_manualsegm.zip']

# URL filenames
URL_FILENAMES_IMAGES = ['healthy.zip', 'glaucoma.zip', 'diabetic_retinopathy.zip']
URL_FILENAMES_LABELS = ['healthy_manualsegm.zip', 'glaucoma_manualsegm.zip', 'diabetic_retinopathy_manualsegm.zip']
ALL_URL_FILENAMES = URL_FILENAMES_IMAGES + URL_FILENAMES_LABELS

# Training data paths
TRAINING_IMAGES_DATA_PATH = 'data/HRF/train/images'
TRAINING_GT_DATA_PATH = 'data/HRF/train/labels'
# Validation data paths
VALIDATION_IMAGES_DATA_PATH = 'data/HRF/validation/images'
VALIDATION_GT_DATA_PATH = 'data/HRF/validation/labels'
# Test data paths
TEST_IMAGES_DATA_PATH = 'data/HRF/test/images/'
TEST_GT_DATA_PATH = 'data/HRF/test/labels/'






def unzip_files(root_path, zip_filenames, data_path):
    # Unzip files    
    for i in range(0, len(zip_filenames)):
        unzip_file(root_path, zip_filenames[i], data_path)



def copy_images(root_folder, filenames, data_path):
    # Create the folder if it doesnt exist
    if not path.exists(data_path):
        makedirs(data_path)
    # Copy images in filenames to data_path
    for i in range(0, len(filenames)):
        current_file = filenames[i]
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
        current_filename = filenames[i]
        # Open the image
        labels = (misc.imread(path.join(root_folder, current_filename)) / 255).astype('int32')
        # Save the image as a .png file
        misc.imsave(path.join(data_path, current_filename[:-3] + 'png'), labels)




def organize_hrf():

    # Check if tmp exists
    if not path.exists('tmp'):
        makedirs('tmp')

    # Download images from the known links to tmp
    for i in range(0, len(URLs)):
        data_path = path.join('tmp', ALL_URL_FILENAMES[i])
        if not path.exists(data_path):
            print('Downloading data from ' + URLs[i])
            urllib.request.urlretrieve(URLs[i], data_path)
        else:
            print(ALL_URL_FILENAMES[i] + ' already exists. Skipping download.')

    # Check if HRF folders exist
    if not path.exists('tmp/HRF/images'):
        makedirs('tmp/HRF/images')
    if not path.exists('tmp/HRF/labels'):
        makedirs('tmp/HRF/labels')

    # Unzip images in tmp/images
    print('Unzipping images...')
    unzip_files('tmp', URL_FILENAMES_IMAGES, 'tmp/HRF/images')
    # Unzip images in tmp/gt
    print('Unzipping labels...')
    unzip_files('tmp', URL_FILENAMES_LABELS, 'tmp/HRF/labels')

    # Generate training/test images
    print('Copying images...')
    # 1. Get image names
    image_filenames = sorted(listdir('tmp/HRF/images'), key=natural_key)
    # 2. Copy training images
    copy_images('tmp/HRF/images', image_filenames[:12], TRAINING_IMAGES_DATA_PATH)
    # 3. Copy validation
    copy_images('tmp/HRF/images', image_filenames[12:15], VALIDATION_IMAGES_DATA_PATH)
    # 4. Copy test images
    copy_images('tmp/HRF/images', image_filenames[-30:], TEST_IMAGES_DATA_PATH)

    # Generate training/test labels 
    print('Copying labels...')
    # 1. Get labels names
    gt_filenames = sorted(listdir('tmp/HRF/labels'), key=natural_key)
    # 2. Copy training labels
    copy_labels('tmp/HRF/labels', gt_filenames[:12], TRAINING_GT_DATA_PATH)
    # 3. Copy validation labels
    copy_labels('tmp/HRF/labels', gt_filenames[12:15], VALIDATION_GT_DATA_PATH)
    # 4. Copy test labels
    copy_labels('tmp/HRF/labels', gt_filenames[-30:], TEST_GT_DATA_PATH)

    # Remove useless folders
    rmtree('tmp/HRF/')




import sys

if __name__ == '__main__':

    # call the main function
    organize_hrf()