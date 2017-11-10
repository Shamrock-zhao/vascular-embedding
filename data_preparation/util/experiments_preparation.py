
from os import path, listdir, makedirs
import numpy as np
from .files_processing import copy_images, natural_key
from glob import glob


def prepare_all_data_together(data_path, datasets):

    # our subsets will be the training and validation set of each data set
    subsets = ['training', 'validation']

    # we will process each of them (training, validation)
    for k in range(0, len(subsets)):

        # make folders where data will be saved
        new_folder = path.join(data_path, 'all-patches', subsets[k])
        if not path.exists(new_folder):
            makedirs(new_folder)

        # copy each set of patches (DRIVE, STARE, CHASEDB1, HRF)
        for i in range(0, len(datasets)):

            # get the name of the training set (/DATASET/training)
            current_training_folder = path.join(data_path, datasets[i], subsets[k])
            # retrieve all patches subfolders (all the patches)
            patches_folder_names = glob(path.join(current_training_folder, 'patches_*'))

            print('Working with {}'.format(current_training_folder))

            # copy the content of each subfolder to the destination folder
            for j in range(0, len(patches_folder_names)):
                current_patches = path.basename(patches_folder_names[j])
                print(current_patches)
                # get all the files in the folder
                if current_patches[-6:]=='labels':
                    current_filenames = sorted(glob(path.join(patches_folder_names[j], '*.gif')), key=natural_key)
                else:
                    current_filenames = sorted(glob(path.join(patches_folder_names[j], '*.png')), key=natural_key)
                print('Copying {} patches'.format(len(current_filenames)))
                # prepare new folder name
                current_new_folder = path.join(new_folder, path.basename(patches_folder_names[j]))
                # copy all the images
                copy_images(patches_folder_names[j], current_filenames, current_new_folder)


    