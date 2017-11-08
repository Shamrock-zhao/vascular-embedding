
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from os import path, makedirs, listdir, rename
from scipy import misc
from shutil import rmtree
from .files_processing import natural_key
from skimage import measure



def preprocess(image, fov_mask):
    
    return image



def pick_random_coordinate(coordinates):
    
    # get number of coordinates
    coordinates_shape = coordinates.shape
    n_row = coordinates_shape[0]
    # pick random element from the coordinates array
    idx = np.random.randint(n_row)
    random_coordinate = coordinates[idx, :]
    # remove it to avoid repetitions
    coordinates = np.delete(coordinates, idx, 0)
    # return each coordinate and the (updated) coordinates array
    return random_coordinate[0], random_coordinate[1], coordinates



def extract_random_patches(dataset_folder, patch_size=64, num_patches=1000, is_training=True, color=True, overwrite=True):
    
    # prepare input folders
    img_folder = path.join(dataset_folder, 'images')
    fov_masks_folder = path.join(dataset_folder, 'masks')

    # get image and fov masks filenames from input folder
    image_filenames = sorted(listdir(img_folder), key=natural_key)
    fov_masks_filenames = sorted(listdir(fov_masks_folder), key=natural_key)
    
    # initialize output folder
    output_image_folder = path.join(dataset_folder, 'patches')
    if overwrite:
        rmtree(output_image_folder)
    if not path.exists(output_image_folder):
        makedirs(output_image_folder)

    # if training data, get also the labels filenames and prepare the output folder
    if is_training:
        gt_folder = path.join(dataset_folder, 'labels')
        gt_filenames = sorted(listdir(gt_folder), key=natural_key)
        output_labels_folder = path.join(dataset_folder, 'labels_patches')
        if overwrite:
            rmtree(output_labels_folder)
        if not path.exists(output_labels_folder):
            makedirs(output_labels_folder)

    # for each image
    for i in range(0, len(image_filenames)):
        
        # identify current image
        current_image_filename = image_filenames[i]
        current_mask_filename = fov_masks_filenames[i]
        print('Processing image ' + current_image_filename)
        
        # open image and preprocess it accordingly
        image = misc.imread(path.join(img_folder, current_image_filename))
        fov_mask = misc.imread(path.join(fov_masks_folder, current_mask_filename))  > 0
        if len(fov_mask.shape) > 2:
            fov_mask = fov_mask[:,:,0]
        image = preprocess(image, fov_mask)

        # precompute pad
        pad = int(patch_size/2)
        # open labels is they exist
        if is_training:
            # identify current image
            current_gt_filename = gt_filenames[i]
            # open the image
            labels = (misc.imread(path.join(gt_folder, current_gt_filename)) / 255).astype('int32')
            # assign labels mask mask
            mask = labels
        else:
            # assign FOV mask as mask
            mask = fov_mask

        # initialize a padded fov
        sizes = mask.shape
        padded_mask = np.zeros(sizes, dtype=bool)
        padded_mask[ pad:sizes[0]-pad, pad:sizes[1]-pad ] = mask[ pad:sizes[0]-pad, pad:sizes[1]-pad ] > 0
        
        # identify connected components in the mask and use its coordinates
        # to extract patches
        region_props = measure.regionprops(measure.label(padded_mask))
        coordinates = None
        for sub_reg in range(0, len(region_props)):
            if coordinates is None:
                coordinates = region_props[sub_reg].coords
            else:
                coordinates = np.concatenate((coordinates, region_props[sub_reg].coords))

        # extract N_subimgs patches
        for j in range(0, num_patches):
            
            # pick random coordinate inside the fov
            x, y, coordinates = pick_random_coordinate(coordinates)

            # get a random patch around the random coordinate
            if color:
                random_patch = image[x-pad:x+pad, y-pad:y+pad, :]
            else:
                random_patch = image[x-pad:x+pad, y-pad:y+pad, 1]
            
            # save the patch
            misc.imsave(path.join(output_image_folder, current_image_filename[:-4] + str(j) + '.png'), random_patch)
            
            # if is training, crop the label as well
            if is_training:               
                random_patch_labels = labels[x-pad : x+pad, y-pad : y+pad]
                misc.imsave(path.join(output_labels_folder, current_gt_filename[:-4] + str(j) + '.gif'), 
                    random_patch_labels)                



def extract_patches_from_training_datasets(dataset_name, patch_size=64, num_patches=1000, is_training=True, color=True, overwrite=True):
    
    # Extract random patches from training/validation data sets
    training_dataset_path = path.join('data', dataset_name, 'training')

    if not path.exists(path.join(training_dataset_path, 'patches')) or overwrite:
        print('Extracting patches from {0} training set...'.format(dataset_name))
        extract_random_patches(training_dataset_path, 
                               patch_size=patch_size, 
                               num_patches=num_patches, 
                               is_training=is_training, 
                               color=color,
                               overwrite=overwrite)
    else:
        print('{0} training set has precomputed patches.'.format(dataset_name))

    # Extract random patches from the validation set    
    validation_dataset_path = path.join('data', dataset_name, 'validation');

    if not path.exists(path.join(validation_dataset_path, 'patches')) or overwrite:
        print('Extracting patches from {0} validation set...'.format(dataset_name))
        extract_random_patches(validation_dataset_path, 
                               patch_size=patch_size, 
                               num_patches=num_patches, 
                               is_training=is_training, 
                               color=color,
                               overwrite=overwrite)
    else:
        print('{0} validation set has precomputed patches.'.format(dataset_name))