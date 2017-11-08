
from skimage import io
from os import path, makedirs, listdir, rename
import numpy as np
from scipy import misc
from .files_processing import natural_key
import matplotlib.pyplot as plt



def preprocess(image, fov_mask):
    
    return image



def pick_random_coordinate_inside_fov(fov_mask):
    
    # identify coordinates inside the FOV mask
    x, y = np.where(fov_mask)
    # pick up x,y index randomly
    i = np.random.randint(len(x))
    j = np.random.randint(len(y))
    # set FOV mask position in zero to avoid repetitions
    fov_mask[x[i], y[j]] = False

    return x[i], y[j], fov_mask



def extract_random_patches(dataset_folder, patch_size=64, num_patches=1000, is_training=True, color=True):
    
    # prepare input folders
    img_folder = path.join(dataset_folder, 'images')
    fov_masks_folder = path.join(dataset_folder, 'masks')

    # get image and fov masks filenames from input folder
    image_filenames = sorted(listdir(img_folder), key=natural_key)
    fov_masks_filenames = sorted(listdir(fov_masks_folder), key=natural_key)
    
    # initialize output folder
    output_image_folder = path.join(dataset_folder, 'patches')
    if not path.exists(output_image_folder):
        makedirs(output_image_folder)

    # if training data, get also the labels filenames and prepare the output folder
    if is_training:
        gt_folder = path.join(dataset_folder, 'labels')
        gt_filenames = sorted(listdir(gt_folder), key=natural_key)
        output_labels_folder = path.join(dataset_folder, 'labels_patches')
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
        # initialize a padded fov
        sizes = fov_mask.shape
        padded_fov = np.zeros(sizes, dtype=bool)
        padded_fov[ pad:sizes[0]-pad, pad:sizes[1]-pad ] = fov_mask[ pad:sizes[0]-pad, pad:sizes[1]-pad ] > 0
        
        # extract N_subimgs patches
        for j in range(0, num_patches):
            
            # pick random coordinate inside the fov
            x, y, padded_fov = pick_random_coordinate_inside_fov(padded_fov)

            # get a random patch around the random coordinate
            if color:
                random_patch = image[x-pad:x+pad, y-pad:y+pad, :]
            else:
                random_patch = image[x-pad:x+pad, y-pad:y+pad, 1]
            
            # save the patch
            misc.imsave(path.join(output_image_folder, current_image_filename[:-4] + str(j) + '.png'), random_patch)
            
            # if is training, crop the label as well
            if is_training:
                
                # identify current image
                current_gt_filename = gt_filenames[i]
                # open the image
                labels = (misc.imread(path.join(gt_folder, current_gt_filename)) / 255).astype('int32')
                random_patch_labels = labels[x-pad : x+pad, y-pad : y+pad]
                misc.imsave(path.join(output_labels_folder, current_gt_filename[:-4] + str(j) + '.gif'), 
                    random_patch_labels)                



def extract_patches_from_training_datasets(dataset_name, patch_size=64, num_patches=1000, is_training=True, color=True):
    
    # Extract random patches from training/validation data sets
    training_dataset_path = path.join('data', dataset_name, 'training')

    if not path.exists(path.join(training_dataset_path, 'patches')):
        print('Extracting patches from {0} training set...'.format(dataset_name))
        extract_random_patches(training_dataset_path, 
                               patch_size=patch_size, 
                               num_patches=num_patches, 
                               is_training=is_training, 
                               color=color)
    else:
        print('{0} training set has precomputed patches.'.format(dataset_name))

    # Extract random patches from the validation set    
    validation_dataset_path = path.join('data', dataset_name, 'validation');

    if not path.exists(path.join(validation_dataset_path, 'patches')):
        print('Extracting patches from {0} validation set...'.format(dataset_name))
        extract_random_patches(validation_dataset_path, 
                               patch_size=patch_size, 
                               num_patches=num_patches, 
                               is_training=is_training, 
                               color=color)
    else:
        print('{0} validation set has precomputed patches.'.format(dataset_name))