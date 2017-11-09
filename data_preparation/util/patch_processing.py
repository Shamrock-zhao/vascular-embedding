
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from os import path, makedirs, listdir, rename
from scipy import misc
from shutil import rmtree
from .files_processing import natural_key
from .image_processing import equalize_fundus_image_intensities
from skimage import measure



def preprocess(image, fov_mask, preprocessing=None):
    
    if preprocessing == 'rgb':
        preprocessed_image = image # RGB image

    elif preprocessing == 'green':
        preprocessed_image = image[:,:,1] # Green band

    elif preprocessing == 'equalized':
        preprocessed_image = equalize_fundus_image_intensities(np.copy(image), fov_mask) # RGB equalized

    '''
    image_size = preprocessed_image.shape
    for i in range(0, image_size[2]):
        preprocessed_image[:,:,i] = np.multiply(preprocessed_image[:,:,i], fov_mask)
    '''
    return preprocessed_image



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



def get_coordinates_from_mask(mask, pad=0):
    # pad the mask if necessary
    if pad==0:
        padded_mask = mask
    else:
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
    return coordinates



def extract_random_patches_from_image(image, labels, fov_mask, coordinates, filename, image_patch_folder, 
    labels_patch_folder, patch_size=64, num_patches=1000):

    '''
    Given an image, its vessel labelling and FOV mask and a series of coordinates to sample,
    extract num_patches squared patches of length patch_size and save them in
    image_patch_folder and labels_patch_folder
    '''

    # preprocess the images using different methods
    rgb = preprocess(image, fov_mask, 'rgb')
    equalized_image = preprocess(np.copy(image), fov_mask, 'equalized')

    # precompute pad
    pad = int(patch_size/2)

    # extract N_subimgs patches
    for j in range(0, num_patches):
        
        # pick random coordinate
        x, y, coordinates = pick_random_coordinate(coordinates)

        # get a patch around the random coordinate
        # from original RGB image
        random_patch = rgb[x-pad:x+pad, y-pad:y+pad, :]

        misc.imsave(path.join(image_patch_folder + '_rgb', filename[:-4] + str(j) + '.png'), random_patch)
        # from equalized RGB image
        random_patch = equalized_image[x-pad:x+pad, y-pad:y+pad, :]
        misc.imsave(path.join(image_patch_folder + '_eq', filename[:-4] + str(j) + '.png'), random_patch)
        
        # crop the label as well
        random_patch_labels = labels[x-pad : x+pad, y-pad : y+pad]
        misc.imsave(path.join(labels_patch_folder, filename[:-4] + str(j) + '.gif'), random_patch_labels)



def replace_folder(folder):
    if path.exists(folder):
        rmtree(folder)
    makedirs(folder)



def extract_random_patches_from_dataset(dataset_folder, patch_size=64, num_patches=1000):
    
    # prepare input folders
    img_folder = path.join(dataset_folder, 'images')
    fov_masks_folder = path.join(dataset_folder, 'masks')

    # get image and fov masks filenames from input folder
    image_filenames = sorted(listdir(img_folder), key=natural_key)
    fov_masks_filenames = sorted(listdir(fov_masks_folder), key=natural_key)
    # get also the labels filenames
    gt_folder = path.join(dataset_folder, 'labels')
    gt_filenames = sorted(listdir(gt_folder), key=natural_key)

    # initialize output folders based name
    output_base_image_folder = path.join(dataset_folder, 'patches')
    output_base_labels_folder = path.join(dataset_folder, 'patches_labels')
    
    # sampling strategies
    sampling_strategies = ['uniform', 'guided-by-labels']

    # precompute pad
    pad = int(patch_size/2)

    # extract patches according to each sampling strategy
    for s in range(0, len(sampling_strategies)):

        # prepare output folders
        output_image_folder = output_base_image_folder + '_' + sampling_strategies[s]
        output_labels_folder = output_base_labels_folder + '_' + sampling_strategies[s]

        # prepare output folders for patches extracted with different
        # image preprocessing methods
        replace_folder(output_image_folder + '_rgb')
        replace_folder(output_image_folder + '_eq')
        replace_folder(output_labels_folder)
    
        # for each image
        for i in range(0, len(image_filenames)):

            # identify current image
            current_image_filename = image_filenames[i]
            current_mask_filename = fov_masks_filenames[i]
            # identify current labels
            current_gt_filename = gt_filenames[i]

            print('Processing image ' + current_image_filename)

            # open image, labels and FOV
            image = misc.imread(path.join(img_folder, current_image_filename))
            labels = misc.imread(path.join(gt_folder, current_gt_filename)) > 0
            fov_mask = misc.imread(path.join(fov_masks_folder, current_mask_filename))  > 0
            if len(fov_mask.shape) > 2:
                fov_mask = fov_mask[:,:,0]

            # random sample the patches
            if sampling_strategies[s]=='uniform':
                
                print('Sampling patches uniformly...')

                # the valid coordinates will be inside the FOV and out of the
                # padded region
                coordinates = get_coordinates_from_mask(fov_mask, pad)
                # randomly sample num_patches patches from the image and labels
                extract_random_patches_from_image(image, labels, fov_mask, coordinates, current_image_filename, output_image_folder, 
                    output_labels_folder, patch_size, num_patches)

            elif sampling_strategies[s]=='guided-by-labels':

                print('Sampling patches guided by labels...')

                # the valid coordinates will be inside the labels, first
                coordinates = get_coordinates_from_mask(labels, pad)
                # randomly sample patches from the image and labels
                extract_random_patches_from_image(image, labels, fov_mask, coordinates, current_image_filename, output_image_folder, 
                    output_labels_folder, patch_size, num_patches // 2)

                # the valid coordinates will be inside the labels, first
                coordinates = get_coordinates_from_mask(np.multiply((1 - labels) > 0, fov_mask), pad)
                # randomly sample num_patches // 2 patches from the image and labels
                extract_random_patches_from_image(image, labels, fov_mask, coordinates, current_image_filename, output_image_folder, 
                    output_labels_folder, patch_size, num_patches // 2)