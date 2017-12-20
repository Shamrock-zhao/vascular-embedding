
from os import path, makedirs, listdir
from scipy import misc

from data_preparation.util.image_processing import get_fov_mask



def generate_fov_masks(dataset_folder, threshold=0.01):

    # prepare paths
    image_folder = path.join(dataset_folder, 'images')
    masks_folder = path.join(dataset_folder, 'masks')

    # check if the folder exists
    if not path.exists(masks_folder):
        makedirs(masks_folder)

    # get image ids
    image_filenames = listdir(image_folder)

    # iterate for each image
    for i in range(0, len(image_filenames)):
        
        # open the image
        I = misc.imread(path.join(image_folder, image_filenames[i]))
        # get the FOV mask
        print('Processing image {} ({}/{})'.format(image_filenames[i], i + 1, len(image_filenames)))
        fov_mask = get_fov_mask(I, threshold)
        # write the FOV mask in disk
        misc.imsave(path.join(masks_folder, image_filenames[i][0:-3] + 'png'), fov_mask)



import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", help="path to the dataset (it has to contain an 'images' folder", type=str)
    parser.add_argument("--t", help="Threshold", type=float, default=0.01)

    args = parser.parse_args()

    generate_fov_masks(args.dataset_folder, args.t)