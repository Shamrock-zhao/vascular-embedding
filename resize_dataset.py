
import numpy as np

from os import path, makedirs, listdir
from scipy import misc

from data_preparation.util.image_processing import get_fov_mask



def resize_dataset(input_dataset_folder, output_dataset_folder, size_x=1444):

    # prepare input paths
    input_image_folder = path.join(input_dataset_folder, 'images')
    input_masks_folder = path.join(input_dataset_folder, 'masks')
    input_label_folder = path.join(input_dataset_folder, 'labels')
    if not path.exists(input_label_folder):
        input_label_folder = None

    # prepare output paths
    output_image_folder = path.join(output_dataset_folder, 'images')
    if not path.exists(output_image_folder):
        makedirs(output_image_folder)
    output_masks_folder = path.join(output_dataset_folder, 'masks')
    if not path.exists(output_masks_folder):
        makedirs(output_masks_folder)
    output_label_folder = path.join(output_dataset_folder, 'labels')
    if not path.exists(output_label_folder) and not (input_label_folder is None):
        makedirs(output_label_folder)

    # get image ids
    image_filenames = sorted(listdir(input_image_folder))
    masks_filenames = sorted(listdir(input_masks_folder))
    if not input_label_folder is None:
        labels_filenames = listdir(input_label_folder)

    # iterate for each image
    for i in range(0, len(image_filenames)):
        
        # get the FOV mask
        print('Processing image {} ({}/{})'.format(image_filenames[i], i + 1, len(image_filenames)))

        # open the image
        I = misc.imread(path.join(input_image_folder, image_filenames[i]))
        # open the mask
        fov_mask = misc.imread(path.join(input_masks_folder, masks_filenames[i]))
        if len(fov_mask.shape)==3:
            fov_mask = fov_mask[:,:,0] > 0
        # open the mask
        if not input_label_folder is None:
            labels = misc.imread(path.join(input_label_folder, labels_filenames[i]))

        # resizing factor
        resizing_factor = size_x / I.shape[0]

        # resize the image
        I = misc.imresize(I, resizing_factor, 'bilinear')

        if resizing_factor > 1:
            fov_mask = misc.imresize(fov_mask, resizing_factor), 'nearest')
            if not input_label_folder is None:
                labels = misc.imresize(labels, resizing_factor) > 128#, 'nearest')
        else:
            fov_mask = misc.imresize(fov_mask, resizing_factor), 'nearest')
            if not input_label_folder is None:
                labels = misc.imresize(labels, resizing_factor, 'nearest')
        print(I.shape)
        print(fov_mask.shape)

        # write the FOV mask in disk
        misc.imsave(path.join(output_image_folder, image_filenames[i]), I)
        misc.imsave(path.join(output_masks_folder, masks_filenames[i]), fov_mask)
        if not input_label_folder is None:
            misc.imsave(path.join(output_label_folder, labels_filenames[i]), labels)



import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset_folder", help="path to the input dataset (it has to contain an 'images' folder", type=str)
    parser.add_argument("output_dataset_folder", help="output data set path", type=str)
    parser.add_argument("--size_x", help="target size in the x axis", type=int, default=1444)

    args = parser.parse_args()
    print(args.size_x)

    resize_dataset(args.input_dataset_folder, args.output_dataset_folder, args.size_x)