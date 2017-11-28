
import numpy as np

from scipy import misc
from os import path, makedirs, listdir
from predict import segment_image

from ast import literal_eval as make_tuple

from learning.metrics import dice_index


def crf_parameter_tuning(validation_data_path, model_filename, compat_vals, sxy_vals, srgb_vals, image_preprocessing='rgb'):
    '''
        compat_vals = (start, stop, step)
    '''

    # prepare folders
    images_folder = path.join(validation_data_path, 'images')
    fov_masks_folder = path.join(validation_data_path, 'masks')
    labels_folder = path.join(validation_data_path, 'labels')

    # open the model
    model = torch.load(model_filename)
    model.eval()

    # get filenames
    images_filenames = listdir(images_folder)
    fov_masks_filenames = listdir(fov_masks_folder)
    labels_filenames = listdir(labels_folder)

    # prepare arrays of values to explore
    compat = np.arange(compat_vals[0], compat_vals[1], compat_vals[2])
    sxy = np.arange(sxy_vals[0], sxy_vals[1], sxy_vals[2])
    srgb = np.arange(srgb_vals[0], srgb_vals[1], srgb_vals[2])

    # initialize numpy array of quality measurements
    dice_coefficients = np.zeros((len(compat), len(sxy), len(srgb)), dtype=np.float32)

    # grid search of parameters
    for i in range(0, len(compat)):
        for j in range(0, len(sxy)):
            for k in range(0, len(srgb)):
                
                # get current configuration
                compat_ = compat[i]
                sxy_ = sxy[j]
                srgb_ = srgb[k]
                # initialize an array of dice coefficients
                current_dice_coefficients = np.zeros((len(images_filenames), 1), dtype=np.float32)

                # segment each image and compute the dice coefficient
                for ii in range(0, len(images_filenames)):
                    
                    # open image, mask and label
                    img = np.asarray(misc.imread(path.join(images_folder, images_filenames[ii])), dtype=np.uint8)
                    fov_mask = np.asarray(misc.imread(path.join(fov_masks_folder, fov_masks_filenames[ii])), dtype=np.uint8) // 255
                    labels = np.asarray(misc.imread(path.join(labels_folder, labels_filenames[ii])), dtype=np.uint8) // 255
                    # segment the image
                    _, segmentation, _ = segment_image(img, fov_mask, model, image_preprocessing, True)
                    # evaluate
                    current_dice_coefficients[ii] = dice_index(labels > 0, segmentation > 0)

                # assign the average dice coefficient to this configuration
                dice_coefficients[i,j,k] = np.mean(current_dice_coefficients)

    # get the best configuration
    best_configuration = np.argmax(dice_coefficients)
    compat_best = best_configuration[0]
    sxy_best = best_configuration[1]
    srgb_best = best_configuration[2]

    # return it
    return compat_best, sxy_best, srgb_best



import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_data_path", help="path to the validation data", type=str)
    parser.add_argument("model_filename", help="fullpath to a pretrained model", type=str)
    parser.add_argument("image_preprocessing", help="image preprocessing strategy", type=str)
    parser.add_argument("--compat_vals", help="compat parameter search space", type=str, default='(0,100,10)')
    parser.add_argument("--sxy_vals", help="sxy parameter search space", type=str, default='(30,100,10)')
    parser.add_argument("--srgb_vals", help="parameter to tune", type=str, default='(3,6,1)')
    
    args = parser.parse_args()

    # make tuples
    args.compat_vals = make_tuple(args.compat_vals)
    args.sxy_vals = make_tuple(args.sxy_vals) 
    args.srgb_vals = make_tuple(args.srgb_vals)

    # tune parameters
    return crf_parameter_tuning(args.validation_data_path, 
                                args.model_filename, 
                                args.compat_vals, 
                                args.sxy_vals, 
                                args.srgb_vals, 
                                image_preprocessing):