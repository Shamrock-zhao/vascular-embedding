
import torch

import numpy as np
import matplotlib.pyplot as plt

from scipy import misc
from os import path, makedirs, listdir
from predict import segment_image

from ast import literal_eval as make_tuple

from learning.metrics import dice_index

from data_preparation.util.files_processing import natural_key


def crf_parameter_tuning(validation_data_path, model_filename, compat_vals, sxy_vals, schan_vals, pairwise_feature, image_preprocessing='rgb'):
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
    images_filenames = sorted(listdir(images_folder), key=natural_key)
    fov_masks_filenames = sorted(listdir(fov_masks_folder), key=natural_key)
    labels_filenames = sorted(listdir(labels_folder), key=natural_key)

    # prepare arrays of values to explore
    #compat = np.arange(compat_vals[0], compat_vals[1], compat_vals[2])
    compat = 1 / 10**np.arange(0, 6, 1)
    sxy = np.arange(sxy_vals[0], sxy_vals[1], sxy_vals[2])
    schan = np.arange(schan_vals[0], schan_vals[1], schan_vals[2])

    # initialize numpy array of quality measurements
    dice_coefficients = np.zeros((len(compat), len(sxy), len(schan)), dtype=np.float32)

    # grid search of parameters
    for i in range(0, len(compat)):
        for j in range(0, len(sxy)):
            for k in range(0, len(schan)):
                
                # print the current configuration
                print('Trying with: compat={}, sxy={}, schan={}'.format(str(compat[i]), str(sxy[j]), str(schan[k])))
                compat_ = compat[i]
                sxy_ = sxy[j]
                schan_ = schan[k]
                # initialize an array of dice coefficients
                current_dice_coefficients = np.zeros((len(images_filenames), 1), dtype=np.float32)

                # segment each image and compute the dice coefficient
                for ii in range(0, len(images_filenames)):
                    
                    # open image, mask and label
                    print('image {}/{}'.format(ii + 1, len(images_filenames)))
                    img = np.asarray(misc.imread(path.join(images_folder, images_filenames[ii])), dtype=np.uint8)
                    fov_mask = np.asarray(misc.imread(path.join(fov_masks_folder, fov_masks_filenames[ii])), dtype=np.uint8) // 255
                    labels = np.asarray(misc.imread(path.join(labels_folder, labels_filenames[ii])), dtype=np.uint8) // 255
                    # segment the image
                    _, segmentation, _ = segment_image(img, fov_mask, model, image_preprocessing, True, 2, sxy_, schan_, compat_, pairwise_feature)
                    # evaluate
                    current_dice_coefficients[ii] = dice_index(labels > 0, segmentation > 0)

                # assign the average dice coefficient to this configuration
                dice_coefficients[i,j,k] = np.mean(current_dice_coefficients)
                print('Dice = {}'.format(str(dice_coefficients[i,j,k])))

    matrix_to_plot = dice_coefficients[0,:,:]
    #matrix_to_plot = dice_coefficients[0,:,:]
    plt.imshow(matrix_to_plot, cmap='coolwarm')
    plt.show()

    # get the best configuration
    index = np.argmax(dice_coefficients)
    i,j,k = np.unravel_index(index, dice_coefficients.shape)

    dice_coefficient = np.max(dice_coefficients)
    compat_best = compat[i]
    sxy_best = sxy[j]
    schan_best = schan[k]

    # return it
    return compat_best, sxy_best, schan_best, dice_coefficient, dice_coefficients



import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_data_path", help="path to the validation data", type=str)
    parser.add_argument("model_filename", help="fullpath to a pretrained model", type=str)
    parser.add_argument("image_preprocessing", help="image preprocessing strategy", type=str)
    parser.add_argument("output_path", help="full path to save the results", type=str)
    parser.add_argument("--pairwise_feature", help="pairwise feature", type=str, default='rgb')
    parser.add_argument("--compat_vals", help="compat parameter search space", type=str, default='(1,2,1)')
    parser.add_argument("--sxy_vals", help="sxy parameter search space", type=str, default='(1,30,1)')
    parser.add_argument("--schan_vals", help="schan parameter search space", type=str, default='(1,30,1)')
    
    args = parser.parse_args()

    # get the data set name
    print(args.validation_data_path)
    dataset_name = args.validation_data_path.split('/')
    if dataset_name[-1]=='':
        dataset_name = dataset_name[-3]
    else:
        dataset_name = dataset_name[-2]
    print(dataset_name)
    print('Evaluation on {} data set'.format(dataset_name))

    # make tuples
    args.sxy_vals = make_tuple(args.sxy_vals) 
    args.schan_vals = make_tuple(args.schan_vals)
    args.compat_vals = make_tuple(args.compat_vals)

    # tune parameters
    compat_best, sxy_best, srgb_best, dice_coefficient, dice_coefficients = crf_parameter_tuning(args.validation_data_path, 
                                                                                                 args.model_filename, 
                                                                                                 args.compat_vals,
                                                                                                 args.sxy_vals, 
                                                                                                 args.schan_vals, 
                                                                                                 args.pairwise_feature,
                                                                                                 args.image_preprocessing)
    # write best configurations on disc
    if not path.exists(args.output_path):
        makedirs(args.output_path)
    with open(path.join(args.output_path, dataset_name + '_' + args.pairwise_feature + '_crf_optimization.txt'), 'w') as file:
        file.write('Dataset: ' + dataset_name + '\n')
        file.write('compat: ' + str(compat_best) + '\n')
        file.write('sxy: ' + str(sxy_best) + '\n')
        file.write('srgb: ' + str(srgb_best) + '\n')
        file.write('Dice coefficient: ' + str(dice_coefficient) + '\n')