
import numpy as np

from os import path, listdir, makedirs
from data_preparation.util.files_processing import natural_key

from learning import metrics

from scipy import misc


def quantitative_evaluation(segmentation_path, ground_truth_path, fov_masks_path):
    
    # get segmentation and ground truth filenames
    segm_files = sorted(listdir(segmentation_path), key=natural_key)
    gt_files = sorted(listdir(ground_truth_path), key=natural_key)
    fov_files = sorted(listdir(fov_masks_path), key=natural_key)

    # initialize arrays for quantitative metrics
    se_ = np.zeros((len(segm_files), 1))
    sp_ = np.zeros((len(segm_files), 1))
    pr_ = np.zeros((len(segm_files), 1))
    dice_ = np.zeros((len(segm_files), 1))
    mcc_ = np.zeros((len(segm_files), 1))
    g_mean_ = np.zeros((len(segm_files), 1))

    # iterate for each image
    for i in range(0, len(segm_files)):
        
        # open segmentation, labels and masks
        segm = misc.imread(path.join(segmentation_path, segm_files[i])) > 0
        gt = misc.imread(path.join(ground_truth_path, gt_files[i])) > 0
        fov_mask = misc.imread(path.join(fov_masks_path, fov_files[i])) > 0

        # compute the evaluation metrics
        se_[i] = metrics.sensitivity(gt, segm, fov_mask)
        sp_[i] = metrics.specificity(gt, segm, fov_mask)
        pr_[i] = metrics.precision(gt, segm, fov_mask)
        dice_[i] = metrics.dice_index(gt, segm)
        mcc_[i] = metrics.mcc(gt, segm, fov_mask)
        g_mean_[i] = metrics.g_mean(gt, segm, fov_mask)

    return se_, sp_, pr_, dice_, mcc_, g_mean_, segm_files



import argparse
import sys
import csv

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("segmentation_path", help="path to the segmentations", type=str)
    parser.add_argument("ground_truth_path", help="path to the ground truth labels", type=str)
    parser.add_argument("fov_masks_path", help="path to the FOV masks", type=str)
    parser.add_argument("output_path", help="output path", type=str)

    args = parser.parse_args()

    # call the main function
    se_, sp_, pr_, dice_, mcc_, g_mean_, segm_files = quantitative_evaluation(args.segmentation_path, args.ground_truth_path, args.fov_masks_path)

    # initialize output folder
    if not path.exists(args.output_path):
        makedirs(args.output_path)

    # write results in a CSV file
    with open(path.join(args.output_path, 'quantitative_evaluation.csv'), 'w', newline='\n') as csvfile:
        results_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
        results_writer.writerow(['Filename', 'Sensitivity', 'Specificity', 'Precision', 'Dice', 'MCC', 'G-mean'])
        
        for i in range(0, len(segm_files)):
            results_writer.writerow([segm_files[i], se_[i], sp_[i], pr_[i], dice_[i], mcc_[i], g_mean_[i]])
        
        results_writer.writerow(['Average', np.mean(se_), np.mean(sp_), np.mean(pr_), np.mean(dice_), np.mean(mcc_), np.mean(g_mean_)])
        results_writer.writerow(['Std', np.std(se_), np.std(sp_), np.std(pr_), np.std(dice_), np.std(mcc_), np.std(g_mean_)])
