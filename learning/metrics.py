
import numpy as np


def dice_index(gt, pred):
    
    n_gt_pos_pixels = (np.flatnonzero(gt)).size
    n_pred_pos_pixels = (np.flatnonzero(pred)).size

    sum_ = n_gt_pos_pixels + n_pred_pos_pixels
    if sum_ == 0:
        return 0
    else:
        n_intersection = np.flatnonzero(np.logical_and(gt, pred)).size
        return 2 * n_intersection / sum_


def dice_indices(gts, preds):
    
    dice_indices_ = []
    for i in range(0, preds.shape[0]):
        dice_indices_.append(dice_index(np.squeeze(gt[i,:,:], axis=0), np.squeeze(preds[i,:,:], axis=0)))
    return dice_indices_

