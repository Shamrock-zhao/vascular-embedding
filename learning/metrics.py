
import numpy as np


def dice_index(gt, pred):
    
    n_gt_pos_pixels = (np.flatnonzero(gt)).size
    n_pred_pos_pixels = (np.flatnonzero(pred)).size

    sum_ = n_gt_pos_pixels + n_pred_pos_pixels + 0.000001
    if sum_ == 0:
        return 0
    else:
        n_intersection = np.flatnonzero(np.logical_and(gt, pred)).size
        return 2 * n_intersection / sum_


def jaccard_index(gt, pred):
    
    gt = np.asarray(gt, dtype=np.uint8)
    pred = np.asarray(pred, dtype=np.uint8)

    n_union = (np.flatnonzero((gt + pred) > 0)).size + 0.000001
    n_intersection = (np.flatnonzero((gt + pred) > 1)).size 

    return n_intersection / n_union


def dice_indices(gts, preds):
    
    jaccard_indices_ = []
    for i in range(0, preds.shape[0]):
        dice_index.append(jaccard_index(np.squeeze(gt[i,:,:], axis=0), np.squeeze(preds[i,:,:], axis=0)))

