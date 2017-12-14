
import numpy as np
import math


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


def sensitivity(gt, pred, fov_mask):
    tp_ = tp(gt, pred)
    fn_ = fn(gt, pred, fov_mask)
    return tp_ / (tp_ + fn_)


def specificity(gt, pred, fov_mask):
    tn_ = tn(gt, pred, fov_mask)
    fp_ = fp(gt, pred)
    return tn_ / (tn_ + fp_)


def precision(gt, pred, fov_mask):
    tp_ = tp(gt, pred)
    fp_ = fp(gt, pred)
    return tp_ / (tp_ + fp_)


def recall(gt, pred, fov_mask):
    return sensitivity(gt, pred, fov_mask)


def f1_score(gt, pred, fov_mask):
    pr_ = precision(gt, pred, fov_mask)
    re_ = recall(gt, pred, fov_mask)
    return 2 * (pr_ * re_) / (pr_ + re_)


def g_mean(gt, pred, fov_mask):
    se_ = sensitivity(gt, pred, fov_mask)
    sp_ = specificity(gt, pred, fov_mask)
    return math.sqrt(se_ * sp_)


def mcc(gt, pred, fov_mask):
    tp_ = tp(gt, pred)
    fn_ = fn(gt, pred, fov_mask)
    fp_ = fp(gt, pred)

    N = len(np.flatnonzero(fov_mask))
    S = (tp_ + fn_) / N
    P = (tp_ + fp_) / N

    return ((tp_ / N) - S * P) / math.sqrt(P * S * (1-S) * (1-P))



def tp(gt, pred):
    return len(np.flatnonzero(np.multiply(gt, pred)))

def tn(gt, pred, fov_mask):
    inverted_gt = np.multiply(np.invert(gt), fov_mask)
    inverted_pred = np.multiply(np.invert(pred), fov_mask)
    return len(np.flatnonzero(np.multiply(inverted_gt, inverted_pred)))

def fn(gt, pred, fov_mask):
    inverted_pred = np.multiply(np.invert(pred), fov_mask)
    return len(np.flatnonzero(np.multiply(gt, inverted_pred)))

def fp(gt, pred):
    return len(np.flatnonzero(np.multiply(np.invert(gt), pred)))



def dice_indices(gts, preds):
    
    dice_indices_ = []
    for i in range(0, preds.shape[0]):
        dice_indices_.append(dice_index(np.squeeze(gts[i,:,:], axis=0), np.squeeze(preds[i,:,:], axis=0)))
    return dice_indices_
