
import os
import numpy as np

from scipy import misc
from functools import partial
from utils import get_square, normalize
import random


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    ids = (f[:-4] for f in os.listdir(dir))
    return ids



def encode_training_validation_data(training_data, validation_data):
    random.seed(7)
    training_data = list(training_data)
    validation_data = list(validation_data)
    random.shuffle(training_data)
    random.shuffle(validation_data)
    return {'train': training_data, 'val': validation_data}



def open_img(ids, dir, suffix, to_squeeze=False):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        im = misc.imread(os.path.join(dir, id + suffix))
        if to_squeeze:
            im = im[:,:,0] // 255
        yield im



def get_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""

    imgs = open_img(ids, dir_img, '.png')
    # need to transform from HWC to CHW
    imgs_switched = map(partial(np.transpose, axes=[2, 0, 1]), imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = open_img(ids, dir_mask, '.gif', True)

    return zip(imgs_normalized, masks)



def get_full_img_and_mask(id, dir_img, dir_mask):
    im = misc.imread(os.path.join(dir_img, id + '.png'))
    mask = misc.imread(os.path.join(dir_mask, id + '.gif')) > 0
    return im, mask[:,:,0]