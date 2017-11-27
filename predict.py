
import torch
import numpy as np

import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import unary_from_softmax
from os import makedirs, listdir, path
from scipy import misc
from data_preparation.util.image_processing import preprocess


def predict(image_path, fov_path, output_path, model_filename, image_preprocessing='rgb', crf=True):
    '''
    '''

    assert image_preprocessing in ['rgb', 'eq', 'clahe'], "Unsuported image preprocessing."

    # retrieve images and fov masks filenames
    img_filenames = listdir(image_path)
    fov_filenames = listdir(fov_path)

    # open the model
    model = torch.load(model_filename)
    model.eval()

    # initialize the output folders
    scores_path = path.join(output_path, 'scores')
    segmentations_path = path.join(output_path, 'segmentations')
    if not path.exists(scores_path):
        makedirs(scores_path)
    if not path.exists(segmentations_path):
        makedirs(segmentations_path)

    # iterate for each img filename
    for i in range(0, len(img_filenames)):
        
        # get current filename
        current_img_filename = img_filenames[i]
        current_fov_filename = fov_filenames[i]

        print('Processing image {}'.format(current_img_filename))

        # open the image and the fov mask
        img = np.asarray(misc.imread(path.join(image_path, current_img_filename)), dtype=np.uint8)
        fov_mask = np.asarray(misc.imread(path.join(fov_path, current_fov_filename)), dtype=np.int32) // 255
        # preprocess the image according to the model
        img = preprocess(img, fov_mask, image_preprocessing)  

        # predict the scores
        scores, segmentation, unary_potentials = model.module.predict_from_full_image(img)
        scores = np.multiply(scores, fov_mask > 0)
        
        if crf:
            segmentation = crf_refinement(unary_potentials, img)
        else:
            segmentation = np.multiply(segmentation, fov_mask > 0)

        # save both files
        misc.imsave(path.join(scores_path, current_img_filename[:-3] + 'png'), scores)
        misc.imsave(path.join(segmentations_path, current_img_filename[:-3] + 'png'), segmentation)


def crf_refinement(unary_potentials, image, n_labels=2):
    
    # initialize the dense crf
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_labels)

    # get negative log probabilities
    #U = unary_from_softmax(unary_potentials)
    U = - (np.reshape(unary_potentials,(n_labels, image.shape[0] * image.shape[1])))

    # set unary potentials
    d.setUnaryEnergy(U)
    # this adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(0.01, 0.01, 0.01), rgbim=image,
                           compat=10000,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Run five inference steps.
    Q = d.inference(10)
    # Find out the most probable class for each pixel.
    MAP = np.asarray(np.argmax(Q, axis=0).reshape(image.shape[0], image.shape[1]), dtype=np.float32);
    return MAP


import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="image path", type=str)
    parser.add_argument("fov_path", help="fov mask path", type=str)
    parser.add_argument("output_path", help="path to save the output files", type=str)
    parser.add_argument("model_full_path", help="path to the model", type=str)
    parser.add_argument("--image_preprocessing", help="image preprocessing strategy", type=str, default='rgb')
    parser.add_argument("--crf", help="boolean indicating if CRF refinement must be used", type=str, default='True')

    args = parser.parse_args()

    # call the main function
    predict(args.image_path, args.fov_path, args.output_path, args.model_full_path, args.image_preprocessing, args.crf.upper()=='TRUE')