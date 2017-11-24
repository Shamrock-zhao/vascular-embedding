
import torch
import numpy as np

from os import makedirs, listdir, path
from scipy import misc
from data_preparation.util.image_processing import preprocess


def predict(image_path, fov_path, output_path, model_filename, image_preprocessing='rgb'):
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
        scores, segmentation = model.module.predict_from_full_image(img)
        scores = np.multiply(scores, fov_mask > 0)
        segmentation = np.multiply(segmentation, fov_mask > 0)
        # save both files
        misc.imsave(path.join(scores_path, current_img_filename[:-3] + 'png'), scores)
        misc.imsave(path.join(segmentations_path, current_img_filename[:-3] + 'png'), segmentation)



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

    args = parser.parse_args()

    # call the main function
    predict(args.image_path, args.fov_path, args.output_path, args.model_full_path, args.image_preprocessing)