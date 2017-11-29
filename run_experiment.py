
import json
import numpy as np

from train import train
from predict import predict, evaluate_predictions

from os import path, makedirs
from glob import glob
from configparser import ConfigParser

from data_preparation.util.files_processing import natural_key



def run_experiment(experiment_folder, validation_set_path, output_path, evaluate=True, load_weights=False, crf=False):

    # retrieve configuration files names
    config_filenames = sorted(glob(experiment_folder + '*.ini'), key=natural_key)

    # get experiment id
    experiment_id = experiment_folder.split('/')[-1]
    if experiment_id == '':
        experiment_id = experiment_folder.split('/')[-2]

    # setup the validation set folders
    val_image_path = path.join(validation_set_path, 'images')
    val_image_labels = path.join(validation_set_path, 'labels')
    val_fov_mask_path = path.join(validation_set_path, 'masks')
    # setup the output path by adding the experiment id
    output_path = path.join(output_path, experiment_id)
    if not path.exists(output_path):
        makedirs(output_path)

    # if necessary, initialize an array of mean dice coefficients
    if evaluate:
        mean_performance_per_experiment = np.zeros((len(config_filenames), 1), dtype=np.float32)

    # run each experiment in the folder
    for i in range(0, len(config_filenames)):
        
        parser = ConfigParser()
        parser.read(config_filenames[i])

        # get the data set name
        dataset_name = parser['folders']['data-path'].split('/')[-1]
        if dataset_name == '':
            dataset_name = parser['folders']['data-path'].split('/')[-2]

        # create a folder with the dataset name
        current_output_path = path.join(output_path, dataset_name, parser['experiment']['name'])
        if not path.exists(current_output_path):
            makedirs(current_output_path)

        # train the model
        model_filename = train(config_filenames[i], load_weights)
        # evaluate on the validation set
        predict(val_image_path, val_fov_mask_path, current_output_path, model_filename, 
                parser['experiment']['image-preprocessing'], crf)
        
        # if necessary, evaluate the results
        if evaluate:
            mean_performance_per_experiment[i], _ = evaluate_predictions(path.join(current_output_path, 'segmentations'), val_image_labels)
    
    # if necessary, write the results on a log file in the root folder
    if evaluate:
        with open(path.join(output_path, dataset_name, experiment_id + '.txt'), 'w') as file:
            for i in range(0, len(config_filenames)):
                file.write(config_filenames[i] + ' - Dice: ' + str(mean_performance_per_experiment[i]))



import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder", help="path to the configuration files of the experiment", type=str)
    parser.add_argument("validation_set_path", help="path to the validation data", type=str)
    parser.add_argument("output_path", help="path to save the results", type=str)
    parser.add_argument("--evaluate", help="boolean indicating if each experiment must be evaluated", type=str, default='True')
    parser.add_argument("--load_weights", help="load pretrained weights when available", type=str, default='False')
    parser.add_argument("--crf", help="CRF refinement", type=str, default='False')

    args = parser.parse_args()

    run_experiment(args.experiment_folder, args.validation_set_path, args.output_path, 
                   args.evaluate.upper()=='TRUE', args.load_weights.upper()=='TRUE', args.crf.upper()=='TRUE')