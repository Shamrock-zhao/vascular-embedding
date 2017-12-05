
from configparser import ConfigParser
from os import makedirs, path


def modify_and_write_config_file(config_files_path, default_config_file, experiment_name, to_modify, new_values):
    
    # open the configuration file
    parser = ConfigParser()
    parser.read(default_config_file)

    # initialize all the configuration files
    output_config_files_folder = path.join(config_files_path, experiment_name)
    if not path.exists(output_config_files_folder):
        makedirs(output_config_files_folder)

    # create a file per each sampling strategy
    for i in range(0, len(new_values)):

        # update the experiment name and the sampling strategy        
        parser.set('experiment', 'name', experiment_name + '-' + new_values[i])
        parser.set(to_modify[0], to_modify[1], new_values[i])
        # save the experiment
        with open(path.join(output_config_files_folder, new_values[i] + '.ini'), 'w') as output_file:
            parser.write(output_file)  
    


import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="root path to save the configuration files", type=str)
    parser.add_argument("default_config_file", help="fullpath to a default configuration", type=str)
    parser.add_argument("parameter", help="parameter to tune", type=str)

    args = parser.parse_args()

    if args.parameter == 'image-preprocessing':
        
        experiment_name = 'exp_eval_image_preprocessing'
        to_modify = ['experiment', 'image-preprocessing']
        parameters_to_test = ['rgb', 'eq', 'clahe']

    elif args.parameter == 'sampling-strategy':

        experiment_name = 'exp_eval_sampling_strategy'
        to_modify = ['experiment', 'sampling-strategy']
        parameters_to_test = ['uniform', 'guided-by-labels']

    elif args.parameter == 'dropout':
        
        experiment_name = 'exp_eval_dropout'
        to_modify = ['architecture', 'dropout']
        parameters_to_test = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']

    elif args.parameter == 'batch-norm':
        
        experiment_name = 'exp_eval_batch_norm'
        to_modify = ['architecture', 'batch-norm']
        parameters_to_test = ['False', 'True']

    elif args.parameter == 'augmented':
        
        experiment_name = 'exp_eval_augmented'
        to_modify = ['training', 'augmented']
        parameters_to_test = ['False', 'True']

    else:
        raise NameError('Unsuported parameter to tune')

    modify_and_write_config_file(args.path, args.default_config_file, experiment_name, to_modify, parameters_to_test)

    

