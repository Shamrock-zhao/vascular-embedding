
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



def exp_eval_sampling_strategy(config_files_path, default_config_file):

    experiment_name = 'exp_eval_sampling_strategy'
    to_modify = ['experiment', 'sampling-strategy']
    sampling_strategies = ['uniform', 'guided-by-labels']

    modify_and_write_config_file(config_files_path, default_config_file, experiment_name, to_modify, sampling_strategies)



def exp_eval_image_preprocessing(config_files_path, default_config_file):
    
    experiment_name = 'exp_eval_image_preprocessing'
    to_modify = ['experiment', 'image-preprocessing']
    preprocessing_strategies = ['rgb', 'eq', 'clahe']

    modify_and_write_config_file(config_files_path, default_config_file, experiment_name, to_modify, preprocessing_strategies)
    


import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="configuration files path", type=str)
    parser.add_argument("default_config_file", help="fullpath to a default configuration", type=str)
    parser.add_argument("parameter", help="parameter to tune", type=str)

    args = parser.parse_args()

    if args.parameter == 'image-preprocessing':
        exp_eval_image_preprocessing(args.path, args.default_config_file)
    elif args.parameter == 'sampling-strategy':
        exp_eval_sampling_strategy(args.path, args.default_config_file)        
    else:
        raise NameError('Unsuported parameter to tune')

    

