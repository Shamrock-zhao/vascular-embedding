
from configparser import ConfigParser
from os import makedirs, path



def exp_eval_sampling_strategy(config_files_path, default_config_file):

    # sampling strategies available
    sampling_strategies = ['uniform', 'guided-by-labels']
    
    # open the configuration files
    parser = ConfigParser()
    parser.read(default_config_file)

    # initialize all the configuration files
    output_config_files_folder = path.join(config_files_path, 'exp_eval_sampling_strategy')
    if not path.exists(output_config_files_folder):
        makedirs(output_config_files_folder)

    # create a file per each sampling strategy
    for i in range(0, len(sampling_strategies)):

        # update the experiment name and the sampling strategy        
        parser.set('experiment', 'name', sampling_strategies[i])
        parser.set('experiment', 'sampling-strategy', sampling_strategies[i])
        # save the experiment
        with open(path.join(output_config_files_folder, sampling_strategies[i] + '.ini'), 'w') as output_file:
            parser.write(output_file)



import argparse
import sys

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="configuration files path", type=str)
    parser.add_argument("default_config_file", help="fullpath to a default configuration", type=str)

    args = parser.parse_args()

    # call the main function
    exp_eval_sampling_strategy(args.path, args.default_config_file)

