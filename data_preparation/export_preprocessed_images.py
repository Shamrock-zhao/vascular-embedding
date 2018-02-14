
from os import path, listdir, makedirs
from scipy import misc

from util.files_processing import natural_key
from util.image_processing import preprocess


def export_preprocessed_images(root_path='../data', datasets=['DRIVE', 'STARE', 'CHASEDB1', 'HRF']):
  '''

  '''
  
  # prepare subsets
  subsets = ['training', 'validation', 'test']
  # prepare suported preprocessing strategies
  preprocessing_strategies = ['rgb', 'clahe', 'eq', 'green']

  # iterate for each data set
  for i in range(0, len(datasets)):
    
    # get current database
    current_dataset = datasets[i]
    print('Database {}'.format(current_dataset))

    # if there is a subpath, then we can iterate for each of them...
    if path.exists(path.join(root_path, current_dataset, subsets[0])):

      # iterate for each subset
      for j in range(0, len(subsets)):
        
        current_subset = subsets[j]

        if not path.exists(path.join(root_path, current_dataset, current_subset)):

          print('Subset {} doesnt exist. Skipping...'.format(current_subset))

        else:
              
          print('Subset {}'.format(current_subset)) 

          # get the folders for the images and the masks
          current_image_folder = path.join(root_path, current_dataset, current_subset, 'images')
          current_mask_folder = path.join(root_path, current_dataset, current_subset, 'masks')
          # get image filenames
          image_filenames = sorted(listdir(current_image_folder), key=natural_key)
          # get mask filenames
          mask_filenames = sorted(listdir(current_mask_folder), key=natural_key)

          # for each preprocessing strategy
          for k in range(0, len(preprocessing_strategies)):
            
            print('Preprocessing strategy {}'.format(preprocessing_strategies[k]))

            # prepare the output folder
            output_folder = current_image_folder + '_' + preprocessing_strategies[k]
            if not path.exists(output_folder):
              makedirs(output_folder)

            # and now, loop for each image
            for ii in range(0, len(image_filenames)):
              
              print('Image {} being processed with {}'.format(image_filenames[ii], preprocessing_strategies[k]))

              # read the image
              image = misc.imread(path.join(current_image_folder, image_filenames[ii]))
              # read the fov mask
              fov_mask = misc.imread(path.join(current_mask_folder, mask_filenames[ii]))
              # get the preprocessed image
              preprocessed_I = preprocess(image, fov_mask, preprocessing_strategies[k])
              # write the image in the output folder
              misc.imsave(path.join(output_folder, image_filenames[ii][:-3] + 'png'), preprocessed_I)

    else:
          
      # get the folders for the images and the masks
      current_image_folder = path.join(root_path, current_dataset, 'images')
      current_mask_folder = path.join(root_path, current_dataset, 'masks')
      # get image filenames
      image_filenames = sorted(listdir(current_image_folder), key=natural_key)
      # get mask filenames
      mask_filenames = sorted(listdir(current_mask_folder), key=natural_key)

      # for each preprocessing strategy
      for k in range(0, len(preprocessing_strategies)):
        
        print('Preprocessing strategy {}'.format(preprocessing_strategies[k]))

        # prepare the output folder
        output_folder = current_image_folder + '_' + preprocessing_strategies[k]
        if not path.exists(output_folder):
          makedirs(output_folder)

        # and now, loop for each image
        for ii in range(0, len(image_filenames)):
          
          print('Image {} being processed with {}'.format(image_filenames[ii], preprocessing_strategies[k]))

          # read the image
          image = misc.imread(path.join(current_image_folder, image_filenames[ii]))
          # read the fov mask
          fov_mask = misc.imread(path.join(current_mask_folder, mask_filenames[ii]))
          # get the preprocessed image
          preprocessed_I = preprocess(image, fov_mask, preprocessing_strategies[k])
          # write the image in the output folder
          misc.imsave(path.join(output_folder, image_filenames[ii][:-3] + 'png'), preprocessed_I)



import sys
import argparse

if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="path to the data sets", type=str, default='../data')
    parser.add_argument("--database", help="database to preprocess", type=str, default='DRIVE, STARE, CHASEDB1, HRF')

    args = parser.parse_args()

    databases = args.database.replace(' ','').split(',')
    export_preprocessed_images(args.data_path, databases)