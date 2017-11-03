
import re
import tarfile, sys
import gzip
import zipfile
from os import path, makedirs, listdir
from scipy import misc
from shutil import move, copyfile



def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]



def untar_file(root_path, tar_filename, data_path):
    # Untar file
    tar = tarfile.open(path.join(root_path, tar_filename))
    tar.extractall(data_path)
    tar.close()



def ungz_file(root_path, current_filename, data_path):
    # Ungzip file
    input_file = gzip.open(path.join(root_path, current_filename), 'rb')
    output_file = open(path.join(data_path, current_filename[:-3]), 'wb')
    output_file.write( input_file.read() )
    input_file.close()
    output_file.close()



def ungz_files(root_path, gz_filenames, data_path):
    ''' 
    Uncompressed all the gz files in a folder
    '''
    # Create folders
    if not path.exists(data_path):
        makedirs(data_path)
    # Ungz files    
    for i in range(0, len(gz_filenames)):
        ungz_file(root_path, gz_filenames[i], data_path)    



def unzip_file(root_path, zip_filename, data_path):
    # Unzip file
    zip_ref = zipfile.ZipFile(path.join(root_path, zip_filename), 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()



def unzip_files(root_path, zip_filenames, data_path):
    # Unzip files    
    for i in range(0, len(zip_filenames)):
        unzip_file(root_path, zip_filenames[i], data_path)



def copy_images(root_folder, filenames, data_path):
        # Create the folder if it doesnt exist
    if not path.exists(data_path):
        makedirs(data_path)
    # Copy images in filenames to data_path
    for i in range(0, len(filenames)):
        current_file = path.basename(filenames[i])
        if current_file[-3:]=='JPG':
            target_filename = current_file[:-3] + 'jpg'
        else:
            target_filename = current_file
        copyfile(path.join(root_folder, current_file), path.join(data_path, target_filename))            



def copy_labels(root_folder, filenames, data_path):
    # Create folders
    if not path.exists(data_path):
        makedirs(data_path)
    # Copy the images
    for i in range(0, len(filenames)):
        current_filename = path.basename(filenames[i])
        # Open the image
        labels = (misc.imread(path.join(root_folder, current_filename)) / 255).astype('int32')
        # Save the image as a .png file
        misc.imsave(path.join(data_path, current_filename[:-3] + 'png'), labels)



def move_files(filenames, source_folder, destination_folder):
    # Move files from source to destination folder
    for i in range(0, len(filenames)):
        move(path.join(source_folder, filenames[i]), path.join(destination_folder, filenames[i]))    