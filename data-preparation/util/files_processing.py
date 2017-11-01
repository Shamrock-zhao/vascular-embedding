
import re
import tarfile, sys
import gzip
import zipfile
from os import path, makedirs, listdir



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

def unzip_file(root_path, zip_filename, data_path):
    # Unzip file
    zip_ref = zipfile.ZipFile(path.join(root_path, zip_filename), 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()