

import urllib.request
from os import path, makedirs, listdir, rename
from shutil import rmtree, move
from glob import glob
from .files_processing import natural_key, unzip_file, copy_images, copy_labels, move_files, unzip_files, untar_file, ungz_files
from .image_processing import generate_fov_masks


def organize_chasedb(data_path='../data'):
    
    # URL to download images
    URL = 'https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip'

    # Training data paths
    TRAINING_IMAGES_DATA_PATH = path.join(data_path, 'CHASEDB1/training/images')
    TRAINING_GT_DATA_PATH = path.join(data_path, 'CHASEDB1/training/labels')
    TRAINING_GT2_DATA_PATH = path.join(data_path, 'CHASEDB1/training/labels2')
    TRAINING_FOV_MASKS_DATA_PATH = path.join(data_path, 'CHASEDB1/training/masks')
    # Validation data paths		
    VALIDATION_IMAGES_DATA_PATH = path.join(data_path, 'CHASEDB1/validation/images')
    VALIDATION_GT_DATA_PATH = path.join(data_path, 'CHASEDB1/validation/labels')
    VALIDATION_GT2_DATA_PATH = path.join(data_path, 'CHASEDB1/validation/labels2')		
    VALIDATION_FOV_MASKS_DATA_PATH = path.join(data_path, 'CHASEDB1/validation/masks')
    # Test data paths
    TEST_IMAGES_DATA_PATH = path.join(data_path, 'CHASEDB1/test/images/')
    TEST_GT_DATA_PATH = path.join(data_path, 'CHASEDB1/test/labels/')
    TEST_GT2_DATA_PATH = path.join(data_path, 'CHASEDB1/test/labels2/')
    TEST_FOV_MASKS_DATA_PATH = path.join(data_path, 'CHASEDB1/test/masks')

    # Check if tmp exists
    if not path.exists('../tmp'):
        makedirs('../tmp')

    # Download images from the known link to tmp
    data_path = path.join('../tmp', 'CHASEDB1.zip')
    if not path.exists(data_path):
        print('Downloading data from ' + URL)
        urllib.request.urlretrieve(URL, data_path)
    else:
        print('CHASEDB1.zip file already exists. Skipping download.')

    # Check if CHASEDB1 folders exist
    if not path.exists('../tmp/CHASEDB1/images'):
        makedirs('../tmp/CHASEDB1/images')
    if not path.exists('../tmp/CHASEDB1/labels'):
        makedirs('../tmp/CHASEDB1/labels')
    if not path.exists('../tmp/CHASEDB1/labels2'):
        makedirs('../tmp/CHASEDB1/labels2') 
    if not path.exists('../tmp/CHASEDB1/masks'):
        makedirs('../tmp/CHASEDB1/masks')                

    # Unzip files in tmp/CHASEDB1
    print('Unzipping images...')
    unzip_file('../tmp', 'CHASEDB1.zip', '../tmp/CHASEDB1')

    # Generate FOV masks
    print('Generating FOV masks...')
    # Get image filenames
    image_filenames = sorted(glob('../tmp/CHASEDB1/*.jpg'), key=natural_key)
    generate_fov_masks('../tmp/CHASEDB1', image_filenames)

    # Move images
    # 1. Get image filenames
    image_filenames = sorted(glob('../tmp/CHASEDB1/*.jpg'), key=natural_key)
    # 2. Copy the first 20 images as test set
    copy_images('../tmp/CHASEDB1', image_filenames[:20], TEST_IMAGES_DATA_PATH)
    # 3. Copy the last 2 images as validation set
    copy_images('../tmp/CHASEDB1', image_filenames[-2:], VALIDATION_IMAGES_DATA_PATH)
    # 4. Copy the images from 20 to 26 as training set
    copy_images('../tmp/CHASEDB1', image_filenames[20:26], TRAINING_IMAGES_DATA_PATH)

    # Move first observer labels
    # 1. Get labels filenames (first observer)
    labels_filenames = sorted(glob('../tmp/CHASEDB1/*1stHO.png'), key=natural_key)
    # 2. Copy the first 20 images as test set
    copy_labels('../tmp/CHASEDB1', labels_filenames[:20], TEST_GT_DATA_PATH)
    # 3. Copy the last 2 images as validation set
    copy_labels('../tmp/CHASEDB1', labels_filenames[-2:], VALIDATION_GT_DATA_PATH)
    # 4. Copy the images from 20 to 26 as training set
    copy_labels('../tmp/CHASEDB1', labels_filenames[20:26], TRAINING_GT_DATA_PATH)

    # Move second observer labels
    # 1. Get labels filenames (first observer)
    labels2_filenames = sorted(glob('../tmp/CHASEDB1/*2ndHO.png'), key=natural_key)
    # 2. Copy the first 20 images as test set
    copy_labels('../tmp/CHASEDB1', labels2_filenames[:20], TEST_GT2_DATA_PATH)
    # 3. Copy the last 2 images as validation set
    copy_labels('../tmp/CHASEDB1', labels2_filenames[-2:], VALIDATION_GT2_DATA_PATH)
    # 4. Copy the images from 20 to 26 as training set
    copy_labels('../tmp/CHASEDB1', labels2_filenames[20:26], TRAINING_GT2_DATA_PATH)

    # Move FOV masks
    # 1. Get labels filenames (first observer)
    fov_filenames = sorted(glob('../tmp/CHASEDB1/*_fov_mask.png'), key=natural_key)
    # 2. Copy the first 20 images as test set
    copy_labels('../tmp/CHASEDB1', fov_filenames[:20], TEST_FOV_MASKS_DATA_PATH)
    # 3. Copy the last 2 images as validation set
    copy_labels('../tmp/CHASEDB1', fov_filenames[-2:], VALIDATION_FOV_MASKS_DATA_PATH)
    # 4. Copy the images from 20 to 26 as training set
    copy_labels('../tmp/CHASEDB1', fov_filenames[20:26], TRAINING_FOV_MASKS_DATA_PATH)

    # Remove useless folders
    rmtree('../tmp/CHASEDB1/')
    print('CHASEDB1 data set ready!')



def organize_drive(data_path='../data'):

    # Check if tmp exists
    if not path.exists('../tmp'):
        makedirs('../tmp')

    # Download images from the known link to tmp
    zip_file = path.join('../tmp', 'DRIVE.zip')
    if not path.exists(zip_file):
        raise ValueError('Download the DRIVE database from https://www.isi.uu.nl/Research/Databases/DRIVE/index.php and save the file in ../tmp.')
    else:
        print('DRIVE.zip file exists. Continuing processing...')

    # Unzip files
    unzip_file('../tmp', 'DRIVE.zip', '../tmp')

    # Rename folders
    # Training set
    rename('../tmp/DRIVE/training/1st_manual', '../tmp/DRIVE/training/labels')
    rename('../tmp/DRIVE/training/mask', '../tmp/DRIVE/training/masks')
    # Test set
    rename('../tmp/DRIVE/test/1st_manual', '../tmp/DRIVE/test/labels')
    rename('../tmp/DRIVE/test/2nd_manual', '../tmp/DRIVE/test/labels2')
    rename('../tmp/DRIVE/test/mask', '../tmp/DRIVE/test/masks')

    # Move images to the validation set
    makedirs('../tmp/DRIVE/validation/images')
    image_filenames = sorted(listdir('../tmp/DRIVE/training/images'), key=natural_key)
    move_files(image_filenames[-5:], '../tmp/DRIVE/training/images', '../tmp/DRIVE/validation/images')

    # Move labels to the validation set
    makedirs('../tmp/DRIVE/validation/labels')
    image_filenames = sorted(listdir('../tmp/DRIVE/training/labels'), key=natural_key)
    move_files(image_filenames[-5:], '../tmp/DRIVE/training/labels', '../tmp/DRIVE/validation/labels')

    # Move masks to the validation set
    makedirs('../tmp/DRIVE/validation/masks')
    image_filenames = sorted(listdir('../tmp/DRIVE/training/masks'), key=natural_key)
    move_files(image_filenames[-5:], '../tmp/DRIVE/training/masks', '../tmp/DRIVE/validation/masks')

    # Done! Now move the folder
    move('../tmp/DRIVE', path.join(data_path, 'DRIVE'))    
    print('DRIVE data set ready!')    



def organize_hrf(data_path='../data'):

    # URLs to download images
    URLs = ['https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip', 
            'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma.zip',
            'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy.zip',
            'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip',
            'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma_manualsegm.zip',
            'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy_manualsegm.zip']

    # URL filenames
    URL_FILENAMES_IMAGES = ['healthy.zip', 'glaucoma.zip', 'diabetic_retinopathy.zip']
    URL_FILENAMES_LABELS = ['healthy_manualsegm.zip', 'glaucoma_manualsegm.zip', 'diabetic_retinopathy_manualsegm.zip']
    ALL_URL_FILENAMES = URL_FILENAMES_IMAGES + URL_FILENAMES_LABELS

    # Training data paths
    TRAINING_IMAGES_DATA_PATH = path.join(data_path, 'HRF/training/images')
    TRAINING_GT_DATA_PATH = path.join(data_path, 'HRF/training/labels')
    TRAINING_FOV_MASKS_DATA_PATH = path.join(data_path, 'HRF/training/masks')
    # Validation data paths		
    VALIDATION_IMAGES_DATA_PATH = path.join(data_path, 'HRF/validation/images')
    VALIDATION_GT_DATA_PATH = path.join(data_path, 'HRF/validation/labels')		
    VALIDATION_FOV_MASKS_DATA_PATH = path.join(data_path, 'HRF/validation/masks')
    # Test data paths
    TEST_IMAGES_DATA_PATH = path.join(data_path, 'HRF/test/images/')
    TEST_GT_DATA_PATH = path.join(data_path, 'HRF/test/labels/')
    TEST_FOV_MASKS_DATA_PATH = path.join(data_path, 'HRF/test/masks')

    # Check if tmp exists
    if not path.exists('../tmp'):
        makedirs('../tmp')

    # Download images from the known links to tmp
    for i in range(0, len(URLs)):
        data_path = path.join('../tmp', ALL_URL_FILENAMES[i])
        if not path.exists(data_path):
            print('Downloading data from ' + URLs[i])
            urllib.request.urlretrieve(URLs[i], data_path)
        else:
            print(ALL_URL_FILENAMES[i] + ' already exists. Skipping download.')

    # Check if HRF folders exist
    if not path.exists('../tmp/HRF/images'):
        makedirs('../tmp/HRF/images')
    if not path.exists('../tmp/HRF/labels'):
        makedirs('../tmp/HRF/labels')
    if not path.exists('../tmp/HRF/masks'):
        makedirs('../tmp/HRF/masks')        

    # Unzip images in tmp/images
    print('Unzipping images...')
    unzip_files('../tmp', URL_FILENAMES_IMAGES, '../tmp/HRF/images')
    # Unzip images in tmp/gt
    print('Unzipping labels...')
    unzip_files('../tmp', URL_FILENAMES_LABELS, '../tmp/HRF/labels')

    # Get image names
    print('Generating FOV masks...')
    image_filenames = sorted(listdir('../tmp/HRF/images'), key=natural_key)
    # Copy them to the masks folder
    copy_images('../tmp/HRF/images', image_filenames, '../tmp/HRF/masks')
    image_filenames = sorted(listdir('../tmp/HRF/masks'), key=natural_key)
    # Generate FOV masks
    generate_fov_masks('../tmp/HRF/masks', image_filenames)

    # Copy training/validation/test images
    print('Copying images...')
    # 1. Get image names
    image_filenames = sorted(listdir('../tmp/HRF/images'), key=natural_key)
    # 2. Copy training images
    copy_images('../tmp/HRF/images', image_filenames[:12], TRAINING_IMAGES_DATA_PATH)
    # 3. Copy validation images
    copy_images('../tmp/HRF/images', image_filenames[12:15], VALIDATION_IMAGES_DATA_PATH)
    # 4. Copy test images
    copy_images('../tmp/HRF/images', image_filenames[-30:], TEST_IMAGES_DATA_PATH)

    # Copy training/validation/test labels 
    print('Copying labels...')
    # 1. Get labels names
    gt_filenames = sorted(listdir('../tmp/HRF/labels'), key=natural_key)
    # 2. Copy training labels
    copy_labels('../tmp/HRF/labels', gt_filenames[:12], TRAINING_GT_DATA_PATH)
    # 3. Copy validation labels
    copy_labels('../tmp/HRF/labels', gt_filenames[12:15], VALIDATION_GT_DATA_PATH)
    # 4. Copy test labels
    copy_labels('../tmp/HRF/labels', gt_filenames[-30:], TEST_GT_DATA_PATH)

    # Copy training/validation/test FOV masks 
    print('Copying FOV masks...')
    # 1. Get masks names
    fov_filenames = sorted(glob('../tmp/HRF/masks/*_fov_mask.png'), key=natural_key)
    # 2. Copy training labels
    copy_labels('../tmp/HRF/masks', fov_filenames[:12], TRAINING_FOV_MASKS_DATA_PATH)
    # 3. Copy validation labels
    copy_labels('../tmp/HRF/masks', fov_filenames[12:15], VALIDATION_FOV_MASKS_DATA_PATH)
    # 4. Copy test labels
    copy_labels('../tmp/HRF/masks', fov_filenames[-30:], TEST_FOV_MASKS_DATA_PATH)


    # Remove useless folders
    rmtree('../tmp/HRF/')
    print('HRF data set ready!')



def organize_iostar(data_path='../data'):
    
    URL = 'http://www.retinacheck.org/datasets#jtabs-3'

    # Check if tmp exists
    if not path.exists('../tmp'):
        makedirs('../tmp')

    # Download images from the known link to tmp
    zip_file = path.join('../tmp', 'IOSTAR-Vessel-Segmentation-Dataset.zip')
    if not path.exists(zip_file):
        raise ValueError('Download the IOSTAR database from {} and save the file in ../tmp.'.format(URL))
    else:
        print('IOSTAR-Vessel-Segmentation-Dataset.zip file exists. Continuing processing...')

    # Unzip files
    unzip_file('../tmp', 'IOSTAR-Vessel-Segmentation-Dataset.zip', '../tmp')

    # We will only have a single folder, as now proposals for training/test are provided in this data set
    rename('../tmp/IOSTAR Vessel Segmentation Dataset', '../tmp/IOSTAR')
    rename('../tmp/IOSTAR/image', '../tmp/IOSTAR/images')
    rename('../tmp/IOSTAR/mask', '../tmp/IOSTAR/masks')
    rename('../tmp/IOSTAR/GT', '../tmp/IOSTAR/labels')

    # Done! Now move the folder
    move('../tmp/IOSTAR', path.join(data_path, 'IOSTAR'))
    print('IOSTAR data set ready!')    



def organize_drhagis(data_path='../data'):
    
    URL = 'http://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/DRHAGIS.zip'

    # Check if tmp exists
    if not path.exists('../tmp'):
        makedirs('../tmp')

    # Download images from the known link to tmp
    zip_file = path.join('../tmp', 'DRHAGIS.zip')
    if not path.exists(zip_file):
        print('Downloading data from ' + URL)
        urllib.request.urlretrieve(URL, zip_file)
    else:
        print('DRHAGIS.zip file exists. Continuing processing...')

    # Unzip files
    unzip_file('../tmp', 'DRHAGIS.zip', '../tmp')

    # We will only have a single folder, as now proposals for training/test are provided in this data set
    rename('../tmp/DRHAGIS/Fundus_Images', '../tmp/DRHAGIS/images')
    rename('../tmp/DRHAGIS/Mask_images', '../tmp/DRHAGIS/masks')
    rename('../tmp/DRHAGIS/Manual_Segmentations', '../tmp/DRHAGIS/labels')

    # Done! Now move the folder
    move('../tmp/DRHAGIS', path.join(data_path, 'DRHAGIS'))
    print('DRHAGIS data set ready!') 



def organize_stare(data_path='../data'):
    
    # URLs to download images
    URLs = ['http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar', 
            'http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar',
            'http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar']
    URLS_FILENAMES = ['stare-images.tar', 'labels-ah.tar', 'labels-vk.tar']

    # Training data paths
    TRAINING_IMAGES_DATA_PATH = path.join(data_path, 'STARE/training/images')
    TRAINING_GT_DATA_PATH = path.join(data_path, 'STARE/training/labels')
    TRAINING_GT2_DATA_PATH = path.join(data_path, 'STARE/training/labels2')
    TRAINING_FOV_MASKS_DATA_PATH = path.join(data_path, 'STARE/training/masks')
    # Validation data paths		
    VALIDATION_IMAGES_DATA_PATH = path.join(data_path, 'STARE/validation/images')
    VALIDATION_GT_DATA_PATH = path.join(data_path, 'STARE/validation/labels')	
    VALIDATION_GT2_DATA_PATH = path.join(data_path, 'STARE/validation/labels2')
    VALIDATION_FOV_MASKS_DATA_PATH = path.join(data_path, 'STARE/validation/masks')
    # Test data paths
    TEST_IMAGES_DATA_PATH = path.join(data_path, 'STARE/test/images/')
    TEST_GT_DATA_PATH = path.join(data_path, 'STARE/test/labels/')
    TEST_GT2_DATA_PATH = path.join(data_path, 'STARE/test/labels2/')
    TEST_FOV_MASKS_DATA_PATH = path.join(data_path, 'STARE/test/masks')

    # Check if tmp exists
    if not path.exists('../tmp'):
        makedirs('../tmp')

    # Download images from the known links to tmp
    for i in range(0, len(URLs)):
        data_path = path.join('../tmp', URLS_FILENAMES[i])
        if not path.exists(data_path):
            print('Downloading data from ' + URLs[i])
            urllib.request.urlretrieve(URLs[i], data_path)
        else:
            print(URLS_FILENAMES[i] + ' already exists. Skipping download.')

    # Check if STARE folders exist
    if not path.exists('../tmp/STARE/images'):
        makedirs('../tmp/STARE/images')
    if not path.exists('../tmp/STARE/labels'):
        makedirs('../tmp/STARE/labels')
    if not path.exists('../tmp/STARE/labels2'):
        makedirs('../tmp/STARE/labels2')           

    # Untar images in tmp/images
    print('Extracting images...')
    untar_file('../tmp', URLS_FILENAMES[0], '../tmp/STARE/images')
    # Untar images in tmp/gt
    print('Extracting labels...')
    untar_file('../tmp', URLS_FILENAMES[1], '../tmp/STARE/labels')
    # Untar images in tmp/gt
    print('Extracting the other labels...')
    untar_file('../tmp', URLS_FILENAMES[2], '../tmp/STARE/labels2')    

    # Copy training/validation/test images
    print('Copying images...')
    # 1. Get image names
    image_filenames = sorted(listdir('../tmp/STARE/images'), key=natural_key)
    # 2. Copy training images
    ungz_files('../tmp/STARE/images', image_filenames[:7], TRAINING_IMAGES_DATA_PATH)
    # 3. Copy validation images
    ungz_files('../tmp/STARE/images', image_filenames[7:10], VALIDATION_IMAGES_DATA_PATH)
    # 4. Copy test images
    ungz_files('../tmp/STARE/images', image_filenames[-10:], TEST_IMAGES_DATA_PATH)

    # Copy training/validation/test labels
    print('Copying labels...')
    # 1. Get image names
    gt_filenames = sorted(listdir('../tmp/STARE/labels'), key=natural_key)
    # 2. Copy training labels
    ungz_files('../tmp/STARE/labels', gt_filenames[:7], TRAINING_GT_DATA_PATH)
    # 3. Copy validation labels
    ungz_files('../tmp/STARE/labels', gt_filenames[7:10], VALIDATION_GT_DATA_PATH)
    # 4. Copy test labels
    ungz_files('../tmp/STARE/labels', gt_filenames[-10:], TEST_GT_DATA_PATH)

    # Copy training/validation/test labels2
    print('Copying the other labels...')
    # 1. Get image names
    gt_filenames = sorted(listdir('../tmp/STARE/labels2'), key=natural_key)
    # 2. Copy training labels2
    ungz_files('../tmp/STARE/labels2', gt_filenames[:7], TRAINING_GT2_DATA_PATH)
    # 3. Copy validation labels2
    ungz_files('../tmp/STARE/labels2', gt_filenames[7:10], VALIDATION_GT2_DATA_PATH)
    # 4. Copy test labels2
    ungz_files('../tmp/STARE/labels2', gt_filenames[-10:], TEST_GT2_DATA_PATH)
    
    # Copy the precomputed masks
    print('Moving precomputed masks...')
    fov_filenames = sorted(listdir('precomputed_data/STARE/masks'), key=natural_key)
    copy_images('precomputed_data/STARE/masks', fov_filenames[:7], TRAINING_FOV_MASKS_DATA_PATH)
    copy_images('precomputed_data/STARE/masks', fov_filenames[7:10], VALIDATION_FOV_MASKS_DATA_PATH)
    copy_images('precomputed_data/STARE/masks', fov_filenames[-10:], TEST_FOV_MASKS_DATA_PATH)

    # Remove useless folders
    rmtree('../tmp/STARE/')
    print('STARE data set ready!')    