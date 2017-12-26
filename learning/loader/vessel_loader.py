
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch

from glob import glob
from PIL import Image
from os import listdir, path
from torch.utils import data
from scipy import misc, ndimage



class PatchFromFundusImageLoader(data.Dataset):
    '''
    Use this data loader if you prefer to compute patches online.
    '''


    def __init__(self, data_folder, split, sampling_strategy='guided-by-labels', image_preprocessing='rgb', is_transform=False, num_patches=200000, patch_size=64, max_zoom=4):
        
        random.seed(7)

        # validate input parameters
        assert split in ['training', 'validation', 'test'], "Unknown split."
        assert sampling_strategy in ['uniform', 'guided-by-labels'], "Unsuported sampling strategy."
        assert image_preprocessing in ['rgb', 'eq', 'clahe'], "Unsuported image preprocessing."

        # configuration
        self.split = split     # type of split
        self.is_transform = is_transform    # if data must be augmented
        self.max_zoom = max_zoom
        self.sampling_strategy = sampling_strategy
        self.num_patches = num_patches
        self.pad = patch_size // 2

        # paths to data
        self.img_path = path.join(data_folder, split, 'images_' + image_preprocessing)
        self.labels_path = path.join(data_folder, split, 'labels')
        
        # collect image ids (names without extension) and shuffle
        self.image_ids = sorted(listdir(self.img_path))
        self.label_ids = sorted(listdir(self.labels_path))

        # read the first image and its annotations
        image = np.asarray(misc.imread(path.join(self.img_path, self.image_ids[0])), dtype=np.uint8)
        label = np.asarray(misc.imread(path.join(self.labels_path, self.label_ids[0])), dtype=np.int32) // 255
        # initialize arrays for the images and the labels
        self.images = np.empty((image.shape[0], image.shape[1], image.shape[2], len(self.image_ids)), dtype=np.uint8)
        self.labels = np.empty((label.shape[0], label.shape[1], len(self.label_ids)), dtype=np.int32)
        self.images[:,:,:,0] = image
        if len(label.shape)==3:
            self.labels[:,:,0] = label[:,:,0]
        else:
            self.labels[:,:,0] = label
        # open them so that we can randomly get a sample from them
        for i in range(1, len(self.image_ids)):
            self.images[:,:,:,i] = np.asarray(misc.imread(path.join(self.img_path, self.image_ids[i])), dtype=np.uint8)
            label = np.asarray(misc.imread(path.join(self.labels_path, self.label_ids[i])), dtype=np.int32) // 255
            if len(label.shape)==3:
                self.labels[:,:,i] = label[:,:,0]
            else:
                self.labels[:,:,i] = label


    def __len__(self):
        return self.num_patches


    def __getitem__(self, index):
        
        # pick a random image
        index_ = random.randint(0, self.images.shape[3]-1)
        current_img = self.images[:,:,:,index_]
        current_lbl = self.labels[:,:,index_]

        # get a random coordinate according to the sampling rule
        if self.sampling_strategy == 'uniform':
            
            i = random.randint(self.pad, current_img.shape[0] - self.pad - 1) 
            j = random.randint(self.pad, current_img.shape[1] - self.pad - 1)

        elif self.sampling_strategy == 'guided-by-labels':
            
            # pad the labels to avoid picking up an element outside the borders
            current_lbl_ = np.ones(current_lbl.shape, dtype=int) * -1
            current_lbl_[self.pad:current_lbl.shape[0]-self.pad, self.pad:current_lbl.shape[1]-self.pad] = current_lbl[self.pad:current_lbl.shape[0]-self.pad, self.pad:current_lbl.shape[1]-self.pad]

            if random.uniform(0, 1) < 0.5:
                # sample centered on background
                x,y = np.where(current_lbl_==0)
            else:
                # sample centered on vessels
                x,y = np.where(current_lbl_==1)

            i = x[random.randint(0, len(x)-1)]
            j = y[random.randint(0, len(y)-1)]
         
        # get the patch
        img = current_img[i-self.pad : i+self.pad, j-self.pad : j+self.pad, :]
        lbl = current_lbl[i-self.pad : i+self.pad, j-self.pad : j+self.pad]

        # use data augmentation if required
        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        # normalize data
        img = np.asarray(img, dtype=np.float32)
        img = (img - np.mean(img)) / (np.std(img) + 0.000001) # normalize by its own mean and standard deviation

        # set up variables for pytorch
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def transform(self, img, lbl):

        # choose a zoom level
        zoom_level = random.uniform(1, self.max_zoom)
        # resize the image
        zoomed_img = misc.imresize(img, zoom_level)
        zoomed_lbl = misc.imresize(lbl, zoom_level, interp='nearest') // 255

        # crop the parts outside
        i = zoomed_img.shape[0] // 2
        j = zoomed_img.shape[1] // 2
        zoomed_img = zoomed_img[ i-self.pad:i+self.pad, j-self.pad:j+self.pad, :]
        zoomed_lbl = zoomed_lbl[ i-self.pad:i+self.pad, j-self.pad:j+self.pad ]

        return zoomed_img, zoomed_lbl



class PatchesFromMultipleDatasets(data.Dataset):
    '''
    Use this data set to load patches from data sets with different resolutions
    '''


    def __init__(self, data_folder, datasets_names, split, sampling_strategy='guided-by-labels', image_preprocessing='rgb', is_transform=False, num_patches=200000, patch_size=64, max_zoom=4):
        
        random.seed(7)

        # validate input parameters
        assert split in ['training', 'validation', 'test'], "Unknown split."
        assert sampling_strategy in ['uniform', 'guided-by-labels'], "Unsuported sampling strategy."
        assert image_preprocessing in ['rgb', 'eq', 'clahe'], "Unsuported image preprocessing."

        # configuration
        self.split = split     # type of split
        self.is_transform = is_transform    # if data must be augmented
        self.sampling_strategy = sampling_strategy
        self.num_patches = num_patches
        self.pad = patch_size // 2
        self.max_zoom = max_zoom

        # collect filenames
        self.image_ids = []
        self.label_ids = []

        for i in range(0, len(datasets_names)):

            # prepare folder names
            img_paths = path.join(data_folder, datasets_names[i], split, 'images_' + image_preprocessing)
            labels_paths = path.join(data_folder, datasets_names[i], split, 'labels')
            # collect image ids (names without extension)
            current_images_ids = sorted(glob(path.join(img_paths, '*.png')))
            current_labels_ids = sorted(glob(path.join(labels_paths, '*.png')))
            if len(current_labels_ids) == 0:
                current_labels_ids = sorted(glob(path.join(labels_paths, '*.gif')))
            # concatenate
            self.image_ids = self.image_ids + current_images_ids
            self.label_ids = self.label_ids + current_labels_ids


    def __len__(self):
        return self.num_patches


    def __getitem__(self, index):
        
        # pick a random image
        index_ = random.randint(0, len(self.image_ids)-1)
        current_img = misc.imread(self.image_ids[index_])
        current_lbl = np.asarray(misc.imread(self.label_ids[index_]), dtype=np.int32) // 255
        # get a random coordinate according to the sampling rule
        if self.sampling_strategy == 'uniform':
            
            i = random.randint(self.pad, current_img.shape[0] - self.pad - 1) 
            j = random.randint(self.pad, current_img.shape[1] - self.pad - 1)

        elif self.sampling_strategy == 'guided-by-labels':
            
            # pad the labels to avoid picking up an element outside the borders
            current_lbl_ = np.ones(current_lbl.shape, dtype=int) * -1
            current_lbl_[self.pad:current_lbl.shape[0]-self.pad, self.pad:current_lbl.shape[1]-self.pad] = current_lbl[self.pad:current_lbl.shape[0]-self.pad, self.pad:current_lbl.shape[1]-self.pad]

            if random.uniform(0, 1) < 0.5:
                # sample centered on background
                x,y = np.where(current_lbl_==0)
            else:
                # sample centered on vessels
                x,y = np.where(current_lbl_==1)

            i = x[random.randint(0, len(x) - 1)]
            j = y[random.randint(0, len(y) - 1)]
         
        # get the patch
        img = current_img[i-self.pad : i+self.pad, j-self.pad : j+self.pad, :]
        lbl = current_lbl[i-self.pad : i+self.pad, j-self.pad : j+self.pad]

        # use data augmentation if required
        if self.is_transform:
            img, lbl = self.transform(img, lbl, self.max_zoom)

        # normalize data
        img = np.asarray(img, dtype=np.float32)
        img = (img - np.mean(img)) / (np.std(img) + 0.000001) # normalize by its own mean and standard deviation

        # set up variables for pytorch
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def transform(self, img, lbl, max_zoom):

        # choose a zoom level
        zoom_level = random.uniform(1, max_zoom)
        # resize the image
        zoomed_img = misc.imresize(img, zoom_level)
        zoomed_lbl = misc.imresize(lbl, zoom_level, interp='nearest') // 255

        # crop the parts outside
        i = zoomed_img.shape[0] // 2
        j = zoomed_img.shape[1] // 2
        zoomed_img = zoomed_img[ i-self.pad:i+self.pad, j-self.pad:j+self.pad, :]
        zoomed_lbl = zoomed_lbl[ i-self.pad:i+self.pad, j-self.pad:j+self.pad ]

        return zoomed_img, zoomed_lbl




class VesselPatchLoader(data.Dataset):
    '''
    Use this data loader is you prefer to work with precomputed patches.
    '''
    
    def __init__(self, data_folder, split, sampling_strategy='guided-by-labels', image_preprocessing='rgb', is_transform=False):
        
        random.seed(7)

        # validate input parameters
        assert split in ['training', 'validation', 'test'], "Unknown split."
        assert sampling_strategy in ['uniform', 'guided-by-labels'], "Unsuported sampling strategy."
        assert image_preprocessing in ['rgb', 'eq', 'clahe'], "Unsuported image preprocessing."

        # class attributes
        self.split = split     # type of split
        self.is_transform = is_transform    # if data must be augmented
        self.img_path = path.join(data_folder, split, 'patches_' + sampling_strategy + '_' + image_preprocessing)
        self.labels_path = path.join(data_folder, split, 'patches_' + sampling_strategy + '_labels')
        
        # collect image ids (names without extension) and shuffle
        self.image_ids = (f[:-4] for f in listdir(self.img_path))
        self.image_ids = list(self.image_ids)
        random.shuffle(self.image_ids)


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, index):
        img_name = self.image_ids[index]
        img_fullname = path.join(self.img_path, img_name + '.png')
        lbl_fullname = path.join(self.labels_path, img_name + '.gif')

        img = misc.imread(img_fullname)
        img = np.asarray(img, dtype=np.uint8)
        
        lbl = misc.imread(lbl_fullname)
        lbl = np.array(lbl, dtype=np.int32)[:,:,0] // 255

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        img = np.asarray(img, dtype=np.float32)
        img = (img - np.mean(img)) / (np.std(img) + 0.000001) # normalize by its own mean and standard deviation

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def transform(self, img, lbl):

        pad = img // 2
        # choose a zoom level
        zoom_level = random.uniform(1, 3)
        # resize the image
        #zoomed_img = misc.imresize(img[:,:,0], zoom_level)
        zoomed_img = ndimage.zoom(img, (zoom_level, zoom_level, 1))
        zoomed_lbl = ndimage.zoom(lbl, (zoom_level, zoom_level), mode='nearest') // 255

        # crop the parts outside
        i = zoomed_img.shape[0] // 2
        j = zoomed_img.shape[1] // 2
        zoomed_img = zoomed_img[ i-pad:i+pad, j-pad:j+pad, :]
        zoomed_lbl = zoomed_lbl[ i-pad:i+pad, j-pad:j+pad ]

        return zoomed_img, zoomed_lbl
