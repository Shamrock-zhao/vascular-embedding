
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from os import listdir, path
from torch.utils import data
from scipy import misc, ndimage



class PatchFromFundusImageLoader(data.Dataset):
    
    def __init__(self, data_folder, split, sampling_strategy='guided-by-labels', image_preprocessing='rgb', is_transform=False, num_patches=200000, patch_size=64):
        
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
        self.labels[:,:,0] = label
        # open them so that we can randomly get a sample from them
        for i in range(1, len(self.image_ids)):
            self.images[:,:,:,i] = np.asarray(misc.imread(path.join(self.img_path, self.image_ids[i])), dtype=np.uint8)
            self.labels[:,:,i] = np.asarray(misc.imread(path.join(self.labels_path, self.label_ids[i])), dtype=np.int32) // 255


    def __len__(self):
        return self.num_patches


    def __getitem__(self, index):
        
        index_ = random.randint(0, self.images.shape[3]-1)
        current_img = self.images[:,:,:,index_]
        current_lbl = self.labels[:,:,index_]

        i = random.randint(self.pad, current_img.shape[0] - self.pad) 
        j = random.randint(self.pad, current_img.shape[1] - self.pad)

        img = current_img[i-self.pad : i+self.pad, j-self.pad : j+self.pad, :]
        lbl = current_lbl[i-self.pad : i+self.pad, j-self.pad : j+self.pad]

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        img = np.asarray(img, dtype=np.float32)
        img = (img - np.mean(img)) / (np.std(img) + 0.000001) # normalize by its own mean and standard deviation

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def transform(self, img, lbl):

        zoom_level = np.random.uniform(1, 3)
        zoomed_img = misc.imresize(img[:,:,0], zoom_level)
        for i in range(1, img.shape[2]):
            zoomed_img = np.dstack((zoomed_img, misc.imresize(img[:,:,i], zoom_level)))
        zoomed_lbl = misc.imresize(lbl, zoom_level, interp='nearest') // 255

        if (zoomed_img.shape[0] > img.shape[0]) or (zoomed_img.shape[1] > img.shape[1]):
            
            first_x = random.randint(0, zoomed_img.shape[0] - img.shape[0])
            first_y = random.randint(0, zoomed_img.shape[1] - img.shape[1])
            zoomed_img = zoomed_img[first_x : img.shape[0] + first_x, first_y : img.shape[1] + first_y, :]
            zoomed_lbl = zoomed_lbl[first_x : lbl.shape[0] + first_x, first_y : lbl.shape[1] + first_y]

        return zoomed_img, zoomed_lbl



class VesselPatchLoader(data.Dataset):
    
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

        zoom_level = np.random.uniform(1, 3)
        zoomed_img = misc.imresize(img[:,:,0], zoom_level)
        for i in range(1, img.shape[2]):
            zoomed_img = np.dstack((zoomed_img, misc.imresize(img[:,:,i], zoom_level)))
        zoomed_lbl = misc.imresize(lbl, zoom_level, interp='nearest') // 255

        if (zoomed_img.shape[0] > img.shape[0]) or (zoomed_img.shape[1] > img.shape[1]):
            
            first_x = random.randint(0, zoomed_img.shape[0] - img.shape[0])
            first_y = random.randint(0, zoomed_img.shape[1] - img.shape[1])
            zoomed_img = zoomed_img[first_x : img.shape[0] + first_x, first_y : img.shape[1] + first_y, :]
            zoomed_lbl = zoomed_lbl[first_x : lbl.shape[0] + first_x, first_y : lbl.shape[1] + first_y]

        return zoomed_img, zoomed_lbl
