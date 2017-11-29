
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from os import listdir, path
from torch.utils import data
from scipy import misc, ndimage




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
        img = (img - np.mean(img)) / np.std(img) # normalize by its own mean and standard deviation

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def transform(self, img, lbl):
        # TODO: Implement data augmentation

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




class FundusImageLoader(data.Dataset):
    
    def __init__(self, data_folder, split=None, image_preprocessing='rgb'):
        
        random.seed(7)

        # validate input parameters
        assert split in [None, 'test'], "Unknown split."
        assert image_preprocessing in ['rgb', 'eq', 'clahe'], "Unsuported image preprocessing."

        # class attributes
        self.split = split     # type of split
        self.image_preprocessing = image_preprocessing # type of preprocessing
        
        if split is None:
            self.img_path = path.join(data_folder, 'images')
            self.fov_mask_path = path.join(data_folder, 'masks')
        else:
            self.img_path = path.join(data_folder, split, 'images')
            self.fov_mask_path = path.join(data_folder, split, 'masks')
        
        # collect image ids
        self.image_ids = listdir(self.img_path)
        self.image_ids = list(self.image_ids)
        # collect fov masks ids
        self.masks_ids = listdir(self.fov_mask_path)
        self.masks_ids = list(self.masks_ids)


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, index):

        img_fullname = path.join(self.img_path, self.image_ids[index])
        mask_fullname = path.join(self.fov_mask_path, self.masks_ids[index])

        img = misc.imread(img_fullname)
        img = np.asarray(img, dtype=np.uint8)

        fov_mask = misc.imread(mask_fullname)
        fov_mask = np.array(fov_mask, dtype=np.int32)[:,:,0] // 255

        img = preprocess(img, fov_mask, self.image_preprocessing)

        img = torch.from_numpy(img).float()

        return img
