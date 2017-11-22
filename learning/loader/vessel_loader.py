
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch

from os import listdir, path
from torch.utils import data
from scipy import misc



class VesselPatchLoader(data.Dataset):
    
    def __init__(self, img_path, labels_path, split, validation_ratio=0.05, is_transform=False):
        
        random.seed(7)

        # if data must be augmented
        self.is_transform = is_transform
        # as we will use this loader for segmentation, n_classes = 2
        self.n_classes = 2
        # assign split
        self.split = split

        # paths where the data are saved
        self.img_path = img_path
        self.labels_path = labels_path
        
        # collect image ids (names without extension) and shuffle
        self.image_ids = (f[:-4] for f in listdir(self.img_path))
        self.image_ids = list(self.image_ids)
        random.shuffle(self.image_ids)

        # training and validation ratios and number of training samples
        self.training_ratio = 1 - validation_ratio
        self.validation_ratio = validation_ratio
        self.n_training_samples = round(len(self.image_ids) * self.training_ratio)

        # split data into training and validation
        self.files = collections.defaultdict(list)
        self.files['training'] = self.image_ids[:self.n_training_samples]
        self.files['validation'] = self.image_ids[self.n_training_samples:]


    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_fullname = path.join(self.img_path, img_name + '.png')
        lbl_fullname = path.join(self.labels_path, img_name + '.gif')

        img = misc.imread(img_fullname)
        img = np.array(img, dtype=np.uint8)

        lbl = misc.imread(lbl_fullname)
        lbl = np.array(lbl, dtype=np.int32)[:,:,0] // 255

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def transform(self, img, lbl):
        # TODO: Implement data augmentation

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl




'''
if __name__ == '__main__':
    local_path = 'data/DRIVE/training/'
    dst = camvidLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.imshow(dst.decode_segmap(labels.numpy()[i]))
            plt.show()

'''