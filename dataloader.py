# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 2021

@author: Tom
"""
from os import listdir
from os.path import isfile, join
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageNetDataset(Dataset):
    """Custom dataset for loading ImageNet Dataset.
    Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    Uses images in img_dir as testing dataset (X).
    Uses labels in val.txt as target labels (Y).
    Class ID's are in 'sysnet.txt'.
    Class descriptions are in 'sysnet_words.txt'.
    """

    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        #List of images files in img_dir
        onlyfiles = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
        onlyfiles.sort()
        self.image_names = onlyfiles
        self.labels = pd.read_csv('val.txt', delimiter=' ', names=['target']).to_numpy()
        self.img_dir = img_dir
        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path = join(self.img_dir, self.image_names[index])
        img = Image.open(image_path).convert('RGB')
        target = self.labels[index]

        if self.transform is not None:
            img_normalized = self.transform(img)

        return {"image":img_normalized, "target":target, "raw":np.asarray(img)}
        
        
    
    