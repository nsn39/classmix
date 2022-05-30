import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from data.city_utils import recursive_glob
from data.augmentations import *

class NaturalDatasetLoader(data.Dataset):

    def __init__(self, root):
        self.root = root 
        self.files = list()
        self.nclasses = 19
        self.images_base = self.root

        self.files = recursive_glob(rootdir=self.images_base, suffix=".jpg")

        if len(self.files) == 0:
            raise Exception(
                "No files found."
            )

    def __len__(self):
        return len(self.files) 

    def __getitem__(self, index):
        img_path = self.files[index].rstrip()
       
        img = m.imread(img_path)
        print("Shape: ", img.shape)
        img = np.array(img, dtype=np.uint8)


        img_name = img_path.split('/')[-1]
        return img, img_path, img_name