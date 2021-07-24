import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    """
        Args: A_directory: directory of the first imgs 
              B_directory: directory of the second imgs
              transform  : any transformations to be applied to the imgs
    """
    def __init__(self, A_directory, B_directory, transform=None):

        # get the data paths

        self.A_dir = A_directory
        self.B_dir = B_directory
        self.transform = transform

        self.A_imgs = os.listdir(A_directory)
        self.B_imgs = os.listdir(B_directory)

        # get the data lengths
        self.ds_length = max(len(self.A_imgs), len(self.B_imgs)) 
        self.A_imgs_length = len(self.A_imgs)
        self.B_imgs_length = len(self.B_imgs)

    def __len__(self):
        return self.ds_length

    def __getitem__(self, index):
        # get the imgs which their turn
        A_img = self.A_imgs[index % self.A_imgs_length]
        B_img = self.B_imgs[index % self.B_imgs_length]

        # read the imgs
        A_path = os.path.join(self.B_dir, A_img)
        B_path = os.path.join(self.B_dir, B_img)

        # convert imgs to rgb as the model's input img must have 3 channels
        A_imgs = np.array(Image.open(A_path).convert("RGB"))
        B_imgs = np.array(Image.open(B_path).convert("RGB"))

        # apply transformations on the imgs
        if self.transform:
            augmentations = self.transform(image=A_imgs, image0=B_imgs)
            A_imgs = augmentations["image"]
            B_imgs = augmentations["image0"]

        # return the imgs
        return A_imgs, B_imgs