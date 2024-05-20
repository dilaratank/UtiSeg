"""
datasets.py

Makes a TVUS Uterus Segmentation dataset, including pre-processing and augmentation for the purpose of uterus segmentation.
"""

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import cv2

class TVUSUterusSegmentationDataset(Dataset):
    """
    Represents a TVUS dataset for the purpose of deep learning-based uterus segmentation

    Arguments
    ---------
    data_folder: the location of the folder where the original data is stored
    mask_folder: the location of the folder where the data masks are stored
    data_type: the type of the data, can be still, video, 3D, or all
    resize: image resolution to resize the images to, for example 128 would be an image resolution of 128x128
    clahe: if clahe is applied to the images or not
    padding: if padding is applied to the images or not
    transform: what kind of transformations are applied to the images

    """
    def __init__(self, data_folder, mask_folder, data_type, resize=128, clahe=False, padding=False, transform=None):
        """ Initializes a new dataset instance """
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.data_type = data_type
        self.transf = transform 
        self.clahe = clahe
        self.padding = padding

        self.image_list = self.get_imgs_list(self.data_folder)
        self.mask_list = self.get_imgs_list(self.mask_folder)

        self.resize = resize
        
    def get_imgs_list(self, root_folder):
        """
        A function that creates an ordered list of image location paths based on imaging type, to be used for dataset creation 
        
        Parameters
        ---------
        root_folder: the location of the folder where images  (either original data or masks) are stored

        Returns
        -------
        image_paths: an ordered list of the location of the images to be used for the dataset 
        """
        image_paths = []

        if self.data_type == "ALL":
            for file_name in os.listdir(root_folder):
                image_path = os.path.join(root_folder, file_name)
                image_paths.append(image_path)
        
        # all other imaging types have different folder structures
        else:
            # Walk through the directory structure
            for patient_number_folder in os.listdir(root_folder):
                patient_folder_path = os.path.join(root_folder, patient_number_folder)
                if os.path.isdir(patient_folder_path):
                    type_folder_path = os.path.join(patient_folder_path, self.data_type)
                    if os.path.exists(type_folder_path):
                        # Iterate through files in data_type folder
                        for file_name in os.listdir(type_folder_path):
                            image_path = os.path.join(type_folder_path, file_name)
                            image_paths.append(image_path)
        
        return image_paths 
        
    def __len__(self):
        """ Returns the length of the dataset """
        return len(self.image_list)
    
    def __getitem__(self, index):
        """ Returns a preprocessed and transformed image and corresponding mask based on the index """
        image = Image.open(self.image_list[index])
        mask = Image.open(self.mask_list[index])

        if self.padding:
            image = ImageOps.pad(image, (self.resize, self.resize), method=Image.BOX)
            mask = ImageOps.pad(mask, (self.resize, self.resize), method=Image.BOX)
        else:
            image = image.resize((self.resize, self.resize))
            mask = mask.resize((self.resize, self.resize))

        if self.clahe:
            # clahe
            image = np.array(image)
            try:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            except:
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
            image[:,:,0] = clahe.apply(image[:,:,0])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
            image = Image.fromarray(image)

        return self.transf(image), self.transf(mask)
    
    def getimage(self, index):
        """ Shows a preprocessed and transformed image and corresponding mask based on the index """
        image = self.get_imgs_list(self.data_folder)[index]
        mask = self.get_imgs_list(self.mask_folder)[index]
        return Image.open(image), Image.open(mask)