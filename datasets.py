import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import torchvision.transforms as tf

class TVUSUterusSegmentationDataset(Dataset):
    def __init__(self, data_folder, mask_folder, data_type, resize=128, transform=None):
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.data_type = data_type
        self.transf = transform 

        self.image_list = self.get_imgs_list(self.data_folder)
        self.mask_list = self.get_imgs_list(self.mask_folder)

        self.resize = resize
        
    def get_imgs_list(self, root_folder):
        image_paths = []

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
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        mask = Image.open(self.mask_list[index])

        image = image.resize((self.resize, self.resize))
        mask = mask.resize((self.resize, self.resize))

        # image.show()
        # mask.show()

        return self.transf(image), self.transf(mask)
    
    def getimage(self, index):
        image = self.get_imgs_list(self.data_folder)[index]
        mask = self.get_imgs_list(self.mask_folder)[index]
        return Image.open(image), Image.open(mask)