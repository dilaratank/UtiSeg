"""
dataloaders.py

Prepares a function that returns a TVUS dataset based on certain parameters.
The dataset can be used to train a U-Net Segmentation model. 
"""

from datasets import TVUSUterusSegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

class NonRandomHorizontalFlip:
    """
    A class that can be used to horizontally flip 50% of the dataset.

    Parameters
    ----------
    flip_probability: How much, in terms of percentage, of the data that will be flipped

    Returns
    -------
    sample: A horizontally flipped TVUS image
    """
    def __init__(self, flip_probability=0.5):
        self.flip_probability = flip_probability

    def __call__(self, sample):
        if torch.rand(1).item() < self.flip_probability:
            # Flip the image horizontally
            sample = torch.flip(sample, dims=[-1])
        return sample

# Prepare list of transformations based on split
train_trans = transforms.Compose([transforms.Grayscale()])
val_test_trans = transforms.Compose([transforms.Grayscale()])

def get_dataloaders(imaging_type, batch_size, img_size, clahe=False, padding=False, random_rotation=False, gaussian_blur=False, f=None):
    """
    A function that returns a TVUS segmentation dataset based on cetrain parameters.

    Parameters
    ----------
    imaging_type: type of the image in the dataset, can be still, video, 3D, or all.
    batch_size: batch size of the dataset
    img_size: image resolution, for example 256 would be a resolution of 256x256
    clahe: if CLAHE is applied to the data or not
    padding: if padding is applied to the data or not
    randomm_rotationn: if random rotation is applied to to the data or not
    gaussian_blur: if gaussian blur is applied to the data or not
    f: which fold should be trained on, if doing cross validation training


    Returns
    -------
    A TVUS segmentation dataset
    """

    #  Prepare further list of transformations based on function parameters
    if f != None:
        data_root_folder = f"/home/sandbox/dtank/my-scratch/data/crossvalidation/new/fold_{f}/"
    else:
        data_root_folder = '/home/sandbox/dtank/my-scratch/data/'
    
    if gaussian_blur:
        train_trans.transforms.append(transforms.GaussianBlur(5))
        val_test_trans.transforms.append(transforms.GaussianBlur(5))

    if random_rotation:
        train_trans.transforms.append(transforms.RandomApply([transforms.RandomRotation(5)], p=0.3))
    
    train_trans.transforms.append(transforms.ToTensor())
    val_test_trans.transforms.append(transforms.ToTensor())

    # Prepare datasets based on imaging types and other function parameters
    if imaging_type == "STILL":
        # STILL #
        # Datasets
        STILL_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train', data_root_folder+'mask/train', 'STILL', resize=img_size, clahe=clahe, padding=padding, transform=train_trans)
        STILL_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/test', data_root_folder+'mask/test', 'STILL', resize=img_size, clahe=clahe, padding=padding, transform=val_test_trans)
        STILL_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/validation', data_root_folder+'mask/validation', 'STILL', resize=img_size, clahe=clahe, padding=padding, transform=val_test_trans)

        # Dataloaders
        STILL_train_dataloader = DataLoader(STILL_train_dataset, batch_size=batch_size, shuffle=True)
        STILL_test_dataloader = DataLoader(STILL_test_dataset, batch_size=batch_size, shuffle=True)
        STILL_val_dataloader = DataLoader(STILL_val_dataset, batch_size=batch_size, shuffle=True)
        return STILL_train_dataloader, STILL_val_dataloader, STILL_test_dataloader
    
    elif imaging_type == "VIDEO":
        # VIDEO #
        # Datasets 
        VIDEO_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train', data_root_folder+'mask/train', 'VIDEO', resize=img_size, clahe=clahe, padding=padding, transform=train_trans)
        VIDEO_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/test', data_root_folder+'mask/test', 'VIDEO', resize=img_size, clahe=clahe, padding=padding, transform=val_test_trans)
        VIDEO_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/validation', data_root_folder+'mask/validation', 'VIDEO', resize=img_size, padding=padding, clahe=clahe, transform=val_test_trans)

        # Dataloaders
        VIDEO_train_dataloader = DataLoader(VIDEO_train_dataset, batch_size=batch_size, shuffle=True)
        VIDEO_test_dataloader = DataLoader(VIDEO_test_dataset, batch_size=batch_size, shuffle=True)
        VIDEO_val_dataloader = DataLoader(VIDEO_val_dataset, batch_size=batch_size, shuffle=True)
        return VIDEO_train_dataloader, VIDEO_val_dataloader, VIDEO_test_dataloader
    
    elif imaging_type == "3D":
        # 3D #
        # Datasets
        VOLUME_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train', data_root_folder+'mask/train', '3D', resize=img_size, clahe=clahe, padding=padding, transform=train_trans)
        VOLUME_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/test', data_root_folder+'mask/test', '3D', resize=img_size, clahe=clahe, padding=padding, transform=val_test_trans)
        VOLUME_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/validation', data_root_folder+'mask/validation', '3D', resize=img_size, clahe=clahe, padding=padding, transform=val_test_trans)

        # Dataloaders
        VOLUME_train_dataloader = DataLoader(VOLUME_train_dataset, batch_size=batch_size, shuffle=True)
        VOLUME_test_dataloader = DataLoader(VOLUME_test_dataset, batch_size=batch_size, shuffle=True)
        VOLUME_val_dataloader = DataLoader(VOLUME_val_dataset, batch_size=batch_size, shuffle=True)
        return VOLUME_train_dataloader, VOLUME_val_dataloader, VOLUME_test_dataloader
    
    elif imaging_type == "ALL":
        # ALL Imaging Types #
        # Datasets
        ALL_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/all_train', data_root_folder+'mask/all_train', 'ALL', resize=img_size, clahe=clahe, padding=padding, transform=train_trans)
        ALL_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/all_test', data_root_folder+'mask/all_test', 'ALL', resize=img_size, clahe=clahe, padding=padding, transform=val_test_trans)
        ALL_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/all_validation', data_root_folder+'mask/all_validation', 'ALL', resize=img_size, clahe=clahe, padding=padding, transform=val_test_trans)

        # Dataloaders
        ALL_train_dataloader = DataLoader(ALL_train_dataset, batch_size=batch_size, shuffle=True)
        ALL_test_dataloader = DataLoader(ALL_test_dataset, batch_size=batch_size, shuffle=True)
        ALL_val_dataloader = DataLoader(ALL_val_dataset, batch_size=batch_size, shuffle=True)
        return ALL_train_dataloader, ALL_test_dataloader, ALL_val_dataloader 