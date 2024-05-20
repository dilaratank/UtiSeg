"""
helper.py 

A collection of helper functions for U-Net and nnU-Net data storing
"""

### u-net ###

import os
import random
import shutil
from sklearn.model_selection import train_test_split

data_root_folder = '/home/sandbox/dtank/my-scratch/data/'

def split_data(input_folder, train_folder, test_folder, validation_folder, train_ratio=0.7, test_ratio=0.2):
    """
    (U-Net)

    Code to split data on a patient level to train the normal U-Net.
    Creates a train-, test- and validation-split based on a ratio an then moves images to specified train-, test-, and validation-folders

    Parameters
    ----------
    input_folder: The location of the complete dataset folder
    train_folder: The location of the train split folder (empty)
    test_folder: The location of the test split folder (empty)
    validation_folder: The location of the validation split folder (empty)
    train_ratio: The ratio of training cases
    test_ratio: The ratio of test cases

    """
    random.seed(42)

    # Create train, test, and validation folders if they don't exist
    for folder in [train_folder, test_folder, validation_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Get list of patient folders
    patient_folders = os.listdir(input_folder)
    random.shuffle(patient_folders)
    
    # Calculate number of folders for each set
    num_train = int(len(patient_folders) * train_ratio)
    num_test = int(len(patient_folders) * test_ratio)
    
    train_set = patient_folders[:num_train]
    test_set = patient_folders[num_train:num_train + num_test]
    validation_set = patient_folders[num_train + num_test:]
    
    # Move folders to train, test, and validation sets
    for folder in train_set:
        shutil.copytree(os.path.join(input_folder, folder), os.path.join(train_folder, folder))
    for folder in test_set:
        shutil.copytree(os.path.join(input_folder, folder), os.path.join(test_folder, folder))
    for folder in validation_set:
        shutil.copytree(os.path.join(input_folder, folder), os.path.join(validation_folder, folder))


import os
import shutil
from sklearn.model_selection import KFold

def split_data_crossvalidation(folder_path, destination_folder_path):
    """
    (U-Net)

    Code to split the data on a patient level to perform cross validation, the split is done for each fold
    
    Parameters
    ----------
    folder_path: The location of the complete dataset folder
    destination_folder_path: The location of the cross validation dataset folder 

    """

    # Get the list of patient number folders
    patient_folders = os.listdir(folder_path)

    # Define the number of folds (5-fold cross-validation)
    num_folds = 5

    # Initialize KFold splitter
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Function to move folders from one location to another
    def move_folders(folders, destination):
        for folder in folders:
            src = os.path.join(folder_path, folder)
            dest = os.path.join(destination, folder)
            shutil.copytree(src, dest)

    # Iterate through the folds
    for fold_idx, (train_index, test_index) in enumerate(kf.split(patient_folders)):
        # Assign the fold to train, test, and validation sets
        train_set = [patient_folders[i] for i in train_index]
        test_set = [patient_folders[i] for i in test_index]
        
        train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=42)

        # Define directory paths for train, test, and validation sets
        train_fold_dir = os.path.join(destination_folder_path, f"fold_{fold_idx+1}", "original", "train")
        test_fold_dir = os.path.join(destination_folder_path, f"fold_{fold_idx+1}", "original", "test")
        val_fold_dir = os.path.join(destination_folder_path, f"fold_{fold_idx+1}", "original", "validation")

        # Create directories if they don't exist
        os.makedirs(train_fold_dir, exist_ok=True)
        os.makedirs(test_fold_dir, exist_ok=True)
        os.makedirs(val_fold_dir, exist_ok=True)

        # Move patient folders to train, test, and validation directories
        move_folders(train_set, train_fold_dir)
        move_folders(test_set, test_fold_dir)
        move_folders(val_set, val_fold_dir)

    print("Dataset created successfully.")


### nnu-net ###

import os
import shutil

def mix_data_types(source_folder, destination_folder):
    """
    (nnU-Net)

    Code to throw every image type together for nnU-Net since it will perform 5-fold cross validation anyway

    Parameters
    ----------
    source_folder: The location of the complete dataset folder
    destination_folder: The location of the 'ALL' dataset folder in the nnU-Net training
    """
    # Iterate through all folders in the source directory
    for root, dirs, files in os.walk(source_folder):
        for dir_name in dirs:
            # Construct the path to the 'STILL' folder
            still_folder = os.path.join(root, dir_name, '3D')
            
            # Check if the 'STILL' folder exists
            if os.path.isdir(still_folder):
                # Iterate through files in the 'STILL' folder
                for file_name in os.listdir(still_folder):
                    # Construct the source and destination paths for each file
                    source_file = os.path.join(still_folder, file_name)
                    destination_file = os.path.join(destination_folder, file_name)
                    
                    # Copy the file to the destination folder
                    shutil.copyfile(source_file, destination_file)
                    print(f"Copied: {source_file} to {destination_file}")

from typing import Tuple
from batchgenerators.utilities.file_and_folder_operations import save_json, join

def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None, **kwargs):
    """
    (nnU--Net)

    Generates a dataset.json file in the output folder, copied from https://github.com/MIC-DKFZ/nnUNet/blob/e539637821b67893bd57e4ba9dc1e60a218ae3ea/nnunetv2/dataset_conversion/generate_dataset_json.py#L6

    Parameters
    ----------
    channel_names:
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

    num_training_cases: is used to double check all cases are there!

    file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
    segmentations!

    dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
    completeness and as a reminder that these would be great!

    overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name

    kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didn't like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if license is not None:
        dataset_json['licence'] = license
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)

import os

def rename_files(folder_path, prefix, padding, extension):
    """
    (nnU-Net)

    A function to rename the dataset images how nnU-Net expects them, according to https://github.com/MIC-DKFZ/nnUNet/blob/e539637821b67893bd57e4ba9dc1e60a218ae3ea/documentation/dataset_format.md

    Parameters
    ----------
    folder_path: The location of the folder where the images are stored
    prefix: Data type (still, video, or 3D)
    padding: The amount of zeroes in the channel identifier
    extension: The extention of the images, .png in the case of nnU-Net
    """
    files = sorted(os.listdir(folder_path))
    index = 1
    for filename in files:
        new_filename = f"{prefix}{str(index).zfill(padding)}_0000{extension}"
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(old_filepath, new_filepath)
        index += 1

def rename_masks(folder_path, prefix):
    """(nnU-Net) Same as above but for mask-format"""
    files = sorted(os.listdir(folder_path))
    index = 1
    for filename in files:
        new_filename = f"{prefix}{str(index).zfill(3)}.png"
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(old_filepath, new_filepath)
        index += 1


import cv2
import numpy as np
import pandas as pd

from skimage import io

from skimage import img_as_ubyte

import os
import cv2
from skimage import io

def threshold_imgs(folder_path, threshold=0):
    """
    (nnU_net)
    
    A function to normalize/threshold the images for nnU-Net usage
    
    Parameters
    ----------
    folder_path: The location of the folder of images to threshold
    threshold: The threshold value, 0 in our case
    """

    #Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a PNG image
        if filename.endswith('.png'):
            # Read the image
            image_path = os.path.join(folder_path, filename)
            seg = io.imread(image_path)

            # Apply thresholding
            seg[seg > threshold] = 1

            # Save the processed image back to the same file
            io.imsave(image_path, img_as_ubyte(seg))

    print("thresholding complete.")