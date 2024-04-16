import os
import random
import shutil
from datasets import TVUSUterusSegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

data_root_folder = '/home/sandbox/dtank/my-scratch/data/'

def split_data(input_folder, train_folder, test_folder, validation_folder, train_ratio=0.7, test_ratio=0.2):
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
    num_validation = len(patient_folders) - num_train - num_test
    
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

# Define input folder containing patient folders
# input_folder = "/home/sandbox/dtank/my-scratch/data/original/UTISEG-DATA-ANONIEM/"
# # input_folder = "/home/sandbox/dtank/my-scratch/data/mask/UTISEG-DATA-ANNOTATED-MASK-ANONIEM/"

# # Define output folders for train, test, and validation sets
# train_folder = "/home/sandbox/dtank/my-scratch/data/original/train/"
# # train_folder = "/home/sandbox/dtank/my-scratch/data/mask/train/"
# test_folder = "/home/sandbox/dtank/my-scratch/data/original/test/"
# # test_folder = "/home/sandbox/dtank/my-scratch/data/mask/test/"
# validation_folder = "/home/sandbox/dtank/my-scratch/data/original/validation"
# # validation_folder = "/home/sandbox/dtank/my-scratch/data/mask/validation"

# # Split data into train, test, and validation sets
# split_data(input_folder, train_folder, test_folder, validation_folder)
# print('done dividing train, test, and validation sets.')

import os
import shutil
from sklearn.model_selection import KFold

def split_data_crossvalidation(folder_path, destination_folder_path):

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

# split_data_crossvalidation("/home/sandbox/dtank/my-scratch/data/mask/UTISEG-DATA-ANNOTATED-MASK-ANONIEM/", "/home/sandbox/dtank/my-scratch/data/crossvalidation")
# split_data_crossvalidation("/home/sandbox/dtank/my-scratch/data/original/UTISEG-DATA-ANONIEM/", "/home/sandbox/dtank/my-scratch/data/crossvalidation")

import os
from collections import defaultdict

def count_image_types(root_dir):
    image_types = defaultdict(int)
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                _, ext = os.path.splitext(file)
                image_types[ext.lower()] += 1
    return image_types

def main():
    root_folder = '/home/sandbox/dtank/my-scratch/data/original/validation/'  # Update this with your actual folder path
    for patient_folder in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, patient_folder)):
            images_count = count_image_types(os.path.join(root_folder, patient_folder, '3D'))
            print(f"Patient {patient_folder}: {images_count}")

if __name__ == "__main__":
    main()
