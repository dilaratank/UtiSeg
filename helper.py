import os
import random
import shutil
from datasets import TVUSUterusSegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms

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

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std


def return_mean_std(imaging_type):
    if imaging_type == "STILL":
        trans = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor()])

        img_size = 256
        clahe = False
        padding = False
        batch_size = 8

        STILL_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train/', data_root_folder+'mask/train/', 'STILL', resize=img_size, clahe=clahe, padding=padding, transform=trans)
        STILL_train_dataloader = DataLoader(STILL_train_dataset, batch_size=batch_size, shuffle=True)        

        mean, std = get_mean_std(STILL_train_dataloader)
        return mean, std