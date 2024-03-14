from datasets import TVUSUterusSegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms

data_root_folder = '/home/sandbox/dtank/my-scratch/data/'
# data_root_folder = '/home/sandbox/dtank/my-scratch/test_data/'
STILL_batch_size = 1
VIDEO_batch_size = 4
VOLUME_batch_size = 4

# Transforms
trans = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()])

# STILL #

# Datasets
STILL_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train/', data_root_folder+'mask/train/', 'STILL', transform=trans)
STILL_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/test/', data_root_folder+'mask/test/', 'STILL', transform=trans)
STILL_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/validation/', data_root_folder+'mask/validation/', 'STILL', transform=trans)

# Dataloaders
STILL_train_dataloader = DataLoader(STILL_train_dataset, batch_size=STILL_batch_size, shuffle=True)
STILL_test_dataloader = DataLoader(STILL_test_dataset, batch_size=STILL_batch_size, shuffle=True)
STILL_val_dataloader = DataLoader(STILL_val_dataset, batch_size=STILL_batch_size, shuffle=True)


# VIDEO #

# Datasets 
VIDEO_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train', data_root_folder+'mask/train', 'VIDEO', transform=trans)
VIDEO_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/test', data_root_folder+'mask/test', 'VIDEO', transform=trans)
VIDEO_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/validation', data_root_folder+'mask/validation', 'VIDEO', transform=trans)

# Dataloaders
VIDEO_train_dataloader = DataLoader(VIDEO_train_dataset, batch_size=VIDEO_batch_size, shuffle=True)
VIDEO_test_dataloader = DataLoader(VIDEO_test_dataset, batch_size=VIDEO_batch_size, shuffle=True)
VIDEO_val_dataloader = DataLoader(VIDEO_val_dataset, batch_size=VIDEO_batch_size, shuffle=True)

# 3D #

# # Datasets
# VOLUME_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train', data_root_folder+'mask/train', '3D')
# VOLUME_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/test', data_root_folder+'mask/test', '3D')
# VOLUME_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/validation', data_root_folder+'mask/validation', '3D')

# Dataloaders
# Dataloaders
# VOLUME_train_dataloader = DataLoader(VOLUME_train_dataset, batch_size=VOLUME_batch_size, shuffle=True)
# VOLUME_test_dataloader = DataLoader(VOLUME_test_dataset, batch_size=VOLUME_batch_size, shuffle=True)
# VOLUME_val_dataloader = DataLoader(VOLUME_val_dataset, batch_size=VOLUME_batch_size, shuffle=True)

