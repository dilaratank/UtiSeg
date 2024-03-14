from datasets import TVUSUterusSegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms

data_root_folder = '/home/sandbox/dtank/my-scratch/data/'
# data_root_folder = '/home/sandbox/dtank/my-scratch/test_data/'
# STILL_batch_size = 1
# VIDEO_batch_size = 4
# VOLUME_batch_size = 4

# Transforms
trans = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()])

def get_dataloaders(imaging_type, batch_size, img_size):
    if imaging_type == "STILL":
        # STILL #
        # Datasets
        STILL_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train/', data_root_folder+'mask/train/', 'STILL', resize=img_size, transform=trans)
        STILL_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/test/', data_root_folder+'mask/test/', 'STILL', resize=img_size, transform=trans)
        STILL_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/validation/', data_root_folder+'mask/validation/', 'STILL', resize=img_size, transform=trans)

        # Dataloaders
        STILL_train_dataloader = DataLoader(STILL_train_dataset, batch_size=batch_size, shuffle=True)
        STILL_test_dataloader = DataLoader(STILL_test_dataset, batch_size=batch_size, shuffle=True)
        STILL_val_dataloader = DataLoader(STILL_val_dataset, batch_size=batch_size, shuffle=True)
        return STILL_train_dataloader, STILL_val_dataloader, STILL_test_dataloader
    
    elif imaging_type == "VIDEO":
        # VIDEO #
        # Datasets 
        VIDEO_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train', data_root_folder+'mask/train', 'VIDEO', resize=img_size, transform=trans)
        VIDEO_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/test', data_root_folder+'mask/test', 'VIDEO', resize=img_size, transform=trans)
        VIDEO_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/validation', data_root_folder+'mask/validation', 'VIDEO', resize=img_size, transform=trans)

        # Dataloaders
        VIDEO_train_dataloader = DataLoader(VIDEO_train_dataset, batch_size=batch_size, shuffle=True)
        VIDEO_test_dataloader = DataLoader(VIDEO_test_dataset, batch_size=batch_size, shuffle=True)
        VIDEO_val_dataloader = DataLoader(VIDEO_val_dataset, batch_size=batch_size, shuffle=True)
        return VIDEO_train_dataloader, VIDEO_val_dataloader, VIDEO_test_dataloader
    
    elif imaging_type == "3D":
        # 3D #
        # Datasets
        VOLUME_train_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/train', data_root_folder+'mask/train', '3D', resize=img_size, transform=trans)
        VOLUME_test_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/test', data_root_folder+'mask/test', '3D', resize=img_size, transform=trans)
        VOLUME_val_dataset = TVUSUterusSegmentationDataset(data_root_folder+'original/validation', data_root_folder+'mask/validation', '3D', resize=img_size, transform=trans)

        # Dataloaders
        VOLUME_train_dataloader = DataLoader(VOLUME_train_dataset, batch_size=batch_size, shuffle=True)
        VOLUME_test_dataloader = DataLoader(VOLUME_test_dataset, batch_size=batch_size, shuffle=True)
        VOLUME_val_dataloader = DataLoader(VOLUME_val_dataset, batch_size=batch_size, shuffle=True)
        return VOLUME_train_dataloader, VOLUME_val_dataloader, VOLUME_test_dataloader

