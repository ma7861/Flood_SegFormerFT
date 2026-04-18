import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import os
import albumentations as A



training_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])


class Sen1Floods11Dataset(Dataset):
    def __init__(self, sar_dir, label_dir, split_list_df, min_db=-30.0, max_db=0.0, max_ratio=4.0, transform=None):
        self.sar_dir = sar_dir
        self.label_dir = label_dir
        self.split_list_df = split_list_df
        self.min_db = min_db
        self.max_db = max_db
        self.max_ratio = max_ratio
        self.transform = transform

    def __len__(self):
        return len(self.split_list_df)

    def __getitem__(self, i):
        sar_file = self.sar_dir + self.split_list_df['SAR'].iloc[i]
        label_file =  self.label_dir + self.split_list_df['Label'].iloc[i]
        
        with rasterio.open(sar_file) as src:
            sar_img = src.read() #(2, 512, 512)
            
        sar_img = np.nan_to_num(sar_img, nan=self.min_db, posinf=self.max_db, neginf=self.min_db)
        sar_img = np.clip(sar_img, self.min_db, self.max_db)
        
        vv = sar_img[0]
        vh = sar_img[1]

        third_channel_ratio = vv / (vh + 1e-5) #SegFormer takes 3 channel input
        third_channel_ratio = np.clip(third_channel_ratio, 0.0, self.max_ratio)

        sar_3channel = np.stack([vv, vh, third_channel_ratio], axis=-1) #(512, 512, 3)
        
        # Normalizing to 0-255
        sar_3channel = (((sar_3channel - sar_3channel.min()) / (sar_3channel.max() - sar_3channel.min() + 1e-5)) * 255).astype(np.uint8)

        with rasterio.open(label_file) as src:
            label = src.read(1) #(512, 512)
        
        label[label == -1] = 255 #-1 is the invalid pixels, will not be used by model
        
        if self.transform: #Since Im only using the 400odd hand labelled images, using augmentation on training set
            augmented = self.transform(image=sar_3channel, mask=label)
            sar_3channel = augmented['image']
            label = augmented['mask']
            
        sar_tensor = torch.tensor(sar_3channel).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.tensor(label).long()
        
        return {"pixel_values": sar_tensor, "labels": label_tensor}