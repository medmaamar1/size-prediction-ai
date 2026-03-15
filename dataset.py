import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class BodyMDataset(Dataset):
    """
    Kaggle-Aware BodyM Dataset Loader.
    Handles joining measurements.csv, hwg_metadata.csv, and subject_to_photo_map.csv.
    Loads front/side silhouettes from mask/ and mask_left/ folders.
    """
    def __init__(self, base_dir, split='train', transform=None):
        self.base_dir = os.path.join(base_dir, split)
        
        # 1. Load sub-CSVs
        meas_df = pd.read_csv(os.path.join(self.base_dir, 'measurements.csv'))
        hwg_df  = pd.read_csv(os.path.join(self.base_dir, 'hwg_metadata.csv'))
        photo_map_df = pd.read_csv(os.path.join(self.base_dir, 'subject_to_photo_map.csv'))
        
        # Standardize columns
        for df in [meas_df, hwg_df, photo_map_df]:
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
            
        # 2. Join Metadata
        # Merge measurements with height/weight/gender
        self.df = pd.merge(meas_df, hwg_df, on='subject_id')
        
        # 3. Map to Photos
        # photo_map_df typically maps subject_id to filenames
        # We assume one front and one side photo per subject for this simplified loader
        # Filter for subjects that have both front and side entries in the map if necessary
        self.df = pd.merge(self.df, photo_map_df, on='subject_id')
        
        # Paper targets (14 measurements)
        self.target_cols = [
            'ankle_cm', 'arm_length_cm', 'bicep_cm', 'calf_cm', 'chest_cm', 
            'forearm_cm', 'head_to_heel_cm', 'hip_cm', 'leg_length_cm', 
            'shoulder_breadth_cm', 'shoulder_to_crotch_cm', 'thigh_cm', 
            'waist_cm', 'wrist_cm'
        ]
        
        # Filtering for data integrity
        cols_to_check = self.target_cols + ['height_cm', 'weight_kg', 'photo_id']
        self.df = self.df.dropna(subset=[c for c in cols_to_check if c in self.df.columns])
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load and Preprocess Silhouettes (using Kaggle structure)
        # Assuming photo_id corresponds to the filename in mask/ and mask_left/
        photo_id = str(row['photo_id'])
        if not photo_id.endswith('.png'):
            photo_id += '.png'
            
        front_path = os.path.join(self.base_dir, 'mask', photo_id)
        side_path  = os.path.join(self.base_dir, 'mask_left', photo_id)
        
        # Convert to single channel (grayscale) then resize
        front_img = Image.open(front_path).convert('L').resize((480, 640)) 
        side_img  = Image.open(side_path).convert('L').resize((480, 640))
        
        # 2. Horizontal Concatenation (640 x 960)
        combined_sil = Image.new('L', (960, 640))
        combined_sil.paste(front_img, (0, 0))
        combined_sil.paste(side_img, (480, 0))
        
        # Convert to tensor and normalize silhouette to [0, 1]
        sil_tensor = transforms.ToTensor()(combined_sil) 
        
        # 3. Metadata Channels (Height/Weight)
        h_val = float(row['height_cm'])
        w_val = float(row['weight_kg'])
        
        # Z-score Normalization (Paper stats)
        h_norm = (h_val - 170.0) / 10.0
        w_norm = (w_val - 75.0) / 15.0
        
        h_channel = torch.full((1, 640, 960), h_norm)
        w_channel = torch.full((1, 640, 960), w_norm)
        
        # 4. Stack final 3D tensor
        input_tensor = torch.cat((sil_tensor, h_channel, w_channel), dim=0) 
        
        # 5. Targets
        targets = torch.tensor(row[self.target_cols].values.astype(float), dtype=torch.float32)
        
        return input_tensor, targets

if __name__ == "__main__":
    print("Paper-compliant BodyMDataset class defined.")
    print("Format: 3-channel Tensor (S, H, W) @ 640x960.")
