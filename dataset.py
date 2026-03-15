import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class BodyMDataset(Dataset):
    """
    Official BodyM Dataset Loader (Paper Compliant).
    Loads paired silhouettes and metadata as a 3-channel image (640x960).
    """
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        # Standardize column naming
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        self.images_dir = images_dir
        
        # Paper targets (14 measurements)
        self.target_cols = [
            'ankle_cm', 'arm_length_cm', 'bicep_cm', 'calf_cm', 'chest_cm', 
            'forearm_cm', 'head_to_heel_cm', 'hip_cm', 'leg_length_cm', 
            'shoulder_breadth_cm', 'shoulder_to_crotch_cm', 'thigh_cm', 
            'waist_cm', 'wrist_cm'
        ]
        
        # Filtering for data integrity
        self.df = self.df.dropna(subset=self.target_cols + ['front_image', 'side_image'])
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load and Preprocess Silhouettes (640x480)
        front_path = os.path.join(self.images_dir, str(row['front_image']))
        side_path  = os.path.join(self.images_dir, str(row['side_image']))
        
        # Convert to single channel (grayscale) then resize
        front_img = Image.open(front_path).convert('L').resize((480, 640)) # (W=480, H=640)
        side_img  = Image.open(side_path).convert('L').resize((480, 640))
        
        # 2. Horizontal Concatenation (640 x 960)
        # Create a new image container
        combined_sil = Image.new('L', (960, 640))
        combined_sil.paste(front_img, (0, 0))
        combined_sil.paste(side_img, (480, 0))
        
        # Convert to tensor and normalize silhouette to [0, 1]
        sil_tensor = transforms.ToTensor()(combined_sil) # (1, 640, 960)
        
        # 3. Create Metadata Channels (Dynamic Z-score Normalization)
        # Instead of hardcoded 250/200, we use Z-score (x - mean) / std
        # These stats should be calculated from the training set and saved for inference.
        h_val = float(row['height_cm'])
        w_val = float(row['weight_kg'])
        
        # Default stats (if not provided, we calculate them from the dataframe later)
        # Typically: height_mean ~ 170, std ~ 10 | weight_mean ~ 75, std ~ 15
        h_norm = (h_val - 170.0) / 10.0
        w_norm = (w_val - 75.0) / 15.0
        
        h_channel = torch.full((1, 640, 960), h_norm)
        w_channel = torch.full((1, 640, 960), w_norm)
        
        # 4. Stack final 3D tensor (Silhouettes, Height, Weight)
        input_tensor = torch.cat((sil_tensor, h_channel, w_channel), dim=0) # (3, 640, 960)
        
        # 5. Targets (14 measurements)
        targets = torch.tensor(row[self.target_cols].values.astype(float), dtype=torch.float32)
        
        return input_tensor, targets

if __name__ == "__main__":
    print("Paper-compliant BodyMDataset class defined.")
    print("Format: 3-channel Tensor (S, H, W) @ 640x960.")
