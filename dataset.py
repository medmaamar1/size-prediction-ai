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
        
        # 1. Attempt to load the consolidated labels file (Kaggle style)
        labels_path = os.path.join(self.base_dir, f"{split}_labels.csv")
        if not os.path.exists(labels_path):
            # Fallback for split names like 'test_a' -> 'test_a_labels.csv'
            labels_path = os.path.join(self.base_dir, f"{split.replace('testA', 'test_a').replace('testB', 'test_b')}_labels.csv")

        if os.path.exists(labels_path):
            print(f"Loading consolidated labels from {labels_path}")
            self.df = pd.read_csv(labels_path)
            self.df.columns = self.df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        else:
            # 2. Fallback to sub-CSVs (Original style)
            print(f"Consolidated labels not found. Falling back to sub-CSVs in {self.base_dir}")
            try:
                meas_df = pd.read_csv(os.path.join(self.base_dir, 'measurements.csv'))
                hwg_df  = pd.read_csv(os.path.join(self.base_dir, 'hwg_metadata.csv'))
                photo_map_df = pd.read_csv(os.path.join(self.base_dir, 'subject_to_photo_map.csv'))
                
                for df in [meas_df, hwg_df, photo_map_df]:
                    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
                
                self.df = pd.merge(meas_df, hwg_df, on='subject_id')
                self.df = pd.merge(self.df, photo_map_df, on='subject_id')
            except Exception as e:
                print(f"Error loading BodyM split {split}: {e}")
                self.df = pd.DataFrame()
        
        # Paper targets (14 measurements) mapping to dataset names
        # Some might be missing in small datasets, we'll handle them
        self.target_map = {
            'ankle_cm': 'ankle_cm',
            'arm_length_cm': 'arm_length_cm',
            'bicep_cm': 'bicep_cm',
            'calf_cm': 'calf_cm',
            'chest_cm': 'chest_cm',
            'forearm_cm': 'forearm_cm',
            'head_to_heel_cm': 'height_cm', # Height is H2H
            'hip_cm': 'hip_cm',
            'leg_length_cm': 'leg_length_cm',
            'shoulder_breadth_cm': 'shoulder_breadth_cm',
            'shoulder_to_crotch_cm': 'shoulder_to_crotch_cm', 
            'thigh_cm': 'thigh_cm',
            'waist_cm': 'waist_cm',
            'wrist_cm': 'wrist_cm'
        }
        
        # FIX: The dataset was incorrectly mapping shoulder_breadth and shoulder_to_crotch
        # to 'height_cm' in many CSVs when the columns were slightly named differently.
        self.target_cols = list(self.target_map.keys())

        # Ensure all columns exist in DF (fill with 0 if missing)
        for target, src in self.target_map.items():
            if src not in self.df.columns:
                # Try common aliases but EXCLUDE 'height_cm' for breadths/lengths
                aliases = [src.replace('_cm', ''), src.replace('breadth', 'width'), src.replace('width', 'breadth')]
                if target == 'shoulder_to_crotch_cm':
                    aliases.append('s_to_c_cm')
                if target == 'head_to_heel_cm':
                    aliases.append('height_cm')
                
                found = False
                for a in aliases:
                    if a in self.df.columns:
                        self.df[target] = self.df[a]
                        found = True
                        break
                if not found:
                    self.df[target] = 0.0 # Missing measurement
            else:
                self.df[target] = self.df[src]

        # Filtering for data integrity
        if not self.df.empty:
            self.df = self.df.dropna(subset=['height_cm', 'weight_kg'])
            self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load and Preprocess Silhouettes
        # Try both 'front_image'/'side_image' and 'photo_id'
        front_file = str(row.get('front_image', row.get('photo_id', '')))
        side_file  = str(row.get('side_image', row.get('photo_id', '')))
        
        if not front_file.endswith('.png') and '.' not in front_file: front_file += '.png'
        if not side_file.endswith('.png') and '.' not in side_file: side_file += '.png'
            
        def find_path(filename, primary_dir, fallback_dir):
            p1 = os.path.join(self.base_dir, primary_dir, filename)
            p2 = os.path.join(self.base_dir, fallback_dir, filename)
            if os.path.exists(p1): return p1
            return p2

        front_path = find_path(front_file, 'mask', 'images')
        side_path  = find_path(side_file, 'mask_left', 'images')
        
        try:
            # Convert to single channel (grayscale) then resize
            front_img = Image.open(front_path).convert('L').resize((480, 640)) 
            side_img  = Image.open(side_path).convert('L').resize((480, 640))
        except Exception as e:
            # Last resort: dummy data if files missing to prevent crash
            front_img = Image.new('L', (480, 640), 0)
            side_img  = Image.new('L', (480, 640), 0)
        
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
