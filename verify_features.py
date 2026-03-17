import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import BodyMDataset
from network import BMNet

def test_feature_alignment():
    print("🧪 --- VERIFYING DATASET FEATURES AND MODEL INPUT ALIGNMENT --- 🧪")
    
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kaggle_base = "/kaggle/input/datasets/maamarmohamed12/bodym-dataset/bodym"
    split = "train" # CHANGED TO TRAIN 
    
    if not os.path.exists(kaggle_base):
        print(f"⚠️ Warning: Dataset not found at {kaggle_base}.")
        return

    # 2. Dataset Verification
    print(f"📦 Loading dataset from: {kaggle_base}, split: {split}")
    dataset = BodyMDataset(kaggle_base, split=split)
    
    if len(dataset) == 0:
        print(f"❌ Error: Dataset is empty.")
        return

    print(f"✅ Dataset loaded with {len(dataset)} samples.")
    
    # --- ADDED DIAGNOSTIC: Check Silhouette Mean for one sample ---
    print("\n📸 Silhouette Diagnostic:")
    img, target = dataset[0]
    sil_mean = img[0].mean().item()
    print(f"   - Input Shape: {img.shape}")
    print(f"   - Combined Silh Mean: {sil_mean:.4f}")
    if sil_mean < 0.20:
        print("   ⚠️ WARNING: Silh mean is low (< 0.20). One profile might be missing (BLACK).")
    else:
        print("   ✅ Silh mean looks healthy (> 0.20).")

    # --- ADDED DIAGNOSTIC: Check Target values ---
    print("\n🎯 Target Alignment Diagnostic (First Sample):")
    for idx, name in enumerate(dataset.target_cols):
        val = target[idx].item()
        print(f"   {idx:02d}: {name.ljust(20)} = {val:.2f} cm")
        if val == 0:
            print(f"      ❌ ERROR: {name} is 0.0! Column fallback failed.")

    # 3. Check Feature Mapping (target_cols)
    expected_features = [
        'ankle_cm', 'arm_length_cm', 'bicep_cm', 'calf_cm', 'chest_cm', 
        'forearm_cm', 'head_to_heel_cm', 'hip_cm', 'leg_length_cm', 
        'shoulder_breadth_cm', 'shoulder_to_crotch_cm', 'thigh_cm', 
        'waist_cm', 'wrist_cm'
    ]
    
    print("\n🔍 Checking feature mapping in BodyMDataset:")
    dataset_features = dataset.target_cols
    missing = [f for f in expected_features if f not in dataset_features]
    if missing:
        print(f"❌ Missing features in dataset.target_cols: {missing}")
    else:
        print(f"✅ All {len(expected_features)} expected features found in dataset mapping.")

    # 4. Data Loader and Batch Verification
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    images, measurements = next(iter(loader))
    
    print("\n🔢 Batch Verification:")
    print(f"   - Image Batch Shape: {images.shape} (Expected: [N, 3, H, W])")
    print(f"   - Measurements Batch Shape: {measurements.shape} (Expected: [N, 14])")
    
    if images.shape[1] != 3:
        print(f"❌ Error: Image channels should be 3 (Front, Side, Depth/Height-Weight). Found {images.shape[1]}.")
    else:
        print(f"✅ Image channels aligned.")

    if measurements.shape[1] != 14:
        print(f"❌ Error: Measurement features should be 14. Found {measurements.shape[1]}.")
    else:
        print(f"✅ Measurement count aligned.")

    # 5. Model Input/Output Alignment
    print("\n🤖 Checking Model Input/Output Alignment:")
    model = BMNet().to(device)
    model.eval()
    
    with torch.no_grad():
        test_input = images.to(device)
        output = model(test_input)
        
        print(f"   - Model Input Shape: {test_input.shape}")
        print(f"   - Model Output Shape: {output.shape} (Expected: [N, 14])")
        
        if output.shape == measurements.shape:
            print(f"✅ SUCCESS: Model output matches dataset target shape.")
        else:
            print(f"❌ FAILURE: Shape mismatch! Output: {output.shape} vs Target: {measurements.shape}")

    # 6. SMPL Generator Integration Check (If applicable)
    print("\n🧬 SMPL Generator Compatibility Check:")
    try:
        from smpl_generator import SMPLDataGenerator
        smpl_features = SMPLDataGenerator().target_cols
        print(f"   - SMPL Generator features: {len(smpl_features)}")
        
        if set(smpl_features) == set(expected_features):
            print("✅ SMPL Generator features match training/eval targets.")
        else:
            diff = set(expected_features) - set(smpl_features)
            print(f"❌ SMPL features mismatch! Missing from SMPL: {diff}")
    except Exception as e:
        print(f"ℹ️ SMPL Generator check skipped or failed (likely missing dependencies/models): {e}")

if __name__ == "__main__":
    test_feature_alignment()
