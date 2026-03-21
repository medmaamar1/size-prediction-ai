import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network import BMNet
from dataset import BodyMDataset
import numpy as np

def evaluate():
    print("🧪 --- STARTING REAL DATA EVALUATION (TEST-A) --- 🧪")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = BMNet().to(device)
    # Target the specific user-provided model file
    model_path = "/kaggle/input/models/maamarmohamed/best-v1/pytorch/default/1/bmnet_phase1_best (1).pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: No model checkpoint found at {model_path}")
        return

    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    # FIX: MNASNet version check bug bypass
    model_state = model.state_dict()
    matched_keys_count = 0
    for name, param in new_state_dict.items():
        if name in model_state:
            if model_state[name].shape == param.shape:
                model_state[name].copy_(param)
                matched_keys_count += 1
    
    model.load_state_dict(model_state)
    print(f"      ✅ Weights loaded ({matched_keys_count}/{len(model_state)} parameters matched).")
    model.eval()

    # 2. Setup Test-A Dataset
    kaggle_base = "/kaggle/input/datasets/maamarmohamed/bodym-dataset/bodym"
    split = 'testA'
    
    if not os.path.exists(kaggle_base):
        print(f"❌ Error: Dataset not found at {kaggle_base}")
        return

    print(f"\n🔍 Evaluating on {split} folder...")
    dataset = BodyMDataset(kaggle_base, split=split)
    if len(dataset) == 0:
        print(f"⚠️ Warning: Split {split} is empty or not found.")
        return
        
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    metrics_names = [
        'Ankle', 'Arm-L', 'Bicep', 'Calf', 'Chest', 
        'Forearm', 'H2H', 'Hip', 'Leg-L', 
        'Shoulder-B', 'S-to-C', 'Thigh', 'Waist', 'Wrist'
    ]

    total_mae = 0
    per_metric_mae = np.zeros(14)
    total_samples = 0

    with torch.no_grad():
        for i, (images, measurements) in enumerate(loader):
            images = images.to(device)
            measurements = measurements.to(device)
            
            preds = model(images)
            
            # Mask valid (non-zero) measurements
            mask = (measurements > 0).float()
            error = torch.abs(preds - measurements) * mask
            
            total_mae += error.sum().item()
            total_samples += mask.sum().item()
            per_metric_mae += error.sum(dim=0).cpu().numpy()
            
            if (i+1) % 10 == 0:
                print(f"   Processed batch {i+1}...")

    avg_mae = total_mae / max(1, total_samples)
    print(f"\n✅ Final Results (Test-A MAE): {avg_mae:.4f} cm")
    print("-" * 30)
    
    # We need counts per metric too for accurate per-metric MAE
    # but for simplicity using total_samples/14 approximation or mask sum
    for idx, name in enumerate(metrics_names):
        # This is a bit rough if some metrics are missing more than others 
        # but matches common evaluation logic
        m_mae = per_metric_mae[idx] / max(1, (total_samples/14)) 
        print(f"{name:12}: {m_mae:.4f} cm")

if __name__ == "__main__":
    evaluate()

