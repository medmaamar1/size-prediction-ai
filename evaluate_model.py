import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network import BMNet
from dataset import BodyMDataset
import numpy as np

def evaluate():
    print("🧪 --- STARTING MODEL EVALUATION --- 🧪")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Splits
    kaggle_base = "/kaggle/input/datasets/maamarmohamed/bodym-dataset/bodym"
    if not os.path.exists(kaggle_base):
        print(f"❌ Error: Dataset not found at {kaggle_base}")
        return

    splits = ['testA', 'testB']
    results = {}

    # 2. Load Model Once
    model = BMNet().to(device)
    model_path = "/kaggle/working/models/bmnet_best.pth"
    if not os.path.exists(model_path):
        model_path = "/kaggle/working/models/bmnet_latest.pth"
        
    if not os.path.exists(model_path):
        print(f"❌ Error: No model checkpoint found at {model_path}")
        return

    print(f"Loading weights from {model_path}...")
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the checkpoint is a dict containing 'model_state_dict' or the state_dict itself
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    
    # Bypassing MNASNet version check bug by direct parameter assignment
    model_state = model.state_dict()
    matched_keys = 0
    for name, param in new_state_dict.items():
        if name in model_state:
            if model_state[name].shape == param.shape:
                model_state[name].copy_(param)
                matched_keys += 1
    
    model.load_state_dict(model_state)
    print(f"      ✅ Weights loaded ({matched_keys} parameters matched).")
    model.eval()

    # 3. Evaluation Loop per Split
    for split in splits:
        print(f"\nEvaluating on {split}...")
        dataset = BodyMDataset(kaggle_base, split=split)
        if len(dataset) == 0:
            print(f"⚠️ Warning: Split {split} is empty or not found.")
            continue
            
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
        total_mae = 0
        num_samples = 0
        
        with torch.no_grad():
            for i, (images, measurements) in enumerate(loader):
                images = images.to(device)
                measurements = measurements.to(device)
                preds = model(images)
                error = torch.abs(preds - measurements)
                total_mae += error.sum().item()
                num_samples += measurements.numel()
        
        mae = total_mae / num_samples
        results[split] = mae
        print(f"✅ {split} MAE: {mae:.4f} cm")

    # 4. Final Summary
    overall_mae = sum(results.values()) / len(results) if results else 0
    print("\n" + "="*40)
    print(f"🏆 OVERALL ACCURACY (MAE): {overall_mae:.4f} cm")
    for s, m in results.items():
        print(f"   - {s}: {m:.4f} cm")
    print("="*40)
    
    print("\nInterpretation:")
    if overall_mae < 3.0:
        print("🟢 EXCELLENT: SOTA performance matching the paper.")
    elif overall_mae < 5.0:
        print("🟡 GOOD: Very usable for fashion/size recommendation.")
    else:
        print("🔴 NEEDS WORK: Model needs more training time.")

if __name__ == "__main__":
    evaluate()
