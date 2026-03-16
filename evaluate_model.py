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
    
    # 1. Load Dataset (Validation Set - TestA)
    kaggle_base = "/kaggle/input/datasets/maamarmohamed/bodym-dataset/bodym"
    if not os.path.exists(kaggle_base):
        print(f"❌ Error: Dataset not found at {kaggle_base}")
        return

    print("Loading validation data (TestA)...")
    val_dataset = BodyMDataset(kaggle_base, split='testA')
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    if len(val_dataset) == 0:
        print("❌ Error: Validation dataset is empty.")
        return

    # 2. Load Model
    model = BMNet().to(device)
    model_path = "/kaggle/working/models/bmnet_best.pth"
    
    if not os.path.exists(model_path):
        # Fallback to latest if best doesn't exist
        model_path = "/kaggle/working/models/bmnet_latest.pth"
        
    if not os.path.exists(model_path):
        print(f"❌ Error: No model checkpoint found at {model_path}")
        return

    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    # Handle DataParallel state_dict if necessary
    state_dict = checkpoint['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Evaluation Loop
    total_mae = 0
    num_samples = 0
    
    print(f"Evaluating on {len(val_dataset)} samples...")
    
    with torch.no_grad():
        for i, (images, measurements) in enumerate(val_loader):
            images = images.to(device)
            measurements = measurements.to(device)
            
            preds = model(images)
            
            # MAE Calculation (centimeters)
            error = torch.abs(preds - measurements)
            total_mae += error.sum().item()
            num_samples += measurements.numel()
            
            if i % 20 == 0:
                current_mae = total_mae / num_samples
                print(f"      Batch {i}/{len(val_loader)} | Current MAE: {current_mae:.2f} cm")

    final_mae = total_mae / num_samples
    print("\n" + "="*30)
    print(f"🏆 FINAL ACCURACY (MAE): {final_mae:.4f} cm")
    print("="*30)
    print("\nInterpretation:")
    if final_mae < 3.0:
        print("🟢 EXCELLENT: SOTA performance matching the paper.")
    elif final_mae < 5.0:
        print("🟡 GOOD: Very usable for fashion/size recommendation.")
    else:
        print("🔴 NEEDS WORK: Model needs more training time.")

if __name__ == "__main__":
    evaluate()
