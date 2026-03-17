import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network import BMNet
from dataset import BodyMDataset
import numpy as np

def evaluate():
    print("🧪 --- STARTING SYNTHETIC MODEL EVALUATION (SMPL-X) --- 🧪")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from smpl_generator import SMPLDataGenerator
    generator = SMPLDataGenerator()
    
    # 1. Load Model
    model = BMNet().to(device)
    model_path = "/kaggle/input/models/maamarmohamed12/ai-model/pytorch/default/1/bmnet_best.pth"
    if not os.path.exists(model_path):
        model_path = "/kaggle/input/models/maamarmohamed12/ai-model/pytorch/default/1/bmnet_checkpoint.pth"
    
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
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    metrics_names = [
        'Ankle', 'Arm-L', 'Bicep', 'Calf', 'Chest', 
        'Forearm', 'H2H', 'Hip', 'Leg-L', 
        'Shoulder-B', 'S-to-C', 'Thigh', 'Waist', 'Wrist'
    ]

    print(f"\n🔍 Evaluating on 100 Synthetic SMPL Bodies...")
    
    total_mae = 0
    per_metric_mae = np.zeros(14)
    num_samples = 100

    with torch.no_grad():
        for i in range(num_samples):
            # Generate random shape
            betas = torch.randn(1, 10, device=device) * 1.5 # Moderate diversity
            combined_sil, gt_measurements, metadata = generator.generate_batch(betas)
            
            # Prepare internal model input
            h_channel = metadata[:, 0].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
            w_channel = metadata[:, 1].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
            inputs = torch.cat([combined_sil / 255.0, h_channel, w_channel], dim=1)
            
            preds = model(inputs)
            error = torch.abs(preds - gt_measurements)
            
            total_mae += error.mean().item()
            per_metric_mae += error.mean(dim=0).cpu().numpy()
            
            if (i+1) % 20 == 0:
                print(f"   Processed {i+1}/{num_samples} bodies...")

    print(f"\n✅ Final Results (Synthetic MAE): {total_mae/num_samples:.4f} cm")
    print("-" * 30)
    for name, mae in zip(metrics_names, per_metric_mae/num_samples):
        print(f"{name:12}: {mae:.4f} cm")

    print("\nInterpretation:")
    avg_mae = total_mae / num_samples
    if avg_mae < 3.0:
        print("🟢 EXCELLENT: SOTA performance matching the paper.")
    elif avg_mae < 5.0:
        print("🟡 GOOD: Very usable for fashion/size recommendation.")
    else:
        print("🔴 NEEDS WORK: Model needs more training time.")

if __name__ == "__main__":
    evaluate()

