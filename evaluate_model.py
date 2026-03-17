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
    kaggle_base = "/kaggle/input/datasets/maamarmohamed12/bodym-dataset/bodym"
    if not os.path.exists(kaggle_base):
        print(f"❌ Error: Dataset not found at {kaggle_base}")
        return

    splits = ['train']
    results = {}

    # 2. Load Model Once
    model = BMNet().to(device)
    model_path = "/kaggle/input/models/maamarmohamed12/ai-model/pytorch/default/1/bmnet_best.pth"
    if not os.path.exists(model_path):
        # Fallback to checkpoint if best is not found
        model_path = "/kaggle/input/models/maamarmohamed12/ai-model/pytorch/default/1/bmnet_checkpoint.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: No model checkpoint found at {model_path}")
        return

    print(f"Loading weights from {model_path}...")
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 1. Extract the state dict (from Resumable Checkpoint or Raw state_dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # 2. Correctly strip 'module.' prefix (from DataParallel training)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    # 3. Robust Loading with Error Tracking
    model_state = model.state_dict()
    
    # FIX: MNASNet version check bug bypass
    # Some torch versions expect a '_metadata' key with version info or they crash
    input_state_dict = {}
    matched_keys_count = 0
    for name, param in new_state_dict.items():
        if name in model_state:
            if model_state[name].shape == param.shape:
                # Direct copying into current model state to bypass version checks
                model_state[name].copy_(param)
                matched_keys_count += 1

    # Load the updated state_dict back into the model
    model.load_state_dict(model_state)
    print(f"      ✅ Weights loaded ({matched_keys_count}/{len(model_state)} parameters matched via bypass).")
    
    model.eval()

    metrics_names = [
        'Ankle', 'Arm-L', 'Bicep', 'Calf', 'Chest', 
        'Forearm', 'H2H (Height)', 'Hip', 'Leg-L', 
        'Shoulder-B', 'S-to-C', 'Thigh', 'Waist', 'Wrist'
    ]

    # 3. Evaluation Loop per Split
    for split in splits:
        print(f"\n" + "-"*20)
        print(f"🔍 Evaluating on {split}...")
        dataset = BodyMDataset(kaggle_base, split=split)
        if len(dataset) == 0:
            print(f"⚠️ Warning: Split {split} is empty or not found.")
            continue
            
        print(f"Dataset columns found: {list(dataset.df.columns)[:10]}...")
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
        
        split_total_mae = 0
        split_num_samples = 0
        per_metric_mae = np.zeros(14)
        per_metric_counts = np.zeros(14)
        
        # Diagnostic: Check first sample
        test_img, test_target = dataset[0]
        print(f"First Sample Info:")
        print(f"   - Input Shape: {test_img.shape}")
        print(f"   - Silh Mean: {test_img[0].mean():.4f} (Should be > 0 if images loaded)")
        print(f"   - Height Channel Val: {test_img[1, 0, 0]:.4f}")
        print(f"   - Weight Channel Val: {test_img[2, 0, 0]:.4f}")

        with torch.no_grad():
            for i, (images, measurements) in enumerate(loader):
                images = images.to(device)
                measurements = measurements.to(device)
                preds = model(images)
                
                if i == 0:
                    print(f"First Batch Raw Comparisons:")
                    print(f"   - Sample 0 Preds:   {preds[0][:5].cpu().numpy()}")
                    print(f"   - Sample 0 Targets: {measurements[0][:5].cpu().numpy()}")
                
                # Mask out zero targets (missing data)
                mask = (measurements > 0).float()
                error = torch.abs(preds - measurements) * mask
                
                split_total_mae += error.sum().item()
                split_num_samples += mask.sum().item()
                
                per_metric_mae += error.sum(dim=0).cpu().numpy()
                per_metric_counts += mask.sum(dim=0).cpu().numpy()
        
        avg_mae = split_total_mae / max(1, split_num_samples)
        results[split] = avg_mae
        print(f"\n✅ {split} MAE: {avg_mae:.4f} cm")
        
        print(f"Metric Breakdown (MAE in cm):")
        for idx, name in enumerate(metrics_names):
            if per_metric_counts[idx] > 0:
                m_mae = per_metric_mae[idx] / per_metric_counts[idx]
                print(f"   {name.ljust(12)}: {m_mae:.2f}")

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
