import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network import BMNet
from dataset import BodyMDataset
import numpy as np

def load_bm_model(model_path, device):
    model = BMNet().to(device)
    if not os.path.exists(model_path):
        print(f"❌ Error: No model checkpoint found at {model_path}")
        return None

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
    return model

def evaluate():
    print("🧪 --- STARTING MODEL EVALUATION --- 🧪")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Splits
    kaggle_base = "/kaggle/input/datasets/maamarmohamed12/bodym-dataset/bodym"
    if not os.path.exists(kaggle_base):
        # Local fallback context
        kaggle_base = "bodym" 
        if not os.path.exists(kaggle_base):
            print(f"❌ Error: Dataset not found at {kaggle_base}")
            return

    splits = ['testA', 'testB']
    model_paths = [
        "/kaggle/input/models/maamarmohamed12/get-size/other/default/1/bmnet_best.pth",
        "/kaggle/input/models/maamarmohamed12/get-size/other/default/1/bmnet_checkpoint.pth"
    ]
    
    all_model_results = {}

    metrics_names = [
        'Ankle', 'Arm-L', 'Bicep', 'Calf', 'Chest', 
        'Forearm', 'H2H (Height)', 'Hip', 'Leg-L', 
        'Shoulder-B', 'S-to-C', 'Thigh', 'Waist', 'Wrist'
    ]

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n" + "="*50)
        print(f"🏆 EVALUATING MODEL: {model_name}")
        print("="*50)
        
        model = load_bm_model(model_path, device)
        if model is None:
            continue
            
        results = {}

        # 3. Evaluation Loop per Split
        for split in splits:
            print(f"\n🔍 Evaluating {model_name} on {split}...")
            dataset = BodyMDataset(kaggle_base, split=split)
            if len(dataset) == 0:
                print(f"⚠️ Warning: Split {split} is empty or not found.")
                continue
                
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
            
            split_total_mae = 0
            split_num_samples = 0
            per_metric_mae = np.zeros(14)
            per_metric_counts = np.zeros(14)
            
            with torch.no_grad():
                for i, (images, measurements) in enumerate(loader):
                    images = images.to(device)
                    measurements = measurements.to(device)
                    preds = model(images)
                    
                    # Mask out zero targets (missing data)
                    mask = (measurements > 0).float()
                    error = torch.abs(preds - measurements) * mask
                    
                    split_total_mae += error.sum().item()
                    split_num_samples += mask.sum().item()
                    
                    per_metric_mae += error.sum(dim=0).cpu().numpy()
                    per_metric_counts += mask.sum(dim=0).cpu().numpy()
            
            avg_mae = split_total_mae / max(1, split_num_samples)
            results[split] = avg_mae
            print(f"✅ {split} MAE: {avg_mae:.4f} cm")
            
        all_model_results[model_name] = results

    # 4. Final Comparison Summary
    print("\n" + "📊" * 20)
    print("       FINAL COMPARISON SUMMARY")
    print("📊" * 20)
    
    for model_name, results in all_model_results.items():
        if results:
            overall_mae = sum(results.values()) / len(results)
            print(f"\nModel: {model_name}")
            print(f"Overall MAE: {overall_mae:.4f} cm")
            for s, m in results.items():
                print(f"   - {s}: {m:.4f} cm")
        else:
            print(f"\nModel: {model_name} - No results available.")
    
    print("\n" + "="*40)
