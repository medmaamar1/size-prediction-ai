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
        print(f" Error: No model checkpoint found at {model_path}")
        return None

    print(f"Loading weights from {model_path}...")
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the checkpoint is a dict containing 'model_state_dict' or the state_dict itself
    if isinstance(checkpoint, dict) and ('model_state_dict' in checkpoint or 'state_dict' in checkpoint):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
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
            else:
                print(f" Shape mismatch for {name}: {model_state[name].shape} vs {param.shape}")
    
    model.load_state_dict(model_state)
    print(f"       Weights loaded ({matched_keys} parameters matched).")
    model.eval()
    return model

def evaluate():
    print(" --- STARTING MODEL EVALUATION --- ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Splits
    kaggle_base = "/kaggle/input/datasets/maamarmohamed12/bodym-dataset/bodym"
    if not os.path.exists(kaggle_base):
        print(f"Checking for dataset at: {kaggle_base}")
        # Local fallback context
        kaggle_base = "bodym" 
        if not os.path.exists(kaggle_base):
            print(f" Error: Dataset not found at {kaggle_base} or fallback.")
            return

    splits = ['train']
    model_paths = [
        "/kaggle/input/models/maamarmohamed12/get-size/other/default/1/bmnet_best.pth",
        "/kaggle/input/models/maamarmohamed12/get-size/other/default/1/bmnet_checkpoint.pth"
    ]
    
    all_model_results = {}

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n" + "="*50)
        print(f"🏆 EVALUATING MODEL: {model_name}")
        print(f"📂 Path: {model_path}")
        print("="*50)
        
        try:
            # We must re-instantiate the model to clear state between evaluations
            model = load_bm_model(model_path, device)
            if model is None:
                print(f"⚠️ Model load returned None for {model_path}")
                continue
        except Exception as e:
            print(f"❌ CRITICAL ERROR loading {model_name}: {str(e)}")
            continue
            
        results = {}

        # 3. Evaluation Loop per Split
        for split in splits:
            print(f"\n🔍 Evaluating {model_name} on {split}...")
            try:
                # Reload dataset per model evaluation to be safe
                dataset = BodyMDataset(kaggle_base, split=split)
                
                # Limit train split evaluation to 1000 samples for speed if it's the main dataset
                if split == 'train' and len(dataset) > 1000:
                    print(f"   Note: Subsampling 1000 samples from train for comparison.")
                    # Workaround for Subset indexing bug with DataLoaders in older PyTorch versions
                    # We create a new dataset with just a chunk to avoid TypeErrors with '.iloc'
                    indices = torch.randperm(len(dataset))[:1000].tolist()
                    dataset.df = dataset.df.iloc[indices].reset_index(drop=True)

                if len(dataset) == 0:
                    print(f"⚠️ Warning: Split {split} is empty or not found.")
                    continue
                    
                loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
                
                split_total_mae = 0
                split_num_samples = 0
                
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
                
                if split_num_samples > 0:
                    avg_mae = split_total_mae / split_num_samples
                    results[split] = avg_mae
                    print(f" {split} MAE: {avg_mae:.4f} cm")
                else:
                    print(f" No valid samples in {split}")
            except Exception as e:
                print(f" Error during split {split} evaluation: {str(e)}")
            
        all_model_results[model_name] = results

    # 4. Final Comparison Summary
    print("\n" + "" * 20)
    print("       FINAL COMPARISON SUMMARY")
    print("" * 20)
    
    for model_name, results in all_model_results.items():
        if results:
            overall_mae = sum(results.values()) / len(results)
            print(f"\nModel: {model_name}")
            print(f"Overall MAE: {overall_mae:.4f} cm")
            for s, m in results.items():
                print(f"   - {s}: {m:.4f} cm")
    
    print("\n" + "="*40)

if __name__ == "__main__":
    evaluate()
