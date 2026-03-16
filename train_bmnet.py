import compat_patch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from network import BMNet
from dataset import BodyMDataset
from smpl_generator import SMPLDataGenerator
import numpy as np

def train_bmnet():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 150 
    target_batch_size = 22 # Paper's batch size
    batch_size = 8 # Physical batch size to fit in VRAM
    learning_rate = 1e-3
    abs_iterations = 5
    abs_eta = 0.1
    
    # 2. Models & Optimization
    model = BMNet().to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        batch_size = 4 # 4 per GPU (Total 8)
    
    # Gradient Accumulation setup to reach paper's batch size of 22
    accumulation_steps = max(1, target_batch_size // batch_size)
    print(f"Effective Batch Size: {batch_size * accumulation_steps} (using {accumulation_steps} accumulation steps)")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    
    # 3. Data Loaders (Kaggle Integration)
    kaggle_base = '/kaggle/input/datasets/maamarmohamed/bodym-dataset/bodym'
    
    if os.path.exists(kaggle_base):
        train_dataset = BodyMDataset(base_dir=kaggle_base, split='train')
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset   = BodyMDataset(base_dir=kaggle_base, split='testA')
        val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"Loaded BodyM Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    else:
        print(f"Warning: {kaggle_base} not found. Skipping real dataloader for local test.")
        train_loader = []
        val_loader = []
    
    # Initialize SMPL Generator for ABS
    smpl_gen = SMPLDataGenerator()
    
    print(f"Starting training on {device}...")
    
    best_val_loss = float('inf')
    output_dir = "/kaggle/working/models"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # If no dataloader (local test), simulate a batch
        if not train_loader:
            print("Running a simulated dummy batch locally for ABS verification...")
            images = torch.zeros(batch_size, 3, 640, 960, device=device)
            targets = torch.zeros(batch_size, 14, device=device)
            batches = [(images, targets)]
        else:
            batches = train_loader

        for batch_idx, (images, targets) in enumerate(batches):
            images, targets = images.to(device), targets.to(device)
            current_batch_size = images.shape[0]
            
            # --- ABS: Adversarial Body Simulator ---
            # Paper: initialize betas around zero, optimize with Adam to maximize loss
            betas = torch.zeros(current_batch_size, 10, dtype=torch.float32, device=device, requires_grad=True)
            with torch.no_grad():
                betas.normal_(mean=0, std=0.01)
                
            optimizer_abs = optim.Adam([betas], lr=abs_eta)
            
            # Freeze model context during ABS gradient ascent
            for param in model.parameters():
                param.requires_grad = False
                
            for i in range(abs_iterations):
                optimizer_abs.zero_grad()
                combined_sil, gt_measurements, metadata = smpl_gen.generate_batch(betas)
                
                h_channel = metadata[:, 0].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
                w_channel = metadata[:, 1].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
                synth_inputs = torch.cat([combined_sil, h_channel, w_channel], dim=1)
                
                preds = model(synth_inputs)
                loss_abs = criterion(preds, gt_measurements)
                
                (-loss_abs).backward(retain_graph=True)
                optimizer_abs.step()
                
                with torch.no_grad():
                    betas.clamp_(-3.0, 3.0)
            
            for param in model.parameters():
                param.requires_grad = True
                
            # --- BMNet Main Training Step (with Gradient Accumulation) ---
            # 1. Forward pass on REAL data
            preds_real = model(images)
            loss_real = criterion(preds_real, targets)
            
            # 2. Forward pass on ADVERSARIAL SYNTHETIC data
            with torch.no_grad():
                combined_sil, gt_measurements, metadata = smpl_gen.generate_batch(betas.detach())
                h_channel = metadata[:, 0].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
                w_channel = metadata[:, 1].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
                synth_inputs = torch.cat([combined_sil, h_channel, w_channel], dim=1)
                
            preds_synth = model(synth_inputs)
            loss_synth = criterion(preds_synth, gt_measurements)
            
            # 3. Accumulated Loss
            total_loss = (loss_real + loss_synth) / accumulation_steps
            total_loss.backward()
            
            # 4. Optimizer Step (Every 'accumulation_steps' batches)
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(batches):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += (loss_real + loss_synth).item()
            
            if batch_idx % (accumulation_steps * 5) == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx} | "
                      f"Loss_Real: {loss_real.item():.4f} | Loss_Synth: {loss_synth.item():.4f} | "
                      f"Effective Batch Error: {(loss_real + loss_synth).item():.4f}")
            
            torch.cuda.empty_cache()
            
            if not train_loader: 
                break # Just one simulated batch locally
                
        # --- Validation & Saving ---
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss / max(1, len(batches)):.4f}")
        
        # Save model checkpoints
        torch.save(model.state_dict(), os.path.join(output_dir, "bmnet_latest.pth"))
        
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "bmnet_best.pth"))
            print(f"Saved new best model with loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_bmnet()
