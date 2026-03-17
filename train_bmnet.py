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
    kaggle_base = '/kaggle/input/datasets/maamarmohamed12/bodym-dataset/bodym'
    
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
    
    # 4. Checkpoint & Resume Logic
    output_dir = "/kaggle/working/models"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "bmnet_checkpoint.pth")
    best_val_loss = float('inf')
    start_epoch = 0
    patience = 3
    patience_counter = 0

    if os.path.exists(checkpoint_path):
        print(f"--- Resuming training from checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"--- Resumed from Epoch {start_epoch} with Best Loss {best_val_loss:.4f} ---")
    else:
        print("--- Starting fresh training (no checkpoint found) ---")

    print(f"Starting training on {device}...")
    
    for epoch in range(start_epoch, num_epochs):
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
            betas = torch.zeros(current_batch_size, 10, dtype=torch.float32, device=device, requires_grad=True)
            with torch.no_grad():
                betas.normal_(mean=0, std=0.01)
                
            optimizer_abs = optim.Adam([betas], lr=abs_eta)
            
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
                
            # --- Main Training Step ---
            preds_real = model(images)
            loss_real = criterion(preds_real, targets)
            
            with torch.no_grad():
                combined_sil, gt_measurements, metadata = smpl_gen.generate_batch(betas.detach())
                h_channel = metadata[:, 0].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
                w_channel = metadata[:, 1].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
                synth_inputs = torch.cat([combined_sil, h_channel, w_channel], dim=1)
                
            preds_synth = model(synth_inputs)
            loss_synth = criterion(preds_synth, gt_measurements)
            
            total_loss = (loss_real + loss_synth) / accumulation_steps
            total_loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(batches):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += (loss_real + loss_synth).item()
            
            if batch_idx % (accumulation_steps * 5) == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx} | "
                      f"Loss_Real: {loss_real.item():.4f} | Loss_Synth: {loss_synth.item():.4f}")
            
            torch.cuda.empty_cache()
            if not train_loader: break
                
        # --- Validation & Early Stopping ---
        avg_epoch_loss = epoch_loss / max(1, len(batches))
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_epoch_loss:.4f}")
        
        # Save Resumable Checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint_state, checkpoint_path)
        
        # Best model saving + Early Stopping logic
        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "bmnet_best.pth"))
            print(f"✅ New Best Model! Loss: {best_val_loss:.4f} | Checkpoint saved.")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"🛑 EARLY STOPPING triggered at Epoch {epoch+1}.")
                break

if __name__ == "__main__":
    train_bmnet()
