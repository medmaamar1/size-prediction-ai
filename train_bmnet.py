import compat_patch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*")

import os
import time
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from network import BMNet
from dataset import BodyMDataset
from smpl_generator import SMPLDataGenerator



# ─────────────────────────────────────────────────────────────────────────────
# HEARTBEAT: prints every 30 min to prevent Kaggle inactivity disconnect
# ─────────────────────────────────────────────────────────────────────────────
def start_heartbeat(interval_seconds=1800):
    def beat():
        while True:
            time.sleep(interval_seconds)
            print(f"[HEARTBEAT] Still training... {time.strftime('%H:%M:%S')}", flush=True)
    t = threading.Thread(target=beat, daemon=True)
    t.start()
    print("✅ Heartbeat started (prints every 30 min to prevent Kaggle timeout)")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: fit SMPL betas to each subject in the 90% training subset.
# Paper §3.2: "initialize β by selecting at random shape parameters that have
# been fitted to real human bodies in the BodyM training set"
# ─────────────────────────────────────────────────────────────────────────────
def build_real_beta_pool(train_subset, smpl_gen, device):
    """
    Fit SMPL beta to every subject. Also extracts heights & weights
    from the dataset metadata channels so we can train h regressor.
    Returns: (N,10) betas, (N,) heights_cm, (N,) weights_kg
    """
    print("--- Building real-body beta pool (SMPL fitting) ---")
    fitted  = []
    heights = []
    weights = []
    crit    = nn.L1Loss()
    n       = len(train_subset)

    for idx in range(n):
        img, gt_meas = train_subset[idx]
        gt_meas = gt_meas.unsqueeze(0).to(device)

        # Recover height/weight from z-scored metadata channels
        # channel 1 = (height - 170) / 10,  channel 2 = (weight - 75) / 15
        h_cm = img[1, 0, 0].item() * 10.0 + 170.0
        w_kg = img[2, 0, 0].item() * 15.0 + 75.0
        heights.append(h_cm)
        weights.append(w_kg)

        beta = torch.zeros(1, 10, device=device, requires_grad=True)
        opt  = optim.Adam([beta], lr=0.05)

        for _ in range(10):   # 10 steps is enough for ABS initialization
            opt.zero_grad()
            _, gt_synth, _ = smpl_gen.generate_batch(beta)
            loss = crit(gt_synth, gt_meas)
            loss.backward()
            opt.step()
            with torch.no_grad():
                beta.clamp_(-3.0, 3.0)

        fitted.append(beta.detach().cpu())
        if (idx + 1) % 100 == 0:
            print(f"  Fitted {idx+1}/{n} subjects")

    pool      = torch.cat(fitted, dim=0)
    heights_t = torch.tensor(heights, dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    print(f"--- beta pool ready: {pool.shape} ---")
    return pool, heights_t, weights_t


def build_pose_pool(n_subjects):
    """
    Build pose pool by adding small perturbations to A-pose.
    Paper: "sample theta randomly from poses of real humans in BodyM"
    Since BodyM subjects are in A-pose, we create diversity via perturbations.
    Returns (N*10, 72) tensor.
    """
    print("--- Building pose pool ---")
    poses = []
    for _ in range(n_subjects * 10):
        pose = torch.zeros(72)
        pose[:3]  = torch.randn(3) * 0.05   # small global orient
        pose[3:]  = torch.randn(69) * 0.1   # small joint perturbations
        poses.append(pose)
    pool = torch.stack(poses)
    print(f"--- Pose pool ready: {pool.shape} ---")
    return pool


def sample_betas_from_pool(pool, batch_size, device):
    """
    Sample betas from pool and add uniform noise within hypercube side=0.5.
    Paper §5.2: "Random sampling is performed uniformly within a hypercube
                 with side length of 0.5 around the real shape parameters"
    So each β_i is perturbed by Uniform(-0.25, +0.25).
    """
    idx   = torch.randint(0, pool.shape[0], (batch_size,))
    betas = pool[idx].clone().to(device)
    # Add uniform noise in [-0.25, 0.25] (hypercube side length = 0.5)
    noise = (torch.rand_like(betas) - 0.5) * 0.5
    betas = (betas + noise).clamp(-3.0, 3.0)
    return betas


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: validation pass on the 10% held-out split
# ─────────────────────────────────────────────────────────────────────────────
def run_validation(model, val_loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            preds  = model(images)
            total += criterion(preds, targets).item() * images.size(0)
            count += images.size(0)
    model.train()
    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: one ABS fine-tuning epoch (fresh synthetic bodies, never repeated)
# Paper §3.2: "Synthetic bodies are not repeated over epochs, so that in
#              10 epochs the network sees roughly 10x more data"
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch_abs(model, n_train, smpl_gen, beta_pool,
                        optimizer, criterion,
                        accumulation_steps, batch_size,
                        abs_iterations, abs_eta,
                        device, epoch, total_epochs,
                        dummy_fallback=False):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    n_batches = max(1, n_train // batch_size) if not dummy_fallback else 1

    for batch_idx in range(n_batches):
        # 1. Init β from real-body pool (paper: fitted to real BodyM subjects)
        if dummy_fallback:
            betas = torch.zeros(batch_size, 10, device=device, requires_grad=True)
        else:
            betas = sample_betas_from_pool(beta_pool, batch_size, device).requires_grad_(True)

        # 2. ABS: gradient ascent on β to maximise BMnet error (freeze weights)
        optimizer_abs = optim.Adam([betas], lr=abs_eta)
        for param in model.parameters():
            param.requires_grad = False

        for _ in range(abs_iterations):
            optimizer_abs.zero_grad()
            combined_sil, gt_meas, metadata = smpl_gen.generate_batch(betas)
            h_ch = metadata[:, 0].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
            w_ch = metadata[:, 1].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
            synth_input = torch.cat([combined_sil, h_ch, w_ch], dim=1)
            preds    = model(synth_input)
            loss_abs = criterion(preds, gt_meas)
            (-loss_abs).backward(retain_graph=True)
            optimizer_abs.step()
            with torch.no_grad():
                betas.clamp_(-3.0, 3.0)

        for param in model.parameters():
            param.requires_grad = True

        # 3. Train BMnet on the adversarial synthetic batch
        with torch.no_grad():
            combined_sil, gt_meas, metadata = smpl_gen.generate_batch(betas.detach())
            h_ch = metadata[:, 0].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
            w_ch = metadata[:, 1].view(-1, 1, 1, 1).expand(-1, -1, 640, 960)
            synth_input = torch.cat([combined_sil, h_ch, w_ch], dim=1)

        preds_synth = model(synth_input)
        loss = criterion(preds_synth, gt_meas) / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == n_batches:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps

        if batch_idx % (accumulation_steps * 5) == 0:
            print(f"  ABS Epoch {epoch+1}/{total_epochs} | "
                  f"Batch {batch_idx}/{n_batches} | "
                  f"Loss: {loss.item() * accumulation_steps:.4f}")

        torch.cuda.empty_cache()
        if dummy_fallback:
            break

    return epoch_loss / max(1, n_batches)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        return 0, float('inf')

    print(f"--- Loading checkpoint: {checkpoint_path} ---")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = (checkpoint['model_state_dict']
                  if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
                  else checkpoint)

    clean = {(k[7:] if k.startswith('module.') else k): v
             for k, v in state_dict.items()}

    model_state = model.state_dict()
    matched = 0
    for name, param in clean.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name].copy_(param)
            matched += 1
    model.load_state_dict(model_state)
    print(f"--- Loaded {matched}/{len(model_state)} parameters ---")

    start_iter    = 0
    best_val_loss = float('inf')
    if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter    = checkpoint.get('iteration', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"--- Resumed from iteration {start_iter:,}, "
                  f"best val {best_val_loss:.4f} ---")
        except Exception:
            print("--- Optimizer state mismatch; using fresh optimizer ---")

    return start_iter, best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def train_bmnet():
    start_heartbeat()  # prevent Kaggle inactivity disconnect
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Paper-exact config ────────────────────────────────────────────────
    # Phase 1 — paper §5: "150k iterations, batch 22, Adam lr=1e-3,
    #            LR reduced at 75% and 88% of training"
    phase1_iterations = 30_000   # practical Kaggle default (2x T4, 12h session)
                                  # paper's full number is 150,000 — resume across sessions to reach it

    # Phase 2 — paper §3.2: "fine-tune for 10 epochs using synthetic examples,
    #            synthetic bodies are not repeated over epochs"
    phase2_epochs     = 10

    # Phase 3 — paper §3.2: "another fine-tuning on real BodyM data"
    # No number given in the paper — we match phase 2 (10 epochs)
    phase3_epochs     = 10

    target_batch_size = 22
    batch_size        = 8
    learning_rate     = 1e-3
    abs_iterations    = 5        # k in the paper
    abs_eta           = 0.1      # η in the paper

    output_dir        = "/kaggle/working/models"
    os.makedirs(output_dir, exist_ok=True)

    kaggle_base       = '/kaggle/input/datasets/maamarmohamed12/bodym-dataset/bodym'
    kaggle_checkpoint = "/kaggle/input/models/maamarmohamed12/get-size/other/default/1/bmnet_best.pth"
    local_checkpoint  = os.path.join(output_dir, "bmnet_checkpoint.pth")

    # ── Model & optimiser ─────────────────────────────────────────────────
    model = BMNet().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model      = nn.DataParallel(model)
        batch_size = 4

    accumulation_steps = max(1, target_batch_size // batch_size)
    print(f"Effective batch size: {batch_size * accumulation_steps} "
          f"({accumulation_steps} accumulation steps)")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    # ── LR scheduler: per-iteration, milestones at 75% and 88% of 150k ───
    milestone1 = int(phase1_iterations * 0.75)   # 112,500
    milestone2 = int(phase1_iterations * 0.88)   # 132,000
    scheduler  = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[milestone1, milestone2], gamma=0.1
    )

    # ── Data: 90/10 split of the training set ────────────────────────────
    # Paper §5: "validation set corresponding to 10% of the training data"
    # This is a split of the TRAIN set — NOT TestA or TestB.
    dummy_fallback = not os.path.exists(kaggle_base)
    if not dummy_fallback:
        full_dataset = BodyMDataset(base_dir=kaggle_base, split='train')
        n_total      = len(full_dataset)                  # 2,287
        n_val        = int(n_total * 0.10)                # ~229
        n_train      = n_total - n_val                    # ~2,058

        train_subset, val_subset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)   # reproducible split
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                  shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_subset,   batch_size=batch_size,
                                  shuffle=False, num_workers=2)

        print(f"Train: {n_train} subjects | Val (10% of train): {n_val} subjects")
    else:
        print(f"WARNING: {kaggle_base} not found — running dummy local test")
        full_dataset = None
        train_subset = None
        train_loader = None
        val_loader   = None
        n_train      = 0

    # ── SMPL generator ────────────────────────────────────────────────────
    smpl_gen = SMPLDataGenerator()

    # ── Load starting checkpoint ──────────────────────────────────────────
    checkpoint_path = (local_checkpoint if os.path.exists(local_checkpoint)
                       else kaggle_checkpoint)
    start_iter, best_val_loss = load_checkpoint(model, optimizer,
                                                checkpoint_path, device)

    # ── Build fitted-β pool + pose pool + train h regressor ─────────────
    beta_pool      = None
    beta_pool_path = os.path.join(output_dir, "beta_pool.pt")
    pose_pool_path = os.path.join(output_dir, "pose_pool.pt")
    h_path         = os.path.join(output_dir, "h_regressor.pth")

    if not dummy_fallback:
        # β pool — fit SMPL to a random subset (500 subjects is enough diversity)
        # Fitting all subjects takes too long and is not necessary
        if os.path.exists(beta_pool_path):
            beta_pool = torch.load(beta_pool_path, map_location='cpu')
            print(f"--- Loaded cached beta pool: {beta_pool.shape} ---")
        else:
            # Sample 500 random subjects from the training subset
            pool_size   = min(500, len(train_subset))
            pool_indices = torch.randperm(len(train_subset))[:pool_size]
            pool_subset  = torch.utils.data.Subset(train_subset, pool_indices)
            print(f"--- Fitting beta pool on {pool_size} subjects (sampled from {len(train_subset)}) ---")
            beta_pool, heights_t, weights_t = build_real_beta_pool(
                pool_subset, smpl_gen, device
            )
            torch.save(beta_pool, beta_pool_path)
            torch.save({'heights': heights_t, 'weights': weights_t},
                       os.path.join(output_dir, "hw_labels.pt"))

        # h regressor: beta -> (height, weight) — differentiable for ABS
        # Paper: "height and weight depend on beta via h()"
        if os.path.exists(h_path):
            smpl_gen.load_h(h_path)
        else:
            hw = torch.load(os.path.join(output_dir, "hw_labels.pt"))
            smpl_gen.train_h(beta_pool.to(device),
                             hw['heights'].to(device),
                             hw['weights'].to(device))

        # pose pool — sample randomly during ABS
        # Paper: "sample theta randomly from poses of real humans in BodyM"
        if os.path.exists(pose_pool_path):
            pose_pool = torch.load(pose_pool_path, map_location='cpu')
            print(f"--- Loaded cached pose pool: {pose_pool.shape} ---")
        else:
            pose_pool = build_pose_pool(n_train)
            torch.save(pose_pool, pose_pool_path)

        smpl_gen.load_pose_pool(pose_pool)

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 1 — Pre-train on real BodyM data for 150k iterations
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print(f"PHASE 1 — Real data pre-training ({phase1_iterations:,} iterations)")
    print("="*60)

    val_every  = 1_000
    save_every = 1_000
    log_every  = 100

    model.train()
    optimizer.zero_grad()

    if dummy_fallback:
        dummy_images  = torch.zeros(batch_size, 3, 640, 960, device=device)
        dummy_targets = torch.zeros(batch_size, 14, device=device)

    iteration = start_iter

    while iteration < phase1_iterations:
        data_iter = (iter([(dummy_images, dummy_targets)])
                     if dummy_fallback else iter(train_loader))

        for images, targets in data_iter:
            if iteration >= phase1_iterations:
                break

            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss  = criterion(preds, targets) / accumulation_steps
            loss.backward()

            if (iteration + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()   # stepped per iteration

            if iteration % log_every == 0:
                print(f"  Iter {iteration:,}/{phase1_iterations:,} | "
                      f"Loss: {loss.item() * accumulation_steps:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # Validate on 10% held-out split
            if iteration % val_every == 0 and val_loader is not None:
                val_loss = run_validation(model, val_loader, criterion, device)
                print(f"  ── Val Loss @ iter {iteration:,}: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save without DataParallel module. prefix
                    state = (model.module.state_dict()
                             if hasattr(model, 'module') else model.state_dict())
                    torch.save(state, os.path.join(output_dir, "bmnet_phase1_best.pth"))
                    print(f"  ✅ New best val loss: {best_val_loss:.4f}")
                model.train()

            if iteration % save_every == 0:
                torch.save({
                    'iteration':            iteration,
                    'phase':                1,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss':        best_val_loss,
                }, local_checkpoint)

            torch.cuda.empty_cache()
            iteration += 1
            if dummy_fallback:
                break

    print(f"Phase 1 complete at iteration {iteration:,}.")

    # Load best phase-1 weights before ABS fine-tuning
    phase1_best = os.path.join(output_dir, "bmnet_phase1_best.pth")
    if os.path.exists(phase1_best):
        load_checkpoint(model, optimizer, phase1_best, device)
        print("--- Loaded best Phase-1 weights for ABS fine-tuning ---")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 2 — ABS fine-tuning: 10 epochs, fresh synthetic bodies each epoch
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PHASE 2 — ABS synthetic fine-tuning (10 epochs)")
    print("="*60)

    optimizer_p2 = optim.Adam(model.parameters(), lr=learning_rate * 0.1)

    for epoch in range(phase2_epochs):
        avg_loss = train_one_epoch_abs(
            model, n_train, smpl_gen, beta_pool,
            optimizer_p2, criterion,
            accumulation_steps, batch_size,
            abs_iterations, abs_eta,
            device, epoch, phase2_epochs,
            dummy_fallback=dummy_fallback
        )

        # Validate on the same 10% split
        val_loss = (run_validation(model, val_loader, criterion, device)
                    if val_loader is not None else avg_loss)

        print(f"ABS Epoch [{epoch+1}/{phase2_epochs}] "
              f"Synth Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        torch.save({
            'iteration':            epoch,
            'phase':                2,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer_p2.state_dict(),
            'best_val_loss':        best_val_loss,
        }, local_checkpoint)

    state = (model.module.state_dict()
             if hasattr(model, 'module') else model.state_dict())
    torch.save(state, os.path.join(output_dir, "bmnet_phase2.pth"))
    print("--- Phase 2 complete ---")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 3 — Final fine-tune on real data (close synthetic→real gap)
    # Paper gives no specific number — we use 10 epochs to match phase 2
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print(f"PHASE 3 — Real data fine-tune to close domain gap ({phase3_epochs} epochs)")
    print("="*60)

    optimizer_p3  = optim.Adam(model.parameters(), lr=learning_rate * 0.01)
    best_val_loss = float('inf')

    for epoch in range(phase3_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer_p3.zero_grad()

        loader_p3 = (iter([(dummy_images, dummy_targets)])
                     if dummy_fallback else iter(train_loader))

        for batch_idx, (images, targets) in enumerate(loader_p3):
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss  = criterion(preds, targets) / accumulation_steps
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer_p3.step()
                optimizer_p3.zero_grad()
            epoch_loss += loss.item() * accumulation_steps
            torch.cuda.empty_cache()
            if dummy_fallback:
                break

        avg_loss = epoch_loss / max(1, batch_idx + 1)
        val_loss = (run_validation(model, val_loader, criterion, device)
                    if val_loader is not None else avg_loss)

        print(f"Phase3 Epoch [{epoch+1}/{phase3_epochs}] "
              f"Train: {avg_loss:.4f} | Val: {val_loss:.4f}")

        torch.save({
            'iteration':            epoch,
            'phase':                3,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer_p3.state_dict(),
            'best_val_loss':        best_val_loss,
        }, local_checkpoint)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = (model.module.state_dict()
                     if hasattr(model, 'module') else model.state_dict())
            torch.save(state, os.path.join(output_dir, "bmnet_best.pth"))
            print(f"  ✅ New best (Phase 3) val loss: {best_val_loss:.4f}")

    print("\n🏁 Training complete.")
    print(f"   Final model: {os.path.join(output_dir, 'bmnet_best.pth')}")


if __name__ == "__main__":
    train_bmnet()