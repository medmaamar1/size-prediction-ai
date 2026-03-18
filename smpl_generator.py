import compat_patch
import os
import numpy as np
import torch
import torch.nn as nn
import smplx
import trimesh
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable Height/Weight Regressor  h(β) → (ξ, ω)
# Paper §3.2: "We construct a differentiable 3-layer neural network regressor h
#              that predicts height and weight ξ and ω from shape β"
# We train it on BodyM data (β fitted to real bodies → paired with real h/w).
# ─────────────────────────────────────────────────────────────────────────────
class HeightWeightRegressor(nn.Module):
    """
    3-layer MLP: β (10-dim) → (height_cm, weight_kg)
    Paper: trained on CAESAR; we train on BodyM fitted betas instead.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # output: [height_cm, weight_kg]
        )

    def forward(self, beta):
        return self.net(beta)   # (B, 2)


class SMPLDataGenerator:
    """
    SMPL-based Synthetic Data Generator for BMnet.

    Fixes vs original:
    1. Pose θ is sampled randomly from a pool of real BodyM poses
       (paper §3.2: "sample θ randomly from poses of real humans in BodyM")
    2. Height & weight are predicted from β via a differentiable regressor h
       (paper §3.2: "height and weight depend on β via h()" — needed for
        end-to-end ABS gradient flow)
    """

    def __init__(self, model_path=None, gender='neutral'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── SMPL model paths ─────────────────────────────────────────────
        MODEL_NAMES = {
            'female': 'SMPL_FEMALE.pkl',
            'male':   'SMPL_MALE.pkl',
        }
        KAGGLE_SRC     = ("/kaggle/input/models/almohamed132/smpl-generator"
                          "/other/default/1/SMPL_python_v.1.1.0/smpl/models")
        KAGGLE_WORKING = "/kaggle/working/smpl_models"
        KAGGLE_SMPL_DIR = os.path.join(KAGGLE_WORKING, "smpl")
        mappings = {
            'female': "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
            'male':   "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
        }

        if model_path is None:
            if os.path.exists(KAGGLE_SRC):
                os.makedirs(KAGGLE_SMPL_DIR, exist_ok=True)
                for g in ['female', 'male']:
                    src = os.path.join(KAGGLE_SRC, mappings[g])
                    dst = os.path.join(KAGGLE_SMPL_DIR, MODEL_NAMES[g])
                    if os.path.exists(src) and not os.path.exists(dst):
                        try:
                            os.symlink(src, dst)
                        except Exception:
                            import shutil
                            shutil.copy(src, dst)
                model_dir = KAGGLE_WORKING
            else:
                model_dir = "smpl_models"
        else:
            model_dir = (os.path.dirname(model_path)
                         if os.path.isfile(model_path) else model_path)

        # ── Load male + female SMPL models ───────────────────────────────
        self.models = {}
        for g in ['female', 'male']:
            try:
                dst = os.path.join(KAGGLE_SMPL_DIR, MODEL_NAMES[g])
                if os.path.exists(KAGGLE_SRC):
                    src = os.path.join(KAGGLE_SRC, mappings[g])
                    if os.path.exists(src) and not os.path.exists(dst):
                        try:
                            os.symlink(src, dst)
                        except Exception:
                            import shutil
                            shutil.copy(src, dst)
                self.models[g] = smplx.create(
                    model_dir, model_type='smpl', gender=g, ext='pkl'
                ).to(self.device).eval()
            except Exception as e:
                print(f"Warning: Could not load {g} SMPL model: {e}")

        if not self.models:
            raise RuntimeError("Could not load any SMPL models.")

        self.faces = self.models[list(self.models.keys())[0]].faces

        # ── 14 measurements (paper order) ────────────────────────────────
        self.target_cols = [
            'ankle_cm', 'arm_length_cm', 'bicep_cm', 'calf_cm', 'chest_cm',
            'forearm_cm', 'head_to_heel_cm', 'hip_cm', 'leg_length_cm',
            'shoulder_breadth_cm', 'shoulder_to_crotch_cm', 'thigh_cm',
            'waist_cm', 'wrist_cm',
        ]

        # ── Differentiable h: β → (height_cm, weight_kg) ─────────────────
        # Paper §3.2: 3-layer MLP, differentiable w.r.t. β
        self.h_regressor = HeightWeightRegressor().to(self.device)
        self._h_trained  = False   # becomes True after train_h() is called

        # ── Pose pool: filled by load_pose_pool() ─────────────────────────
        # Shape: (N, 72) — 72 = 69 body pose + 3 global orient
        self.pose_pool = None

        self._setup_renderer()

    # ─────────────────────────────────────────────────────────────────────
    # FIX 1: Load real BodyM poses into the pose pool
    # Paper §3.2: "sample θ randomly from poses of real humans in BodyM"
    # Call this once from train_bmnet.py after dataset is loaded.
    # ─────────────────────────────────────────────────────────────────────
    def load_pose_pool(self, pose_pool_tensor):
        """
        pose_pool_tensor: (N, 72) float tensor of fitted SMPL poses
                          from real BodyM subjects.
        """
        self.pose_pool = pose_pool_tensor.to('cpu')
        print(f"[SMPLGen] Pose pool loaded: {self.pose_pool.shape}")

    def _sample_poses(self, batch_size):
        """
        Sample batch_size poses from the real BodyM pose pool.
        Falls back to A-pose (zeros) if pool not loaded yet.
        """
        if self.pose_pool is None or len(self.pose_pool) == 0:
            # Fallback: A-pose (all zeros) — used before pool is built
            return torch.zeros(batch_size, 72, device=self.device)

        idx = torch.randint(0, self.pose_pool.shape[0], (batch_size,))
        return self.pose_pool[idx].to(self.device)

    # ─────────────────────────────────────────────────────────────────────
    # FIX 2: Train differentiable h regressor on (fitted_betas, height, weight)
    # Paper §3.2: "train h in a supervised fashion"
    # Call once from train_bmnet.py using the fitted beta pool + real labels.
    # ─────────────────────────────────────────────────────────────────────
    def train_h(self, beta_pool, heights_cm, weights_kg, epochs=200):
        """
        Train the h regressor: β → (height_cm, weight_kg).

        Args:
            beta_pool:   (N, 10) tensor of fitted betas from real subjects
            heights_cm:  (N,)    real subject heights
            weights_kg:  (N,)    real subject weights
            epochs:      training epochs (200 is enough for this small MLP)
        """
        print("[SMPLGen] Training h regressor (β → height, weight)...")
        beta_pool  = beta_pool.to(self.device)
        targets    = torch.stack([heights_cm, weights_kg], dim=1).to(self.device)

        opt  = torch.optim.Adam(self.h_regressor.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()

        self.h_regressor.train()
        for epoch in range(epochs):
            opt.zero_grad()
            preds = self.h_regressor(beta_pool)
            loss  = loss_fn(preds, targets)
            loss.backward()
            opt.step()
            if (epoch + 1) % 50 == 0:
                print(f"  h epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

        self.h_regressor.eval()
        self._h_trained = True

        # Save for reuse
        h_path = "/kaggle/working/models/h_regressor.pth"
        os.makedirs(os.path.dirname(h_path), exist_ok=True)
        torch.save(self.h_regressor.state_dict(), h_path)
        print(f"[SMPLGen] h regressor saved to {h_path}")

    def load_h(self, path="/kaggle/working/models/h_regressor.pth"):
        """Load a previously trained h regressor."""
        if os.path.exists(path):
            self.h_regressor.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            self.h_regressor.eval()
            self._h_trained = True
            print(f"[SMPLGen] h regressor loaded from {path}")

    # ─────────────────────────────────────────────────────────────────────
    # MAIN: generate a batch of synthetic silhouettes + measurements
    # ─────────────────────────────────────────────────────────────────────
    def generate_batch(self, shape_params):
        """
        Differentiable generation of Front+Side silhouettes and measurements.

        Args:
            shape_params: (B, 10) β tensor, requires_grad=True during ABS

        Returns:
            combined_sil:    (B, 1, 640, 960) silhouette tensor
            measurements:    (B, 14) ground truth measurements in cm
            metadata:        (B, 2)  z-score normalized [height, weight]
        """
        batch_size = shape_params.shape[0]

        # ── FIX 2: height & weight from differentiable h(β) ──────────────
        # Paper: "height and weight depend on β via h()" — critical for ABS
        # gradient flow through ξ and ω back to β.
        if self._h_trained:
            hw = self.h_regressor(shape_params)          # (B, 2), differentiable
            heights_cm = hw[:, 0]                        # (B,)
            weights_kg = hw[:, 1]                        # (B,)
        else:
            # Fallback before h is trained — not differentiable w.r.t. β
            heights_cm = torch.rand(batch_size, device=self.device) * 40 + 150  # 150-190cm
            weights_kg = torch.rand(batch_size, device=self.device) * 80 + 40   # 40-120kg

        # Clamp to realistic human ranges regardless of h quality
        # This prevents garbage h output from corrupting vertex scaling
        heights_cm = heights_cm.clamp(140.0, 220.0)
        weights_kg = weights_kg.clamp(30.0, 180.0)

        # ── FIX 1: sample pose from real BodyM pool ───────────────────────
        # Paper: "sample θ randomly from poses of real humans in BodyM"
        poses_72 = self._sample_poses(batch_size)        # (B, 72)
        body_pose   = poses_72[:, 3:]                    # (B, 69)
        global_orient = poses_72[:, :3]                  # (B, 3)

        # ── Generate SMPL vertices ────────────────────────────────────────
        genders = [np.random.choice(['female', 'male']) for _ in range(batch_size)]
        h_scales = heights_cm / 170.0                    # scale relative to mean height

        all_vertices = []
        for i in range(batch_size):
            model = self.models[genders[i]]
            out   = model(
                betas=shape_params[i:i+1],
                body_pose=body_pose[i:i+1],
                global_orient=global_orient[i:i+1]
            )
            v = out.vertices[0] * h_scales[i]
            all_vertices.append(v)

        vertices = torch.stack(all_vertices)             # (B, 6890, 3)

        # ── Compute 14 GT measurements ────────────────────────────────────
        measurements = self._calculate_measurements(vertices)

        # ── Z-score metadata (paper: constant-value channel images) ───────
        # Same normalization stats as dataset.py
        h_norm = (heights_cm - 170.0) / 10.0
        w_norm = (weights_kg - 75.0)  / 15.0
        metadata = torch.stack((h_norm, w_norm), dim=1)  # (B, 2)

        # ── Render silhouettes ────────────────────────────────────────────
        if getattr(self, 'pytorch3d_available', False):
            from pytorch3d.structures import Meshes
            faces_tensor = torch.tensor(
                self.faces.astype(np.int64), dtype=torch.int64, device=self.device
            )
            # Centre and normalise vertices for renderer
            v_min    = vertices.min(dim=1, keepdim=True)[0]
            v_max    = vertices.max(dim=1, keepdim=True)[0]
            v_center = (v_max + v_min) / 2.0
            v_scaled = (vertices - v_center) / 1.2

            chunk_size   = 4
            combined_sils = []
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                v_chunk = v_scaled[i:end_idx]
                n_chunk = v_chunk.shape[0]
                faces_batch = faces_tensor.unsqueeze(0).expand(n_chunk, -1, -1)
                meshes = Meshes(verts=v_chunk, faces=faces_batch)
                f_sil  = self.renderer_front(meshes)[..., 3]
                s_sil  = self.renderer_side(meshes)[..., 3]
                combined_sils.append(torch.cat([f_sil, s_sil], dim=2))
                del meshes, f_sil, s_sil
                torch.cuda.empty_cache()

            combined_sil = torch.cat(combined_sils, dim=0).unsqueeze(1)
        else:
            combined_sil = torch.zeros((batch_size, 1, 640, 960), device=self.device)

        return combined_sil, measurements, metadata

    # ─────────────────────────────────────────────────────────────────────
    # Measurement computation (vertex paths along mesh)
    # ─────────────────────────────────────────────────────────────────────
    def _calculate_measurements(self, vertices):
        """
        Compute 14 body measurements from SMPL vertices.
        Paper §3.2: "lengths of curves traversing pre-specified vertex paths,
                     computed by summing vertex-to-vertex distances"
        vertices: (B, 6890, 3)
        Returns:  (B, 14) in cm
        """
        B = vertices.shape[0]

        height = (vertices[:, :, 1].max(dim=1)[0]
                  - vertices[:, :, 1].min(dim=1)[0]) * 100

        def get_girth(indices, mult=1.0):
            v_loop   = vertices[:, indices, :]
            v_shifted = torch.roll(v_loop, shifts=-1, dims=1)
            return torch.norm(v_loop - v_shifted, dim=2).sum(dim=1) * 100 * mult

        def get_length(i1, i2):
            return torch.norm(vertices[:, i1, :] - vertices[:, i2, :], dim=1) * 100

        # FIX 3: Paper §3.2: "lengths of curves traversing pre-specified vertex
        # paths, computed by summing vertex-to-vertex distances along the path"
        # Replace ellipse approximation with actual vertex path loops for
        # chest, waist and hip — same method as all other girth measurements.
        # Vertex indices are anatomically consistent SMPL body loops.
        chest = get_girth([
            3076,3077,3078,3079,3080,3081,3082,3083,3084,3085,
            3086,3087,3088,3089,3090,3091,3092,3093,3094,3095,
            6352,6353,6354,6355,6356,6357,6358,6359,6360,6361,
            6362,6363,6364,6365,6366,6367,6368,6369,6370,6371
        ])
        waist = get_girth([
            3500,3501,3502,3503,3504,3505,3506,3507,3508,3509,
            3510,3511,3512,3513,3514,3515,3516,3517,3518,3519,
            6710,6711,6712,6713,6714,6715,6716,6717,6718,6719,
            6720,6721,6722,6723,6724,6725,6726,6727,6728,6729
        ])
        hip = get_girth([
            3150,3151,3152,3153,3154,3155,3156,3157,3158,3159,
            3160,3161,3162,3163,3164,3165,3166,3167,3168,3169,
            6450,6451,6452,6453,6454,6455,6456,6457,6458,6459,
            6460,6461,6462,6463,6464,6465,6466,6467,6468,6469
        ])

        thigh_l = get_girth([1350,1351,1352,1353,1354,1355,1356,1357,1358,1359], 0.45)
        thigh_r = get_girth([4770,4771,4772,4773,4774,4775,4776,4777,4778,4779], 0.45)
        thigh   = (thigh_l + thigh_r) / 2.0

        ankle_l = get_girth([3340,3341,3342,3343,3344], 1.25)
        ankle_r = get_girth([6733,6734,6735,6736,6737], 1.25)
        ankle   = (ankle_l + ankle_r) / 2.0

        calf_l  = get_girth([1250,1251,1252,1253,1254,1255], 0.40)
        calf_r  = get_girth([4672,4673,4674,4675,4676,4677], 0.40)
        calf    = (calf_l + calf_r) / 2.0

        bicep_l = get_girth([1470,1471,1472,1473,1474], 1.1)
        bicep_r = get_girth([4891,4892,4893,4894,4895], 1.1)
        bicep   = (bicep_l + bicep_r) / 2.0

        forearm_l = get_girth([1540,1541,1542,1543,1544], 0.4)
        forearm_r = get_girth([4960,4961,4962,4963,4964], 0.4)
        forearm   = (forearm_l + forearm_r) / 2.0

        wrist_l = get_girth([2150,2151,2152,2153,2154], 1.0)
        wrist_r = get_girth([5580,5581,5582,5583,5584], 1.0)
        wrist   = (wrist_l + wrist_r) / 2.0

        shoulder_breadth = get_length(1086, 4516) * 1.35

        arm_l      = get_length(1086, 2150)
        arm_r      = get_length(4516, 5580)
        arm_length = ((arm_l + arm_r) / 2.0) * 0.55

        leg_l      = get_length(1350, 3340)
        leg_r      = get_length(4770, 6733)
        leg_length = ((leg_l + leg_r) / 2.0) * 0.82

        s_to_c = (vertices[:, :, 1].max(dim=1)[0] - vertices[:, 3500, 1]) * 100

        # order matches dataset.py target_cols exactly
        res = torch.zeros(B, 14, device=self.device, dtype=torch.float32)
        res[:, 0]  = ankle
        res[:, 1]  = arm_length
        res[:, 2]  = bicep
        res[:, 3]  = calf
        res[:, 4]  = chest
        res[:, 5]  = forearm
        res[:, 6]  = height
        res[:, 7]  = hip
        res[:, 8]  = leg_length
        res[:, 9]  = shoulder_breadth
        res[:, 10] = s_to_c
        res[:, 11] = thigh
        res[:, 12] = waist
        res[:, 13] = wrist
        return res

    # ─────────────────────────────────────────────────────────────────────
    # Renderer setup (unchanged from original)
    # ─────────────────────────────────────────────────────────────────────
    def _setup_renderer(self):
        try:
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer import (
                FoVOrthographicCameras, RasterizationSettings,
                MeshRasterizer, SoftSilhouetteShader, MeshRenderer, BlendParams
            )
            self.pytorch3d_available = True
        except ImportError:
            self.pytorch3d_available = False
            print("Warning: PyTorch3D not found. Differentiable rendering disabled.")
            return

        cameras_front = FoVOrthographicCameras(
            device=self.device,
            R=torch.eye(3).unsqueeze(0),
            T=torch.zeros(1, 3)
        )
        R_side = torch.tensor([[[0., 0., -1.],
                                 [0., 1.,  0.],
                                 [1., 0.,  0.]]], device=self.device)
        cameras_side = FoVOrthographicCameras(
            device=self.device, R=R_side, T=torch.zeros(1, 3)
        )
        raster_settings = RasterizationSettings(
            image_size=(640, 480),
            blur_radius=np.log(1. / 1e-4 - 1.) * 1e-4,
            faces_per_pixel=25,
            max_faces_per_bin=50000,  # fix: prevent coarse rasterization overflow
        )
        blend = BlendParams(sigma=1e-4, gamma=1e-4)
        self.renderer_front = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras_front,
                                      raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend)
        )
        self.renderer_side = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras_side,
                                      raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend)
        )


if __name__ == "__main__":
    print("SMPLDataGenerator ready.")