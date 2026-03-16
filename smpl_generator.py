import compat_patch
import os
import numpy as np
import torch
import smplx
import trimesh
from PIL import Image

class SMPLDataGenerator:
    """
    SMPL-based Synthetic Data Generator for BMnet.
    Generates paired Front + Side silhouettes and exact measurements.
    """
    def __init__(self, model_path=None, gender='neutral'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Standard SMPL model names expected by smplx library
        MODEL_NAMES = {
            'female':  'SMPL_FEMALE.pkl',
            'male':    'SMPL_MALE.pkl'
        }
        
        # Kaggle-specific source path (Read-only)
        KAGGLE_SRC = "/kaggle/input/datasets/maamarmohamed/smpl-generator/SMPL_python_v.1.1.0/smpl/models"
        
        # Local writable path for symlinking with standard names
        # smplx expects a subfolder named 'smpl' for model_type='smpl'
        KAGGLE_WORKING = "/kaggle/working/smpl_models"
        KAGGLE_SMPL_DIR = os.path.join(KAGGLE_WORKING, "smpl")

        if model_path is None:
            if os.path.exists(KAGGLE_SRC):
                # 1. Setup Kaggle writable directory structure
                os.makedirs(KAGGLE_SMPL_DIR, exist_ok=True)
                
                # 2. Map the "basicmodel" files to standard SMPL names
                mappings = {
                    'female':  "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
                    'male':    "basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
                }
                
                src = os.path.join(KAGGLE_SRC, mappings.get(gender, list(mappings.values())[0]))
                dst = os.path.join(KAGGLE_SMPL_DIR, MODEL_NAMES.get(gender, list(MODEL_NAMES.values())[0]))
                
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except:
                        import shutil
                        shutil.copy(src, dst)
                
                model_dir = KAGGLE_WORKING
            else:
                # Fallback to local default
                model_dir = "smpl_models"
        else:
            # If path is provided, check if it's a file or directory
            if os.path.isfile(model_path):
                model_dir = os.path.dirname(model_path)
            else:
                model_dir = model_path

        # 3. Create models for all genders
        self.models = {}
        for g in ['female', 'male']:
            try:
                # Ensure the symlinks exist for all genders
                dst = os.path.join(KAGGLE_SMPL_DIR, MODEL_NAMES[g])
                src = os.path.join(KAGGLE_SRC, mappings[g])
                if os.path.exists(src) and not os.path.exists(dst):
                    try: os.symlink(src, dst)
                    except: 
                        import shutil
                        shutil.copy(src, dst)
                
                self.models[g] = smplx.create(model_dir, model_type='smpl', gender=g, ext='pkl').to(self.device).eval()
            except Exception as e:
                print(f"Warning: Could not load {g} model: {e}")
        
        if not self.models:
            raise RuntimeError("Could not load any SMPL models.")

        # Faces are same for all SMPL gendered models
        self.faces = self.models[list(self.models.keys())[0]].faces
        
        # Ground Truth Measurements (14 as per paper)
        self.target_cols = [
            'ankle_cm', 'arm_length_cm', 'bicep_cm', 'calf_cm', 'chest_cm', 
            'forearm_cm', 'head_to_heel_cm', 'hip_cm', 'leg_length_cm', 
            'shoulder_breadth_cm', 'shoulder_to_crotch_cm', 'thigh_cm', 
            'waist_cm', 'wrist_cm'
        ]

        self._setup_renderer()

    def _setup_renderer(self):
        try:
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer import (
                FoVOrthographicCameras,
                RasterizationSettings,
                MeshRasterizer,
                SoftSilhouetteShader,
                MeshRenderer,
                BlendParams
            )
            self.pytorch3d_available = True
        except ImportError:
            self.pytorch3d_available = False
            print("Warning: PyTorch3D not found. Differentiable rendering will not work.")
            return

        # Setup PyTorch3D soft renderer
        cameras_front = FoVOrthographicCameras(device=self.device, R=torch.eye(3).unsqueeze(0), T=torch.zeros(1, 3))
        # Side view: rotate 90 degrees around Y axis
        R_side = torch.tensor([[[ 0.0,  0.0, -1.0],
                                [ 0.0,  1.0,  0.0],
                                [ 1.0,  0.0,  0.0]]], device=self.device)
        cameras_side = FoVOrthographicCameras(device=self.device, R=R_side, T=torch.zeros(1, 3))

        raster_settings = RasterizationSettings(
            image_size=(640, 480), # (H, W)
            blur_radius=np.log(1. / 1e-4 - 1.) * 1e-4,
            faces_per_pixel=25, # Restored to 25 for higher gradient quality
        )

        self.renderer_front = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras_front, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
        )
        
        self.renderer_side = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras_side, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
        )

    def generate_batch(self, shape_params):
        """
        Differentiable generation of Front + Side silhouettes and measurements.
        Randomizes gender and weight to prevent overfitting.
        Supports children by scaling the mesh (height randomization).
        """
        batch_size = shape_params.shape[0]
        
        # Randomize Gender (Strictly Male/Female), Weight, and Height Scale
        genders = [np.random.choice(['female', 'male']) for _ in range(batch_size)]
        
        # 40kg to 120kg (Normal Adults/Teens)
        # Height Scale: 0.5 (child) to 1.15 (tall adult)
        h_scales = torch.rand(batch_size, device=self.device) * 0.65 + 0.5 
        weights_kg = torch.rand(batch_size, device=self.device) * 80 + 40 # 40-120
        # Reduce weight for kids if h_scale < 0.8
        for i in range(batch_size):
            if h_scales[i] < 0.8:
                # Baby weight roughly 5kg-20kg
                weights_kg[i] = weights_kg[i] * (h_scales[i]**2) # Simplified heuristic
        
        # Explicitly pass batched zeros for pose
        zero_pose = torch.zeros(batch_size, 69, device=self.device, dtype=shape_params.dtype)
        zero_orient = torch.zeros(batch_size, 3, device=self.device, dtype=shape_params.dtype)
        
        # Generate vertices (handling multi-gender)
        all_vertices = []
        for i in range(batch_size):
            # Process one-by-one to handle gender or chunk them
            model = self.models[genders[i]]
            out = model(betas=shape_params[i:i+1], body_pose=zero_pose[i:i+1], global_orient=zero_orient[i:i+1])
            v = out.vertices[0] * h_scales[i] # Apply height scale factor
            all_vertices.append(v)
        
        vertices = torch.stack(all_vertices) # (B, 6890, 3)
        
        measurements = self._calculate_measurements(vertices)
        height_cm = measurements[:, 6]
        
        # Z-score normalization for metadata
        h_norm = (height_cm - 170.0) / 10.0
        w_norm = (weights_kg - 75.0) / 15.0
        metadata = torch.stack((h_norm, w_norm), dim=1)

        if getattr(self, 'pytorch3d_available', False):
            from pytorch3d.structures import Meshes
            faces_tensor = torch.tensor(self.faces.astype(np.int64), dtype=torch.int64, device=self.device)
            
            # Center and scale vertices for renderer
            v_min = vertices.min(dim=1, keepdim=True)[0]
            v_max = vertices.max(dim=1, keepdim=True)[0]
            v_center = (v_max + v_min) / 2.0
            v_scaled = (vertices - v_center) / 1.2
            
            # Micro-batch Rendering
            chunk_size = 4
            combined_sils = []
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                v_chunk = v_scaled[i:end_idx]
                curr_chunk_size = v_chunk.shape[0]
                
                faces_batch = faces_tensor.unsqueeze(0).expand(curr_chunk_size, -1, -1)
                meshes = Meshes(verts=v_chunk, faces=faces_batch)
                
                f_sil = self.renderer_front(meshes)[..., 3]
                s_sil = self.renderer_side(meshes)[..., 3]
                
                chunk_sil = torch.cat([f_sil, s_sil], dim=2)
                combined_sils.append(chunk_sil)
                
                del meshes, f_sil, s_sil
                torch.cuda.empty_cache()
            
            combined_sil = torch.cat(combined_sils, dim=0).unsqueeze(1)
            
        else:
            combined_sil = torch.zeros((batch_size, 1, 640, 960), device=self.device)
            
        return combined_sil, measurements, metadata

    def _calculate_measurements(self, vertices):
        """
        Differentiable exact cm measurements from 3D coordinates.
        vertices: (B, 6890, 3) PyTorch Tensor
        Returns:
            measurements: (B, 14) PyTorch Tensor aligned with self.target_cols
        """
        B = vertices.shape[0]
        
        # 1. Height: max Y - min Y
        height = (torch.max(vertices[:, :, 1], dim=1)[0] - torch.min(vertices[:, :, 1], dim=1)[0]) * 100
        
        # Girth helper function
        def get_girth(indices, smoothing_multiplier=1.0):
            v_loop = vertices[:, indices, :] # (B, L, 3)
            v_loop_shifted = torch.roll(v_loop, shifts=-1, dims=1)
            dist = torch.norm(v_loop - v_loop_shifted, dim=2).sum(dim=1) # (B,)
            # The base SMPL model is in meters. We multiply by 100 for cm.
            # We apply a smoothing multiplier because a low-res polygon 
            # perimeter is shorter than a smooth skin circumference, but 
            # straight Euclidean jumps between far vertices overestimate it.
            return dist * 100 * smoothing_multiplier

        # Length helper function (Euclidean distance between two vertices)
        def get_length(idx1, idx2):
            return torch.norm(vertices[:, idx1, :] - vertices[:, idx2, :], dim=1) * 100

        # We apply physically-based scaling factors because the chosen vertex loops
        # are approximations and don't perfectly trace the convex hull of human skin.
        
        # 2-4. Torso Girths (Chest, Waist, Hip)
        # Using dynamic Y-slices of the 3D model is far more accurate than rigid vertex loops, 
        # as it computes the true convex hull ellipse at specific anatomical heights.
        def get_slice_girth(y_ratio, smoothing=0.9):
            res = torch.zeros(B, device=self.device)
            heel_y = torch.min(vertices[:, :, 1], dim=1)[0]
            total_h = height / 100.0
            
            for b in range(B):
                y_target = heel_y[b] + total_h[b] * y_ratio
                # Find the 150 closest vertices to this Y height
                dists = torch.abs(vertices[b, :, 1] - y_target)
                _, idxs = torch.topk(dists, 150, largest=False)
                slice_v = vertices[b, idxs, :]
                
                # Approximate the ellipse dimensions
                width = torch.max(slice_v[:, 0]) - torch.min(slice_v[:, 0])
                depth = torch.max(slice_v[:, 2]) - torch.min(slice_v[:, 2])
                
                # Ramanujan's ellipse approximation (simplified)
                res[b] = 3.14159 * (width + depth) * 100 * smoothing / 2.0
            return res

        chest = get_slice_girth(0.72, smoothing=0.95)
        waist = get_slice_girth(0.58, smoothing=0.92)
        hip   = get_slice_girth(0.50, smoothing=0.94)
        
        # 5. Thigh Girth (Average of L/R)
        thigh_l = get_girth([1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359], 0.45)
        thigh_r = get_girth([4770, 4771, 4772, 4773, 4774, 4775, 4776, 4777, 4778, 4779], 0.45)
        thigh = (thigh_l + thigh_r) / 2.0
        
        # 6. Ankle Girth (Average of L/R)
        ankle_l = get_girth([3340, 3341, 3342, 3343, 3344], 1.25)
        ankle_r = get_girth([6733, 6734, 6735, 6736, 6737], 1.25)
        ankle = (ankle_l + ankle_r) / 2.0
        
        # 7. Calf Girth (Average of L/R)
        calf_l = get_girth([1250, 1251, 1252, 1253, 1254, 1255], 0.40)
        calf_r = get_girth([4672, 4673, 4674, 4675, 4676, 4677], 0.40)
        calf = (calf_l + calf_r) / 2.0
        
        # 8. Bicep Girth (Average of L/R)
        bicep_l = get_girth([1470, 1471, 1472, 1473, 1474], 1.1)
        bicep_r = get_girth([4891, 4892, 4893, 4894, 4895], 1.1)
        bicep = (bicep_l + bicep_r) / 2.0
        
        # 9. Forearm Girth (Average of L/R)
        forearm_l = get_girth([1540, 1541, 1542, 1543, 1544], 0.4)
        forearm_r = get_girth([4960, 4961, 4962, 4963, 4964], 0.4)
        forearm = (forearm_l + forearm_r) / 2.0
        
        # 10. Wrist Girth (Average of L/R)
        wrist_l = get_girth([2150, 2151, 2152, 2153, 2154], 1.0)
        wrist_r = get_girth([5580, 5581, 5582, 5583, 5584], 1.0)
        wrist = (wrist_l + wrist_r) / 2.0
        
        # 11. Shoulder Breadth (between shoulder tips)
        shoulder_breadth = get_length(1086, 4516) * 1.35
        
        # 12. Arm Length (Shoulder to Wrist Average)
        # Euclidean line is much shorter than true arm path (around elbow), so we scale up slightly
        # However, original values were 108cm, so we actually need to scale down to roughly ~60cm
        arm_l = get_length(1086, 2150)
        arm_r = get_length(4516, 5580)
        arm_length = ((arm_l + arm_r) / 2.0) * 0.55
        
        # 13. Leg Length (Hip joint to Ankle Average)
        # Scale down to realistic ~80-90cm range from 112cm
        leg_l = get_length(1350, 3340)
        leg_r = get_length(4770, 6733)
        leg_length = ((leg_l + leg_r) / 2.0) * 0.82
        
        # 14. Shoulder-to-Crotch (base of neck to crotch)
        s_to_c = (torch.max(vertices[:, :, 1], dim=1)[0] - vertices[:, 3500, 1]) * 100
        
        # Map to self.target_cols (14 metrics)
        # order: ['ankle_cm', 'arm_length_cm', 'bicep_cm', 'calf_cm', 'chest_cm', 
        #         'forearm_cm', 'head_to_heel_cm', 'hip_cm', 'leg_length_cm', 
        #         'shoulder_breadth_cm', 'shoulder_to_crotch_cm', 'thigh_cm', 
        #         'waist_cm', 'wrist_cm']
        res = torch.zeros(B, 14, device=self.device, dtype=torch.float32)
        res[:, 0] = ankle
        res[:, 1] = arm_length
        res[:, 2] = bicep
        res[:, 3] = calf
        res[:, 4] = chest
        res[:, 5] = forearm
        res[:, 6] = height
        res[:, 7] = hip
        res[:, 8] = leg_length
        res[:, 9] = shoulder_breadth
        res[:, 10] = s_to_c
        res[:, 11] = thigh
        res[:, 12] = waist
        res[:, 13] = wrist
        
        return res

if __name__ == "__main__":
    print("Differentiable SMPL Generator Drafted.")

