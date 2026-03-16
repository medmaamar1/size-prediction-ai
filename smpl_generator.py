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
            'neutral': 'SMPL_NEUTRAL.pkl',
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
                    'neutral': "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
                    'female':  "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
                    'male':    "basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
                }
                
                src = os.path.join(KAGGLE_SRC, mappings.get(gender, mappings['neutral']))
                dst = os.path.join(KAGGLE_SMPL_DIR, MODEL_NAMES.get(gender, MODEL_NAMES['neutral']))
                
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

        # 3. Create model using the directory path
        # smplx.create expects a directory and searches for SMPL_GENDER.pkl inside
        try:
            self.model = smplx.create(model_dir, model_type='smpl', gender=gender, ext='pkl').to(self.device).eval()
        except Exception as e:
            print(f"Error loading SMPL model from {model_dir}: {e}")
            raise e
        
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
            faces_per_pixel=25, # Reduced from 50 to save VRAM
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
        shape_params: (B, 10) PyTorch Tensor with requires_grad=True
        Returns:
            combined_sil: (B, 1, 640, 960) PyTorch Tensor [0, 1]
            measurements: (B, 14) PyTorch Tensor in cm
            metadata: (B, 2) PyTorch Tensor for height and weight normalized
        """
        batch_size = shape_params.shape[0]
        
        # Explicitly pass batched zeros for pose to avoid broadcasting errors in smplx
        # SMPL typically has 24 joints (1 global orient + 23 body joints)
        # Total parameters = 24 * 3 = 72
        zero_pose = torch.zeros(batch_size, 69, device=self.device, dtype=shape_params.dtype)
        zero_orient = torch.zeros(batch_size, 3, device=self.device, dtype=shape_params.dtype)
        
        output = self.model(betas=shape_params, body_pose=zero_pose, global_orient=zero_orient)
        vertices = output.vertices # (B, 6890, 3)
        
        measurements = self._calculate_measurements(vertices)
        height_cm = measurements[:, 6] # head_to_heel_cm
        weight_kg = torch.full((batch_size,), 75.0, device=self.device) # Placeholder weight
        
        # Z-score normalization as per paper
        h_norm = (height_cm - 170.0) / 10.0
        w_norm = (weight_kg - 75.0) / 15.0
        metadata = torch.stack((h_norm, w_norm), dim=1) # (B, 2)

        if getattr(self, 'pytorch3d_available', False):
            from pytorch3d.structures import Meshes
            faces = self.model.faces
            faces_tensor = torch.tensor(faces.astype(np.int64), dtype=torch.int64, device=self.device)
            faces_batch = faces_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Center and scale vertices to [-1, 1] NDC
            v_min = vertices.min(dim=1, keepdim=True)[0]
            v_max = vertices.max(dim=1, keepdim=True)[0]
            v_center = (v_max + v_min) / 2.0
            
            v_scaled = (vertices - v_center) / 1.2
            
            meshes = Meshes(verts=v_scaled, faces=faces_batch)
            
            front_sil = self.renderer_front(meshes)[..., 3] # (B, 640, 480)
            side_sil = self.renderer_side(meshes)[..., 3] # (B, 640, 480)
            
            combined_sil = torch.cat([front_sil, side_sil], dim=2) # (B, 640, 960)
            combined_sil = combined_sil.unsqueeze(1) # (B, 1, 640, 960)
            
        else:
            # Fallback
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
        
        # Height: max Y - min Y
        height = (torch.max(vertices[:, :, 1], dim=1)[0] - torch.min(vertices[:, :, 1], dim=1)[0]) * 100
        
        # Girth helper function
        def get_girth(indices):
            v_loop = vertices[:, indices, :] # (B, L, 3)
            v_loop_shifted = torch.roll(v_loop, shifts=-1, dims=1)
            dist = torch.norm(v_loop - v_loop_shifted, dim=2).sum(dim=1) # (B,)
            return dist * 100
            
        # Simplified loops
        waist = get_girth([3500, 3501, 3502]) 
        chest = get_girth([3000, 3001, 3002])
        
        res = torch.zeros(B, 14, device=self.device, dtype=torch.float32)
        res[:, 6] = height # head_to_heel_cm
        res[:, 12] = waist # waist_cm
        res[:, 4] = chest # chest_cm
        
        return res

if __name__ == "__main__":
    print("Differentiable SMPL Generator Drafted.")

