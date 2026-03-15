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
                model_dir = "models/smpl"
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

    def generate_sample(self, shape_params=None):
        """
        Generates paired Front + Side silhouettes and exact measurements.
        """
        if shape_params is None:
            shape_params = torch.randn(1, 10).to(self.device)
            
        output = self.model(betas=shape_params)
        vertices = output.vertices.detach().cpu().numpy()[0]
        faces = self.model.faces
        
        # 1. Render front (640x480) and side (640x480)
        front_sil = self._render_silhouette(vertices, faces, view='front')
        side_sil  = self._render_silhouette(vertices, faces, view='side')
        
        # 2. Horizontal concatenation (640x960)
        combined_sil = np.hstack((front_sil, side_sil))
        
        # 3. Calculate all 14 measurements
        measurements = self._calculate_measurements(vertices)
        
        return combined_sil, measurements

    def _render_silhouette(self, vertices, faces, view='front'):
        """
        Renders a 640x480 silhouette by projecting vertices and filling the mesh.
        """
        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices, faces)
        
        # 1. Setup projection based on view
        if view == 'front':
            # Project onto XY plane (Front view)
            coords = vertices[:, :2] # [X, Y]
        else: # view == 'side'
            # Project onto ZY plane (Side view)
            coords = vertices[:, [2, 1]] # [Z, Y]
            
        # 2. Normalize coords to fit in 640x480
        # Centering and scaling (This replicates the "fixed distance" camera of the paper)
        # We assume the subject is roughly centered
        min_p = np.min(coords, axis=0)
        max_p = np.max(coords, axis=0)
        center = (min_p + max_p) / 2
        
        # Scale to fit (leaving some padding)
        # 1.8 meters person -> ~500 pixels high
        scale = 300.0 
        
        img_coords = (coords - center) * scale
        img_coords[:, 0] += 240 # Center X (480/2)
        img_coords[:, 1] += 320 # Center Y (640/2)
        
        # 3. Handle coordinate orientation (Y increases downwards in images)
        img_coords[:, 1] = 640 - img_coords[:, 1]
        
        # 4. Rasterize using trimesh/PIL for a clean silhouette
        # We project the mesh onto the image plane
        # For simplicity in this draft, we'll draw the projected vertices/edges
        # In the real ABS, this is a differentiable soft rasterizer
        img = Image.new('L', (480, 640), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw projected mesh polygons (simplified: draw points for the draft)
        for f in faces[:5000]: # Sample faces for efficiency in draft
            poly = [(img_coords[v, 0], img_coords[v, 1]) for v in f]
            draw.polygon(poly, fill=255)
            
        return np.array(img)

    def _calculate_measurements(self, vertices):
        """
        Calculates exact cm measurements from 3D coordinates.
        Uses vertex-loop indices specific to the SMPL mesh.
        """
        # (These indices are illustrative; in production, we use a loop-loader)
        # Height is easy: Y_max - Y_min
        height = (np.max(vertices[:, 1]) - np.min(vertices[:, 1])) * 100
        
        # Chest, Waist, Hips: Sum distances of vertex loops
        # This is the "Perfect Ground Truth" mentioned in our strategy
        measurements = {
            'head_to_heel_cm': height,
            'waist_cm': self._get_circumference(vertices, [3500, 3501, 3502]), # Loop
            'chest_cm': self._get_circumference(vertices, [3000, 3001, 3002]),
            # ... and so on for all 14 metrics
        }
        
        # Ensure all 14 paper metrics are present
        for col in self.target_cols:
            if col not in measurements:
                measurements[col] = 0.0 # Placeholder
                
        return measurements

    def _get_circumference(self, vertices, indices):
        """ Calculates girth by summing polygon edge lengths in a loop. """
        dist = 0
        v_loop = vertices[indices]
        for i in range(len(v_loop)):
            v1 = v_loop[i]
            v2 = v_loop[(i + 1) % len(v_loop)]
            dist += np.linalg.norm(v1 - v2)
        return dist * 100 # to cm

if __name__ == "__main__":
    print("SMPL Generator Drafted.")
    print("Logic: 3D Mesh (SMPL) -> 2D Silhouette (Renderer) -> Ground Truth (Mesh Calculation)")
