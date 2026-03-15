import torch
import torch.nn as nn
import torchvision.models as models

class BMNet(nn.Module):
    """
    BMnet: Body Measurement Network (Official Paper Version)
    Estimates 14 body measurements from silhouette images + Height/Weight channels.
    Paper: "Human Body Measurement Estimation with Adversarial Augmentation"
    """
    def __init__(self, num_measurements=14):
        super(BMNet, self).__init__()
        
        # 1. Backbone: MNASNet-1.0 (as per paper)
        # The paper uses MNASNet with a depth multiplier of 1.
        self.backbone = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
        
        # Remove the default classifier
        # MNASNet-1.0 final feature dimension is 1280
        self.features = self.backbone.layers
        
        # 2. Regression Header
        # "fed into an MLP comprising a hidden layer of 128 neurons and 14 outputs"
        self.header = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, num_measurements)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, 3, 640, 960) 
                        Channel 0: Symmetrically concatenated silhouettes
                        Channel 1: Constant Height metadata image
                        Channel 2: Constant Weight metadata image
        """
        x = self.features(x)
        x = self.header(x)
        return x

if __name__ == "__main__":
    # Test with paper-compliant dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BMNet().to(device)
    
    # Batch size 2, 3 channels, 640x960 resolution
    dummy_input = torch.randn(2, 3, 640, 960).to(device)
    
    output = model(dummy_input)
    print(f"BMnet (v2) successfully processed paper-compliant input.")
    print(f"Output shape: {output.shape} (Expected: [2, 14])")
