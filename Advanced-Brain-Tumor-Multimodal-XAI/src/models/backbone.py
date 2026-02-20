import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class NeuroGenBackbone(nn.Module):
    """
    Swin-UNETR Backbone for volumetric feature extraction.
    Outputs multi-scale hierarchical representations.
    """
    def __init__(self, img_size=(128, 128, 128), in_channels=1, feature_size=48):
        super().__init__()
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=1, # Dummy out, we use hidden states
            feature_size=feature_size
        )

    def forward(self, x):
        # Extract hierarchical features: [Res1, Res2, Res3, Res4, Bottleneck]
        hidden_states = self.swin.swinViT(x)
        return hidden_states[-1] # Focus on the deep bottleneck (768 dim)
