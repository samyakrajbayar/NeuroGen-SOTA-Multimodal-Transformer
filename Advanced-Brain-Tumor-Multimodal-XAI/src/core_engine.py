import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetResBlock
import numpy as np
from monai.transforms import Compose, LoadImaged, Spacing, NormalizeIntensity

class MultiHeadCrossAttention(nn.Module):
    """
    Bio-Inspired Cross-Attention Fusion.
    Allows Radiomic 'Clinical Context' to attend to 'Visual MRI' regions.
    """
    def __init__(self, visual_dim=768, radiomic_dim=102, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=num_heads, batch_first=True)
        
        # Project Radiomics to match Visual embedding space
        self.rad_proj = nn.Linear(radiomic_dim, visual_dim)
        self.norm_v = nn.LayerNorm(visual_dim)
        self.norm_r = nn.LayerNorm(visual_dim)

    def forward(self, v_feat, r_feat):
        # v_feat: [B, C, D, H, W] -> [B, N, C]
        b, c, d, h, w = v_feat.shape
        v_flat = v_feat.view(b, c, -1).permute(0, 2, 1) 
        
        # r_feat: [B, R] -> [B, 1, C]
        r_proj = self.rad_proj(r_feat).unsqueeze(1)
        
        # Multi-Head Attention: Radiomics (Query) attends to Image (Key/Value)
        v_norm = self.norm_v(v_flat)
        r_norm = self.norm_r(r_proj)
        
        attn_out, weights = self.mha(query=r_norm, key=v_norm, value=v_norm)
        
        return attn_out.squeeze(1), weights

class RadiogenomicHead(nn.Module):
    """
    Predicts molecular markers (IDH1, MGMT, 1p/19q) alongside tumor classification.
    """
    def __init__(self, input_dim=768):
        super().__init__()
        
        # Branch 1: Tumor Type (Glioma, Meningioma, Pituitary)
        self.type_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3)
        )
        
        # Branch 2: Molecular Marker (IDH Mutation Status - Binary)
        self.idh_predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Branch 3: MGMT Promoter Methylation (Regression/Binary)
        self.mgmt_predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        type_logits = self.type_classifier(x)
        idh_prob = self.idh_predictor(x)
        mgmt_prob = self.mgmt_predictor(x)
        return {
            "tumor_type": type_logits,
            "idh_status": idh_prob,
            "mgmt_methylation": mgmt_prob
        }

class NeuroGenMultimodalTransformer(nn.Module):
    """
    Decoupled Multimodal Transformer for Brain Tumor Classification + Radiogenomics.
    """
    def __init__(self, img_size=(128, 128, 128), radiomic_dim=102):
        super().__init__()
        
        # 1. Visual Backbone (Swin-UNETR)
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=1, # Dummy out, we use hidden states
            feature_size=48
        )
        
        # 2. Cross-Modal Attention Fusion
        self.cma = MultiHeadCrossAttention(visual_dim=768, radiomic_dim=radiomic_dim)
        
        # 3. Radiogenomic Prediction Head (Multi-task)
        self.radiogenomic_head = RadiogenomicHead(input_dim=768)
        
        # 4. Uncertainty Estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Variance output
        )

    def forward(self, x_img, x_radiomics):
        # 1. Extract visual features
        hidden_states = self.backbone.swinViT(x_img)
        visual_features = hidden_states[-1] # [B, 768, 4, 4, 4]
        
        # 2. Cross-Modal Fusion (Radiomics queries Image)
        fused_features, attn_weights = self.cma(visual_features, x_radiomics)
        
        # 3. Radiogenomic Predictions
        predictions = self.radiogenomic_head(fused_features)
        
        # 4. Uncertainty Estimation
        uncertainty = self.uncertainty_head(fused_features)
        
        return {
            "predictions": predictions,
            "uncertainty": uncertainty,
            "attention_weights": attn_weights
        }

    def predict_with_confidence(self, x_img, x_radiomics, n_iter=10):
        """
        Monte Carlo Dropout for Epistemic Uncertainty.
        """
        results = []
        uncertainties = []
        
        for _ in range(n_iter):
            # Enable dropout during inference
            self.train()
            output = self.forward(x_img, x_radiomics)
            results.append(output["predictions"])
            uncertainties.append(output["uncertainty"])
        
        # Compute mean and variance
        mean_preds = {k: torch.stack([r[k] for r in results]).mean(0) for k in results[0].keys()}
        mean_uncertainty = torch.stack(uncertainties).mean(0)
        
        return mean_preds, mean_uncertainty

if __name__ == "__main__":
    # Initialize NeuroGen Model
    model = NeuroGenMultimodalTransformer()
    
    dummy_mri = torch.randn(1, 1, 128, 128, 128)
    dummy_rad = torch.randn(1, 102)
    
    output = model(dummy_mri, dummy_rad)
    print("Tumor Type Logits:", output["predictions"]["tumor_type"])
    print("IDH Probability:", output["predictions"]["idh_status"])
    print("Uncertainty:", output["uncertainty"])
    print("Attention Weights Shape:", output["attention_weights"].shape)
    
    # Test Uncertainty Estimation
    mean_preds, mean_uncertainty = model.predict_with_confidence(dummy_mri, dummy_rad)
    print("Mean Tumor Type:", mean_preds["tumor_type"])
    print("Mean Uncertainty:", mean_uncertainty)
