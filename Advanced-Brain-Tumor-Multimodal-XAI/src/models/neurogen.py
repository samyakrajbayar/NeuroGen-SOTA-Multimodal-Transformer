import torch
import torch.nn as nn
from .backbone import NeuroGenBackbone
from .fusion import MultiHeadCrossAttention
from .heads import RadiogenomicHead

class NeuroGenMultimodalTransformer(nn.Module):
    """
    Decoupled Multimodal Transformer for Brain Tumor Classification + Radiogenomics.
    """
    def __init__(self, img_size=(128, 128, 128), radiomic_dim=102):
        super().__init__()
        
        # 1. Visual Backbone (Swin-UNETR)
        self.backbone = NeuroGenBackbone(img_size=img_size)
        
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
        visual_features = self.backbone(x_img) # [B, 768, D, H, W]
        
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
