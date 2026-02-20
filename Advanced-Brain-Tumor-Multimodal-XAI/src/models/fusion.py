import torch
import torch.nn as nn

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
