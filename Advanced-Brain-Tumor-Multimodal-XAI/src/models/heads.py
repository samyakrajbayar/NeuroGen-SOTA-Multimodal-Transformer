import torch
import torch.nn as nn

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
