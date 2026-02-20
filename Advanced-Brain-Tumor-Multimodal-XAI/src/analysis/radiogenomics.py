import torch
import torch.nn as nn

class RadiogenomicAnalyzer:
    """
    Predicts molecular markers (IDH1, MGMT, 1p/19q) alongside tumor classification.
    """
    def __init__(self):
        # IDH Mutation Status (Binary)
        self.idh_model = nn.Sequential(
            nn.Linear(102, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # MGMT Promoter Methylation (Binary)
        self.mgmt_model = nn.Sequential(
            nn.Linear(102, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 1p/19q Codeletion (Binary)
        self.codeletion_model = nn.Sequential(
            nn.Linear(102, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def predict(self, radiomic_features):
        idh_pred = self.idh_model(radiomic_features)
        mgmt_pred = self.mgmt_model(radiomic_features)
        codeletion_pred = self.codeletion_model(radiomic_features)
        
        return {
            "idh_mutation": idh_pred,
            "mgmt_methylation": mgmt_pred,
            "codeletion_status": codeletion_pred
        }

class RadiogenomicReport:
    """
    Generates a clinical report for Radiogenomic findings.
    """
    def __init__(self):
        self.marker_names = {
            "idh_mutation": "IDH1 Mutation Status",
            "mgmt_methylation": "MGMT Promoter Methylation",
            "codeletion_status": "1p/19q Codeletion"
        }

    def generate_report(self, predictions):
        report = "Radiogenomic Analysis Report:\n"
        for marker, pred in predictions.items():
            prob = pred.item()
            status = "Positive" if prob > 0.5 else "Negative"
            report += f"\n{self.marker_names[marker]}: {status} (Probability: {prob:.2f})"
        
        return report
