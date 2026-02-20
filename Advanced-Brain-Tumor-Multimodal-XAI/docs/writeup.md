# NeuroGen: SOTA Multimodal Transformer for Brain Tumor Classification + Radiogenomics

## Vision
The goal of NeuroGen is to bridge the gap between "Black Box" deep learning and clinical practice in neuro-oncology. This system predicts not only tumor types (Glioma, Meningioma, Pituitary) but also molecular markers (IDH1, MGMT, 1p/19q) with quantified uncertainty.

## Key Innovations
1.  **3D Swin-Transformer:** Captures global spatial context and tumor infiltration patterns that 2D CNNs miss.
2.  **Cross-Modal Attention (CMA):** Radiomic features (clinical biomarkers) actively "query" MRI spatial features for intelligent fusion.
3.  **Radiogenomic Prediction:** Joint prediction of tumor type and molecular markers (IDH1 mutation, MGMT methylation, 1p/19q codeletion).
4.  **Bayesian Uncertainty:** Monte Carlo Dropout provides epistemic uncertainty for clinical safety.
5.  **Clinical-Grade Preprocessing:** N4 Bias Field Correction and Non-Local Means denoising.

## Impact
This system can be used as a "Second Opinion" tool in neuro-oncology. If the model predicts a Glioblastoma with high IDH mutation probability, the Grad-CAM heatmap will highlight the tumor core, allowing the clinician to verify the model's focus. This creates a human-in-the-loop system built on transparency and safety.

## Clinical Safety Protocol
1.  **High Confidence (Var < 0.1):** Model prediction is shown to the clinician.
2.  **Low Confidence (Var > 0.4):** Model displays a "Review Required" warning and highlights the region of ambiguity.
3.  **Radiogenomic Report:** Generated for every prediction, providing molecular marker probabilities.
