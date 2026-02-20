# NeuroGen: SOTA Multimodal Transformer for Brain Tumor Classification + Radiogenomics

![Architecture](https://img.shields.io/badge/Architecture-Swin--ViT%20%2B%20CMA-orange)
![Radiogenomics](https://img.shields.io/badge/Radiogenomics-IDH%2C%20MGMT%2C%201p%2F19q-green)
![Uncertainty](https://img.shields.io/badge/Uncertainty-Bayesian%20AI-red)

## 🚀 Overview
This repository implements **NeuroGen**, a **State-of-the-Art (SOTA) Multimodal Transformer** for Brain Tumor Classification and Radiogenomics. It predicts not only tumor types but also molecular markers (IDH1, MGMT, 1p/19q) with quantified uncertainty.

### 🔬 Why this is "Most Advanced"
1.  **3D Swin-Transformer Backbone:** Captures global spatial context for tumor infiltration patterns.
2.  **Cross-Modal Attention (CMA):** Radiomic features actively "query" MRI spatial features for intelligent fusion.
3.  **Radiogenomic Prediction:** Predicts molecular markers (IDH1, MGMT, 1p/19q) alongside tumor types.
4.  **Bayesian Uncertainty:** Monte Carlo Dropout provides epistemic uncertainty for clinical safety.
5.  **Clinical-Grade Preprocessing:** N4 Bias Field Correction & Non-Local Means denoising.

## 📂 Modular Project Structure
```bash
.
├── src/models/
│   ├── backbone.py          # Swin-UNETR Transformer
│   ├── fusion.py           # Cross-Modal Attention
│   ├── heads.py            # Radiogenomic Prediction
│   └── neurogen.py         # Main NeuroGen Model
├── src/data/
│   ├── pipeline.py         # MONAI-powered Dataset
│   ├── preprocessing.py    # N4 Bias & Denoising
│   └── radiomics.py        # Feature Extraction
├── src/interpret/
│   ├── xai.py              # Advanced XAI (SHAP + Volumetric Grad-CAM)
│   └── uncertainty.py      # Aleatoric vs. Epistemic
└── src/analysis/
    └── radiogenomics.py    # Molecular Marker Analysis
```

## 🛠️ Advanced Tech Stack
- **Transformer Engine:** MONAI SwinUNETR (3D Vision Transformer)
- **Fusion Logic:** Multi-Head Cross-Attention (CMA)
- **Radiomics:** PyRadiomics (Clinical Texture Analysis)
- **Uncertainty:** Bayesian MC Dropout
- **XAI:** SHAP + Integrated Gradients
- **Data Ops:** SimpleITK + NiBabel + MONAI Transforms

## 📊 Clinical Safety Features
- **Radiogenomic Predictions:** IDH1 mutation, MGMT methylation, 1p/19q codeletion.
- **Uncertainty Quantification:** Epistemic variance for every prediction.
- **Multi-Task Learning:** Joint Segmentation + Classification + Radiogenomics.
- **Explainable AI:** Volumetric Grad-CAM heatmaps for tumor regions.

## ⚙️ Installation
```bash
pip install torch torchvision monai captum shap nibabel SimpleITK dipy pyradiomics
```

## 🚀 Usage
```python
from src.models.neurogen import NeuroGenMultimodalTransformer
from src.data.preprocessing import NeuroGenPreprocessor

# Initialize
model = NeuroGenMultimodalTransformer()
preprocessor = NeuroGenPreprocessor()

# Preprocess MRI volume
clean_volume = preprocessor.process_volume("mri_scan.nii.gz")

# Predict with confidence
mean_preds, uncertainty = model.predict_with_confidence(clean_volume, radiomic_features)
```

## 🤝 Contributing
This work is designed for **Research Purposes Only** and is not a substitute for professional medical diagnosis. I welcome feedback from researchers and clinicians in Neuro-Oncology and AI.
