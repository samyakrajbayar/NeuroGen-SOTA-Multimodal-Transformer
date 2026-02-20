# Technical Proposal: SOTA Multimodal Transformer Framework

## 1. Architectural Shift: From CNN to Swin-ViT
We have transitioned from standard 3D CNNs to a **Swin-UNETR Transformer** backbone. 
*   **Why?** Transformers capture global context through Self-Attention, which is critical for identifying distal infiltration patterns in high-grade tumors that CNNs (with local kernels) often miss.

## 2. Advanced Fusion: Cross-Modal Attention (CMA)
Simple concatenation (Early Fusion) or Gated Summation (Late Fusion) are insufficient for complex biomedical data. 
*   **Mechanism:** Our CMA module uses the Radiomic Feature Vector as a **Query** to attend to the **Keys/Values** of the volumetric image features. 
*   **Result:** The model dynamically "looks" at different MRI regions depending on the quantitative texture signatures (e.g., if the texture is heterogeneous, the model focuses more on the necrotic core).

## 3. Trust & Safety: Bayesian Approximation
We implement **Monte Carlo Dropout** to provide a Bayesian interpretation of the model's output. Every diagnosis is accompanied by a **Standard Deviation (SD)**, quantifying the model's epistemic uncertainty.

## 4. Multi-Task Learning (MTL)
The model is trained simultaneously for **Voxel-wise Segmentation** and **Patient-level Classification**. This dual-objective forces the transformer to learn high-fidelity spatial representations that are anatomically accurate, preventing the model from taking "shortcuts" based on image noise.
