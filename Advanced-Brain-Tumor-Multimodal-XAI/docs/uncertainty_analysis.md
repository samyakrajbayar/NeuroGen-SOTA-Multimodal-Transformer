# Uncertainty Analysis in Medical AI: Aleatoric vs. Epistemic

## 1. Introduction
In brain tumor classification, a model saying "I don't know" is as important as a correct diagnosis. We implement Bayesian approximation via Monte Carlo (MC) Dropout to quantify uncertainty.

## 2. Types of Uncertainty
### A. Aleatoric Uncertainty (Data Noise)
*   **Source:** Low-resolution MRI, motion artifacts, or poor contrast injection.
*   **Behavior:** It is inherent in the input data and cannot be reduced by more training data.
*   **Our Solution:** The model outputs a variance parameter (Uncertainty Head) to estimate observation noise.

### B. Epistemic Uncertainty (Model Knowledge)
*   **Source:** Out-of-distribution (OOD) cases. For example, a model trained on adults seeing a pediatric brain MRI.
*   **Behavior:** This uncertainty decreases as the model sees more diverse examples.
*   **Our Solution:** MC Dropout. By running inference 10-50 times with dropout enabled, we measure the "disagreement" between different internal sub-networks.

## 3. Clinical Safety Protocol
1.  **High Confidence (Var < 0.1):** Model prediction is shown to the clinician.
2.  **Low Confidence (Var > 0.4):** Model displays a "Review Required" warning and highlights the region of ambiguity.
3.  **Ambiguity Mapping:** We visualize the variance across the MRI volume to show *where* the model is confused (e.g., at the infiltrative margin of a Glioma).
