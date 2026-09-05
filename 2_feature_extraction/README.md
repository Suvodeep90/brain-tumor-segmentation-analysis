# 2 — Feature Extraction

Extracts all feature families from raw MRI volumes and segmentation masks.
Run each group in any order; they write independent output CSVs.

| Subfolder | Features |
|---|---|
| `shape/` | Volume, surface area, local curvature |
| `intensity/` | First-order stats (mean, entropy, kurtosis, …), centroid position |
| `texture/` | GLCM, GLDM, GLRLM, GLSZM, NGTDM (via PyRadiomics) |
| `model_based/` | Gradient saliency, softmax probability features from U-Net |
| `radiomics/` | PyRadiomics extraction and post-processing |
