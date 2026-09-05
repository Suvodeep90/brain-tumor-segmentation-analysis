# 1 — Segmentation Models

Contains inference code for the 3D U-Net used in the paper.

| Notebook | Purpose |
|---|---|
| `unet_3d/Run_Inference.ipynb` | Load a trained 3D U-Net checkpoint, run on BraTS test volumes, write per-case Dice scores to CSV |
| `unet_3d/Individual_Model.ipynb` | Per-model Dice analysis; computes concordant-good / concordant-poor case sets |

**nnU-Net** and **TransBTS** were run using their official repositories. Results CSVs are consumed by the downstream analysis notebooks.
