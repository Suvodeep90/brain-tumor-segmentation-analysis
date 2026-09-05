# 1 — Segmentation Models (3D U-Net)

Custom 3D U-Net for brain tumor segmentation on BraTS 2020.

| Notebook | Purpose |
|---|---|
| `unet_3d/Train_UNet3D.ipynb` | **Full training pipeline** — data prep, architecture, Trainer class, training loop, curve plots |
| `unet_3d/Run_Inference.ipynb` | Load a trained checkpoint, run inference on the validation set, write per-case Dice CSV |
| `unet_3d/Individual_Model.ipynb` | Per-model Dice analysis; computes concordant-good / concordant-poor case sets |

## Architecture

Encoder-decoder with 4 encoding stages and 4 decoding stages.

| Block | Channels (base `n=24`) |
|---|---|
| Input conv | 4 → 24 |
| Enc 1 | 24 → 48 |
| Enc 2 | 48 → 96 |
| Enc 3 | 96 → 192 |
| Enc 4 (bottleneck) | 192 → 192 |
| Dec 1–4 | mirror of encoder |
| Output | 24 → 3 |

Each conv block: `Conv3D → GroupNorm → ReLU` × 2.  
Upsampling: trilinear interpolation + skip connection concat.

## Training Details

| Setting | Value |
|---|---|
| Loss | BCE + Dice |
| Optimizer | Adam, lr=1e-3 |
| Scheduler | ReduceLROnPlateau (patience=7, factor=0.5) |
| Epochs | 70 |
| Batch size | 1 |
| Validation fold | fold 0 (stratified 6-fold) |
| Input channels | 4 (FLAIR, T1, T1CE, T2) |
| Output classes | 3 (WT, TC, ET) |

## Outputs

Saved to `config.model_path` (default: `Models/Unet-3D_New/`):

```
best_model.pth          ← best val Dice checkpoint
last_epoch_model.pth    ← final epoch checkpoint
train_log.csv           ← per-epoch metrics
training_curves.png     ← loss and Dice plots
```

## nnU-Net and TransBTS

See `../0_model_setup/README.md` for setup instructions and inference commands.
