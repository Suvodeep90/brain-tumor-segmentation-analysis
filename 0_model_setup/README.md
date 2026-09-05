# 0 — Segmentation Model Setup

This section covers how to set up and run each of the three segmentation models
used in the paper. The models produce per-case Dice score CSVs consumed by all
downstream analysis notebooks.

---

## Models

| Model | Architecture | Official repo |
|---|---|---|
| **3D U-Net** | Custom 3D encoder-decoder | In `1_segmentation_models/unet_3d/` |
| **nnU-Net v2** | Self-configuring framework | https://github.com/MIC-DKFZ/nnUNet |
| **TransBTS** | Transformer-based U-Net | https://github.com/Rubics-Xuan/TransBTS |

---

## Environment Setup

Each model requires its own conda environment due to conflicting dependencies.

```bash
# Analysis notebooks (feature extraction, oracle, figures)
conda env create -f environments/environment_pytorch.yml
conda activate pytorch

# nnU-Net inference
conda env create -f environments/environment_nnunet.yml
conda activate nnunet

# TransBTS inference
conda env create -f environments/environment_TransBTS.yml
conda activate TransBTS
```

---

## 3D U-Net

Uses the custom architecture defined in `1_segmentation_models/unet_3d/Run_Inference.ipynb`.

1. Update `GlobalConfig.root_dir` and `model_path` at the top of the notebook
2. Point `train_root_dir` / `test_root_dir` to your BraTS data
3. Run all cells — outputs `Unet_test_dice.csv`

---

## nnU-Net v2

Install from the official repo (v2.2 was used):

```bash
pip install nnunetv2==2.2
```

### Dataset conversion

```bash
nnUNetv2_convert_MSD_dataset -i /path/to/BraTS2020 -overwrite_id 082
```

### Training (5-fold cross-validation)

```bash
nnUNetv2_train 082 3d_fullres 0
nnUNetv2_train 082 3d_fullres 1
nnUNetv2_train 082 3d_fullres 2
nnUNetv2_train 082 3d_fullres 3
nnUNetv2_train 082 3d_fullres 4
```

### Inference

```bash
nnUNetv2_predict \
    -i /path/to/BraTS2020/imagesTs \
    -o /path/to/output \
    -d 082 \
    -c 3d_fullres \
    -f all \
    --save_probabilities
```

Output: `summary.json` with per-case Dice scores for WT, TC, ET.

---

## TransBTS

Clone from the official repo and use the data scripts in `transbts_data/`:

```bash
git clone https://github.com/Rubics-Xuan/TransBTS.git
cd TransBTS
conda activate TransBTS
```

### Data preprocessing

```bash
# Convert BraTS NIfTI to .pkl (required by TransBTS data loader)
python transbts_data/preprocess.py \
    --root /path/to/BraTS2023 \
    --output_dir /path/to/TransBTS/data
```

The `train.txt` and `valid.txt` in `transbts_data/` define the train/validation split used in the paper (BraTS 2023 cases).

### Training

```bash
python train.py \
    --train_file data/train.txt \
    --valid_file data/valid.txt \
    --output_dir checkpoint/
```

### Inference

```bash
python predict.py \
    --test_file data/valid.txt \
    --model checkpoint/best_model.pth \
    --output_dir results/TransBTS_output/
```

Output: `TransBTS_summary.json` with per-case WT/TC/ET Dice.

---

## Output Format Expected by Analysis Notebooks

All downstream notebooks expect the Dice result files at:

```
results/
├── Result/
│   ├── Vanilla_Unet/
│   │   └── Unet_test_dice.csv          # columns: case_id, WT dice, TC dice, ET dice
│   ├── nnUnet/
│   │   └── nnUNetTrainerNoDA/
│   │       └── summary.json            # nnU-Net evaluation JSON
│   └── TransBTS/
│       └── submission/
│           └── TransBTS2023-11-03/
│               └── TransBTS_summary.json
```

Update the paths at the top of each notebook if your layout differs.
