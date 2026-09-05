# Brain Tumor Segmentation Analysis

Code for the paper **"When Neural Segmentation Models Stumble: Radiomics and Model-Based Signatures of Brain Tumor Segmentation Failure"**.

We study what makes deep-learning segmentation succeed or fail on BraTS scans by comparing concordant-good and concordant-poor cases across three models (3D U-Net, nnU-Net, TransBTS), extracting a rich set of radiomic features, and training an oracle model to predict Dice scores from those features alone.

All experiments use the **BraTS 2020** and **BraTS 2023** datasets.

---

## Repository Layout

```
brain-tumor-segmentation-analysis/
│
├── segmentation/                   # Model training & inference
│   ├── unet3d/
│   │   ├── src/                    # Reusable Python package
│   │   │   ├── model.py            – UNet3d architecture (GroupNorm, trilinear up)
│   │   │   ├── dataset.py          – BratsDataset, k-fold CSV builder
│   │   │   ├── losses.py           – BCE+Dice loss, Dice/IoU metrics
│   │   │   └── __init__.py
│   │   ├── Train_UNet3D.ipynb      – Full 6-fold cross-validation training loop
│   │   ├── Run_Inference.ipynb     – Batch inference → per-case Dice scores
│   │   └── Individual_Model_Analysis.ipynb
│   ├── nnunet/                     # nnUNet v2 — CLI commands & config notes
│   ├── transbts/                   # TransBTS data preparation scripts
│   │   ├── BraTS.py                – TransBTS dataset loader
│   │   ├── preprocess.py           – NIfTI → pkl conversion
│   │   ├── train.txt               – Train split (BraTS 2023 case IDs)
│   │   └── valid.txt               – Validation split
│   └── environments/               # Conda environment files
│       ├── environment_pytorch.yml – Analysis & 3D U-Net environment
│       ├── environment_nnunet.yml  – nnUNet v2 (CUDA 11.8)
│       └── environment_TransBTS.yml
│
├── feature_extraction/             # Radiomic feature computation
│   ├── shape/                      # Volume, curvature
│   │   ├── Calculate_Volume.ipynb
│   │   └── Calculate_Curvature.ipynb
│   ├── intensity/                  # First-order histogram, spatial position
│   │   ├── Calculate_Intensity.ipynb
│   │   └── Calculate_Position.ipynb
│   ├── texture/                    # GLCM, GLDM, GLRLM, GLSZM, NGTDM
│   │   ├── Calculate_Texture.ipynb
│   │   └── calculate_texture.py
│   ├── model_based/                # Probability map & gradient-saliency features
│   │   └── Calculate_Probability_Features.ipynb
│   └── radiomics/                  # PyRadiomics baseline
│       ├── pyradiomics.ipynb
│       ├── Process_Radiomics_Features.ipynb
│       └── Radiomics_Shape_WT_TC_ET.ipynb
│
├── statistical_analysis/           # Feature significance & selection
│   ├── feature_selection/
│   │   ├── feature_selection.py    – Mann-Whitney U test + Cliff's delta
│   │   └── Selected_Features.ipynb – Final ranked feature inventory
│   └── correlation/
│       ├── Feature_Correlations.ipynb
│       └── Feature_Correlation_Analysis.ipynb
│
├── oracle_model/                   # Dice score prediction
│   ├── training/
│   │   ├── Oracle_Model_Regression.ipynb    – Gradient Boosting (main oracle)
│   │   ├── Oracle_Model_Classification.ipynb
│   │   ├── Oracle_Model_DNN.ipynb
│   │   ├── Oracle_Model_with_HPO.ipynb
│   │   └── hpo.py
│   ├── ablation/
│   │   ├── Oracle_Feature_Analysis.ipynb
│   │   ├── Oracle_Feature_Ablation_Detailed.ipynb
│   │   └── oracle_improvement.py
│   ├── subregion_prediction/
│   │   ├── Oracle_Subregion_Prediction.ipynb
│   │   └── oracle_subregion_all_features.py
│   └── probabilistic_comparison/
│       └── Oracle_Probabilistic_Comparison.ipynb
│
├── figures/                        # Publication figure generation
│   ├── main_paper/
│   │   ├── shape/Shape_Generate_Charts.ipynb
│   │   ├── intensity/Hist_Based_Features_Generate_Charts.ipynb
│   │   ├── model_based/ModelBased_Generate_Charts.ipynb
│   │   └── texture/{GLCM,GLDM,GLRLM,GLSZM,NGTDM}_Generate_Charts.ipynb
│   └── supplementary/
│       ├── intensity_subregion/Firstorder_Subregion_Generate_Charts.ipynb
│       └── texture_subregion/Texture_Subregion_Generate_Charts.ipynb
│
└── visualization/                  # Exploratory visualizations
    ├── tSNE_Plot.ipynb             – t-SNE of good vs. poor cases
    └── Visualize_MRI_Images.ipynb  – MRI/mask overlay
```

---

## Environments

Three separate Conda environments are used:

```bash
# Analysis, feature extraction, oracle, figures
conda env create -f segmentation/environments/environment_pytorch.yml
conda activate pytorch

# nnUNet v2
conda env create -f segmentation/environments/environment_nnunet.yml
conda activate nnunet

# TransBTS
conda env create -f segmentation/environments/environment_TransBTS.yml
conda activate TransBTS
```

---

## Data

| Dataset | Purpose | Link |
|---------|---------|------|
| BraTS 2020 | 3D U-Net training & feature extraction | [Synapse](https://www.synapse.org/#!Synapse:syn25829067/wiki/612080) |
| BraTS 2023 | nnUNet & TransBTS inference | [Synapse](https://www.synapse.org/#!Synapse:syn51156910/wiki/621615) |

Expected layout (`root_dir` in each notebook points here):

```
data/
├── BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
│   └── BraTS20_Training_001/{flair,t1,t1ce,t2,seg}.nii.gz
└── BraTS2023_TrainingData/
    └── ...
```

---

## Reproducing the Pipeline

### 1  Train 3D U-Net (BraTS 2020)
```bash
conda activate pytorch
jupyter nbconvert --to notebook --execute \
    segmentation/unet3d/Train_UNet3D.ipynb
```

### 2  Run nnUNet (BraTS 2023)
```bash
conda activate nnunet
export nnUNet_raw="data/brats2023/raw"
export nnUNet_preprocessed="data/brats2023/preprocessed"
export nnUNet_results="results/nnunet"

nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
nnUNetv2_train 001 3d_fullres 0   # repeat for folds 1–4
nnUNetv2_predict \
    -i data/brats2023/raw/imagesTs \
    -o results/nnunet/predictions \
    -d 001 -c 3d_fullres --save_probabilities
```

### 3  Run TransBTS (BraTS 2023)
```bash
conda activate TransBTS
python segmentation/transbts/preprocess.py
python train.py \
    --train_file segmentation/transbts/train.txt \
    --valid_file segmentation/transbts/valid.txt
```

### 4  Extract Features
```bash
conda activate pytorch
# Run any notebook in feature_extraction/ — each is self-contained
jupyter nbconvert --to notebook --execute feature_extraction/shape/Calculate_Volume.ipynb
# ... repeat for intensity, texture, model_based, radiomics
```

### 5  Statistical Analysis
```bash
conda activate pytorch
python statistical_analysis/feature_selection/feature_selection.py
jupyter nbconvert --to notebook --execute \
    statistical_analysis/correlation/Feature_Correlations.ipynb
```

### 6  Oracle Model
```bash
conda activate pytorch
jupyter nbconvert --to notebook --execute \
    oracle_model/training/Oracle_Model_Regression.ipynb
```

### 7  Generate Figures
```bash
conda activate pytorch
for nb in figures/main_paper/**/*.ipynb figures/supplementary/**/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb"
done
```

---

## Citation

```bibtex
@article{majumder2025brain,
  title   = {When Neural Segmentation Models Stumble: Radiomics and Model-Based
             Signatures of Brain Tumor Segmentation Failure},
  author  = {Majumder, Suvodeep and others},
  year    = {2025}
}
```
