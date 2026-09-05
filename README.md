# When Neural Segmentation Models Stumble — Code Repository

This repository contains the complete analysis code for the paper
**"When Neural Segmentation Models Stumble: Radiomics and Model-Based Signatures of Brain Tumor Segmentation Failure"**.

All experiments use the BraTS 2020 and BraTS 2023 datasets.
The three segmentation models evaluated are: **3D U-Net**, **nnU-Net**, and **TransBTS**.

---

## Repository Structure

```
brain-tumor-segmentation-analysis/
│
├── 0_model_setup/                  # Environment files + inference setup for all 3 models
│   ├── README.md                       – Step-by-step instructions for 3D U-Net, nnU-Net, TransBTS
│   ├── environments/
│   │   ├── environment_pytorch.yml     – Analysis & feature extraction environment
│   │   ├── environment_nnunet.yml      – nnU-Net v2 inference environment (CUDA 11.8)
│   │   └── environment_TransBTS.yml   – TransBTS inference environment
│   └── transbts_data/
│       ├── BraTS.py                    – TransBTS BraTS dataset loader
│       ├── preprocess.py               – NIfTI → pkl conversion for TransBTS
│       ├── train.txt                   – Train split (BraTS 2023 case IDs used in paper)
│       └── valid.txt                   – Validation split
│
├── 1_segmentation_models/          # CNN model inference
│   └── unet_3d/
│       ├── Run_Inference.ipynb         – Run the trained 3D U-Net on BraTS test cases;
│       │                                 outputs per-case Dice scores
│       └── Individual_Model.ipynb      – Single-model Dice analysis and overlap selection
│
├── 2_feature_extraction/           # Extract features from MRI + segmentation masks
│   ├── shape/
│   │   ├── Calculate_Volume.ipynb      – Volume and surface area of tumor subregions
│   │   └── Calculate_Curvature.ipynb   – Local curvature of the tumor surface (trimesh)
│   ├── intensity/
│   │   ├── Calculate_Intensity.ipynb   – First-order intensity statistics (mean, entropy, …)
│   │   └── Calculate_Position.ipynb    – Tumor centroid and spatial position features
│   ├── texture/
│   │   ├── Calculate_Texture.ipynb     – GLCM / GLDM / GLRLM / GLSZM / NGTDM textures
│   │   └── calculate_texture.py        – Standalone script version of the above
│   ├── model_based/
│   │   └── Calculate_Probability_Features.ipynb
│   │                                   – Softmax probability and gradient-saliency features
│   │                                     extracted from the U-Net's output layer
│   └── radiomics/
│       ├── pyradiomics.ipynb           – PyRadiomics feature extraction setup
│       ├── Process_Radiomics_Features.ipynb – Post-process / clean raw radiomics CSVs
│       └── Radiomics_Shape_WT_TC_ET.ipynb   – Shape features per tumor subregion (WT/TC/ET)
│
├── 3_statistical_analysis/         # Feature selection and correlation
│   ├── feature_selection.py            – Mann-Whitney U test + Cliff's delta filtering
│   ├── Feature_Correlations.ipynb      – Pairwise correlation heatmaps
│   ├── Selected_Features.ipynb         – Final significant-feature inventory
│   └── feature_correlation/
│       └── Feature_Correlation_Analysis.ipynb – In-depth correlation / redundancy analysis
│
├── 4_figure_generation/            # Reproduce every paper figure
│   ├── shape/
│   │   └── Shape_Generate_Charts.ipynb              – Fig 1 (shape features)
│   ├── intensity_wt/
│   │   └── Hist_Based_Features_Generate_Charts.ipynb – Fig 2 (first-order, WT only)
│   ├── texture_wt/
│   │   ├── GLCM_Generate_Charts.ipynb               – Fig 4a (GLCM)
│   │   ├── GLDM_Generate_Charts.ipynb               – Fig 4b (GLDM)
│   │   ├── GLRLM_Generate_Charts.ipynb              – Fig 4c (GLRLM)
│   │   ├── GLSZM_Generate_Charts.ipynb              – Fig 4d (GLSZM)
│   │   └── NGTDM_Generate_Charts.ipynb              – Fig 4e (NGTDM)
│   ├── model_based/
│   │   └── ModelBased_Generate_Charts.ipynb         – Fig 3 (saliency + probability)
│   │                                                  Appendix Figs 1–2
│   ├── intensity_subregion/
│   │   └── Firstorder_Subregion_Generate_Charts.ipynb – Appendix Figs 3–4 (intensity WT/TC/ET)
│   └── texture_subregion/
│       └── Texture_Subregion_Generate_Charts.ipynb  – Appendix Figs 5–24 (texture WT/TC/ET)
│
├── 5_oracle_model/                 # Oracle Dice-prediction model
│   ├── Oracle_Model_Regression.ipynb   – Gradient Boosting Regressor (main oracle)
│   ├── Oracle_Model_Classification.ipynb – Concordant-poor classifier variant
│   ├── Oracle_Model_DNN.ipynb          – Deep neural network oracle variant
│   ├── Oracle_Model_with_HPO.ipynb     – Oracle with Bayesian hyperparameter search
│   ├── hpo.py                          – HPO helper utilities
│   ├── feature_analysis/
│   │   ├── Oracle_Feature_Analysis.ipynb        – Were only significant features used? (Appendix D)
│   │   ├── Oracle_Feature_Ablation_Detailed.ipynb – Full feature-subset ablation
│   │   └── oracle_improvement.py                – Utility functions for ablation
│   ├── subregion_prediction/
│   │   ├── Oracle_Subregion_Prediction.ipynb    – Per-subregion Dice prediction (Appendix E)
│   │   └── oracle_subregion_all_features.py     – Script version for batch runs
│   └── probabilistic_comparison/
│       └── Oracle_Probabilistic_Comparison.ipynb – Oracle vs probabilistic baseline comparison
│
└── 6_visualization/                # Supplementary visualizations
    ├── tSNE_Plot.ipynb                 – t-SNE embedding of good vs poor cases
    └── Visualize_MRI_Images.ipynb      – MRI/mask overlay visualizations
```

---

## Setup

Each model runs in a dedicated conda environment; the analysis notebooks
share a single environment. See `0_model_setup/README.md` for full details.

```bash
# Analysis notebooks (feature extraction, oracle, figures)
conda env create -f 0_model_setup/environments/environment_pytorch.yml
conda activate pytorch

# nnU-Net inference  →  pip install nnunetv2==2.2
conda env create -f 0_model_setup/environments/environment_nnunet.yml

# TransBTS inference
conda env create -f 0_model_setup/environments/environment_TransBTS.yml
```

---

## Data

The BraTS datasets are publicly available after registration:

- **BraTS 2020**: https://www.synapse.org/#!Synapse:syn25829067/wiki/612080
- **BraTS 2023**: https://www.synapse.org/#!Synapse:syn51156910/wiki/621615

Expected directory layout (update `root_dir` in each notebook):

```
data/
├── BraTS2020_TrainingData/
│   └── MICCAI_BraTS2020_TrainingData/
│       └── BraTS20_Training_001/
│           ├── BraTS20_Training_001_flair.nii.gz
│           ├── BraTS20_Training_001_t1.nii.gz
│           ├── BraTS20_Training_001_t1ce.nii.gz
│           ├── BraTS20_Training_001_t2.nii.gz
│           └── BraTS20_Training_001_seg.nii.gz
└── BraTS2023_TrainingData/
    └── ...
```

---

## Reproducing the Paper Results

Run the notebooks **in order**:

1. **`0_model_setup/`** — set up environments and run segmentation model inference
2. **`1_segmentation_models/`** — (3D U-Net specific) run inference to get Dice scores per case
2. **`2_feature_extraction/`** — extract all feature families
3. **`3_statistical_analysis/`** — Mann-Whitney U + Cliff's delta feature selection
4. **`4_figure_generation/`** — generate all paper and appendix figures
5. **`5_oracle_model/`** — train and evaluate the oracle Dice-prediction model

Each notebook documents its input CSV path and output location at the top.
Outputs are written to `results/` (created automatically).

---

## Citation

```
@article{majumder2025brain,
  title   = {When Neural Segmentation Models Stumble: Radiomics and Model-Based
             Signatures of Brain Tumor Segmentation Failure},
  author  = {Majumder, Suvodeep and others},
  year    = {2025}
}
```
