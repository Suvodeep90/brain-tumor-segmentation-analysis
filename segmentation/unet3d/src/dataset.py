"""
BraTS dataset loader for 3D U-Net training and inference.

Expects the standard BraTS directory layout:
    <root>/
        BraTS20_Training_001/
            BraTS20_Training_001_flair.nii
            BraTS20_Training_001_t1.nii
            BraTS20_Training_001_t1ce.nii
            BraTS20_Training_001_t2.nii
            BraTS20_Training_001_seg.nii   (absent for test phase)
        BraTS20_Training_002/
            ...
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import albumentations as A
from albumentations import Compose
from sklearn.model_selection import StratifiedKFold


# ── Augmentations ─────────────────────────────────────────────────────────────

def get_augmentations(phase: str) -> Compose:
    transforms = [A.HorizontalFlip(p=0.5)] if phase == "train" else []
    return Compose(transforms, is_check_shapes=False)


# ── Dataset ───────────────────────────────────────────────────────────────────

class BratsDataset(Dataset):
    """
    Loads multi-modal MRI volumes and (optionally) segmentation masks.

    Args:
        df:         DataFrame with columns ['Brats20ID', 'path', 'fold']
        phase:      'train', 'val', or 'test'
        is_resize:  if True, volumes are resized to (78, 120, 120)
    """

    MODALITIES = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]

    def __init__(self, df: pd.DataFrame, phase: str = "train",
                 is_resize: bool = False):
        self.df           = df.reset_index(drop=True)
        self.phase        = phase
        self.augmentations = get_augmentations(phase)
        self.is_resize    = is_resize

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row       = self.df.iloc[idx]
        case_id   = row["Brats20ID"]
        root_path = row["path"]

        # Load and stack all four modalities
        images = []
        for mod in self.MODALITIES:
            img = self._load_nii(os.path.join(root_path, case_id + mod))
            if self.is_resize:
                img = self._resize(img)
            images.append(self._normalize(img))

        img = np.stack(images)                                  # (4, D, H, W)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))    # (4, W, H, D)

        if self.phase == "test":
            return {"Id": case_id, "image": img}

        # Load segmentation mask
        mask = self._load_nii(os.path.join(root_path, case_id + "_seg.nii"))
        if self.is_resize:
            mask = self._resize(mask)
            mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
        mask = self._preprocess_mask(mask)                      # (3, W, H, D)

        aug  = self.augmentations(image=img.astype(np.float32),
                                  mask=mask.astype(np.float32))
        return {"Id": case_id, "image": aug["image"], "mask": aug["mask"]}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_nii(path: str) -> np.ndarray:
        return np.asarray(nib.load(path).dataobj)

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        dmin = data.min()
        return (data - dmin) / (data.max() - dmin + 1e-9)

    @staticmethod
    def _resize(data: np.ndarray) -> np.ndarray:
        return resize(data, (78, 120, 120), preserve_range=True)

    @staticmethod
    def _preprocess_mask(mask: np.ndarray) -> np.ndarray:
        """Convert BraTS label map (0/1/2/4) → three binary channels (WT/TC/ET)."""
        wt = ((mask == 1) | (mask == 2) | (mask == 4)).astype(np.float32)
        tc = ((mask == 1) | (mask == 4)).astype(np.float32)
        et = (mask == 4).astype(np.float32)
        out = np.stack([wt, tc, et])                           # (3, D, H, W)
        return np.moveaxis(out, (0, 1, 2, 3), (0, 3, 2, 1))   # (3, W, H, D)


# ── CSV builder ───────────────────────────────────────────────────────────────

def build_train_csv(train_root_dir: str, out_csv: str,
                    n_folds: int = 6, seed: int = 55) -> pd.DataFrame:
    """
    Scan the BraTS training directory, merge survival / name-mapping CSVs,
    assign stratified k-fold labels, and write the result to *out_csv*.
    """
    survival_df     = pd.read_csv(os.path.join(train_root_dir, "survival_info.csv"))
    name_mapping_df = pd.read_csv(os.path.join(train_root_dir, "name_mapping.csv"))
    name_mapping_df = name_mapping_df.rename(
        {"BraTS_2020_subject_ID": "Brats20ID"}, axis=1
    )

    df = survival_df.merge(name_mapping_df, on="Brats20ID", how="left")
    df["path"]    = df["Brats20ID"].apply(lambda x: os.path.join(train_root_dir, x))
    df["Age_bin"] = pd.cut(df["Age"].fillna(df["Age"].median()), bins=4, labels=False)

    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["Age_bin"])):
        df.loc[val_idx, "fold"] = fold

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} cases → {out_csv}")
    return df


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloader(path_to_csv: str, phase: str, fold: int = 0,
                   batch_size: int = 1, num_workers: int = 4) -> DataLoader:
    df = pd.read_csv(path_to_csv)
    if phase == "train":
        split_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        shuffle  = True
    else:
        split_df = df.loc[df["fold"] == fold].reset_index(drop=True)
        shuffle  = False
    print(f"{phase}: {len(split_df)} cases  (fold {fold})")
    return DataLoader(
        BratsDataset(split_df, phase),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
    )
