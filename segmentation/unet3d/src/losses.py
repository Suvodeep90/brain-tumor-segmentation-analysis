"""
Loss functions and evaluation metrics for 3D brain tumor segmentation.

Losses
------
DiceLoss      — soft Dice computed on sigmoid probabilities
BCEDiceLoss   — BCE + Dice (primary training objective)

Metrics
-------
dice_coef_metric        — mean Dice over a batch
jaccard_coef_metric     — mean IoU (Jaccard) over a batch
dice_coef_metric_per_classes — per-class Dice for WT / TC / ET
Meter                   — accumulates batch-level scores across an epoch
"""

import numpy as np
import torch
import torch.nn as nn


# ── Losses ────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss computed on flattened sigmoid probabilities."""

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n     = targets.size(0)
        probs = torch.sigmoid(logits).view(n, -1)
        tgt   = targets.view(n, -1)
        inter = 2.0 * (probs * tgt).sum()
        union = probs.sum() + tgt.sum()
        return 1.0 - (inter + self.eps) / union


class BCEDiceLoss(nn.Module):
    """Combined BCE + soft Dice loss (primary training criterion)."""

    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape, \
            f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}"
        return self.bce(logits, targets) + self.dice(logits, targets)


# ── Metrics ───────────────────────────────────────────────────────────────────

def dice_coef_metric(probs: torch.Tensor, truth: torch.Tensor,
                     threshold: float = 0.5, eps: float = 1e-9) -> float:
    """Mean Dice score over a batch."""
    preds  = (probs >= threshold).float()
    scores = []
    for i in range(preds.shape[0]):
        inter = 2.0 * (truth[i] * preds[i]).sum()
        union = truth[i].sum() + preds[i].sum()
        scores.append(1.0 if (truth[i].sum() == 0 and preds[i].sum() == 0)
                      else float((inter + eps) / union))
    return float(np.mean(scores))


def jaccard_coef_metric(probs: torch.Tensor, truth: torch.Tensor,
                        threshold: float = 0.5, eps: float = 1e-9) -> float:
    """Mean IoU (Jaccard) score over a batch."""
    preds  = (probs >= threshold).float()
    scores = []
    for i in range(preds.shape[0]):
        inter = (preds[i] * truth[i]).sum()
        union = preds[i].sum() + truth[i].sum() - inter + eps
        scores.append(1.0 if (truth[i].sum() == 0 and preds[i].sum() == 0)
                      else float((inter + eps) / union))
    return float(np.mean(scores))


def dice_coef_metric_per_classes(
    probs: np.ndarray,
    truth: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-9,
    classes: list = ["WT", "TC", "ET"],
) -> dict:
    """Per-class Dice scores for a batch (used during evaluation)."""
    scores  = {k: [] for k in classes}
    preds   = (probs >= threshold).astype(np.float32)
    n_cls   = probs.shape[1]
    assert n_cls == len(classes), f"Expected {len(classes)} classes, got {n_cls}"

    for i in range(probs.shape[0]):
        for j, cls in enumerate(classes):
            inter = 2.0 * (truth[i, j] * preds[i, j]).sum()
            union = truth[i, j].sum() + preds[i, j].sum()
            if truth[i, j].sum() == 0 and preds[i, j].sum() == 0:
                scores[cls].append(1.0)
            else:
                scores[cls].append(float((inter + eps) / union))
    return scores


class Meter:
    """Accumulates Dice and IoU over batches within one epoch."""

    def __init__(self, threshold: float = 0.5):
        self.threshold   = threshold
        self.dice_scores: list = []
        self.iou_scores:  list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        self.dice_scores.append(dice_coef_metric(probs, targets, self.threshold))
        self.iou_scores.append(jaccard_coef_metric(probs, targets, self.threshold))

    def get_metrics(self) -> tuple:
        """Returns (mean_dice, mean_iou) over all accumulated batches."""
        return float(np.mean(self.dice_scores)), float(np.mean(self.iou_scores))
