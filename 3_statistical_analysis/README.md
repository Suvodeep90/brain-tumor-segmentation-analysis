# 3 — Statistical Analysis

Filters features to those significantly different between concordant-good and
concordant-poor segmentation cases using Mann-Whitney U (p < 0.05) and
Cliff's delta effect size.

| File | Purpose |
|---|---|
| `feature_selection.py` | Core selection logic; callable as a module |
| `Feature_Correlations.ipynb` | Pairwise Pearson/Spearman correlation heatmaps |
| `Selected_Features.ipynb` | Final inventory of 258 curated features used by the Oracle |
| `feature_correlation/` | Redundancy analysis to identify highly correlated feature pairs |
