# 5 — Oracle Model

Trains a Gradient Boosting Regressor on the 258 significant features to
predict per-case Dice scores, establishing an upper bound on explainability.

| File | Purpose |
|---|---|
| `Oracle_Model_Regression.ipynb` | Main oracle (GBR) — cross-validated Dice prediction |
| `Oracle_Model_Classification.ipynb` | Binary classifier variant (concordant-poor detection) |
| `Oracle_Model_DNN.ipynb` | MLP variant |
| `Oracle_Model_with_HPO.ipynb` | GBR with Bayesian HPO |
| `hpo.py` | HPO utility functions |
| `feature_analysis/` | Ablation: does using only significant features matter? (Appendix D) |
| `subregion_prediction/` | Per-subregion (WT/TC/ET) Dice prediction (Appendix E) |
| `probabilistic_comparison/` | Oracle vs probabilistic model baseline |
