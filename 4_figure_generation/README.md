# 4 — Figure Generation

Reproduces every figure in the paper and supplementary appendix.
Each notebook is self-contained: it loads pre-computed CSVs, runs statistics,
and writes PDFs to `results/final_figures/`.

| Subfolder | Paper location |
|---|---|
| `shape/` | Fig 1 — shape features |
| `intensity_wt/` | Fig 2 — first-order intensity (WT) |
| `model_based/` | Fig 3 — saliency & probability; Appendix Figs 1–2 |
| `texture_wt/` | Fig 4 — GLCM/GLDM/GLRLM/GLSZM/NGTDM (WT) |
| `intensity_subregion/` | Appendix Figs 3–4 — first-order per subregion |
| `texture_subregion/` | Appendix Figs 5–24 — texture per subregion |
