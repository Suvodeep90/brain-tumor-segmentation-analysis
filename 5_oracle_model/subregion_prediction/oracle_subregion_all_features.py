"""
Oracle Subregion Prediction — All Features
Runs 5-fold CV GBR for 3 segmentation models × 3 subregions (WT, TC, ET)
using all 824 radiomics + model-based features.
Saves:
  Results/Figures/oracle_subregion_all_features.pdf
  Results/Json_summary/oracle_subregion_all_features.json
"""

import os, json, warnings
import numpy as np, pandas as pd, pickle as pkl
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings('ignore')

AR   = Path('/Users/suvodeepmajumder/Document/Columbia/Results/Analysis_Results')
RES  = Path('/Users/suvodeepmajumder/Document/Columbia/Results/Result')
FIGS = Path('/Users/suvodeepmajumder/Document/Columbia/Results/Figures')
JSON = Path('/Users/suvodeepmajumder/Document/Columbia/Results/Json_summary')

# ── Load all radiomics features ───────────────────────────────────────────────
print("Loading radiomics features (Tumor_WT)...")
RADIOMICS_DIR = AR / 'Radiomics/Tumor_WT'
analysis_types = sorted([f.replace('.pkl','') for f in os.listdir(RADIOMICS_DIR) if f.endswith('.pkl')])
frames = []
for atype in analysis_types:
    with open(RADIOMICS_DIR / f'{atype}.pkl', 'rb') as f:
        raw = pkl.load(f)
    for prop_name, patient_dict in raw.items():
        df = pd.DataFrame.from_dict(patient_dict, orient='index').astype(float)
        df.columns = [f'{prop_name}_{c}_{atype}' for c in df.columns]
        frames.append(df)
radiomics_df = frames[0]
for d in frames[1:]: radiomics_df = radiomics_df.join(d, how='outer')
print(f"  Radiomics: {radiomics_df.shape}")

# ── Load model-based features ─────────────────────────────────────────────────
print("Loading model-based features...")
vol  = pd.read_csv(AR/'volume/GLI-Tumor_volumns.csv', index_col='Unnamed: 0')[
           ['ED','ET','NCR','WT_volume','TC_volume','ET_volume','TC_WT_ratio','ET_WT_ratio','ET_TC_ratio']]
prob = pd.read_csv(AR/'probability/Probability_Tumor_boundary.csv', index_col='Unnamed: 0')
curv = pd.read_csv(AR/'curverature/curverature.csv', index_col='Unnamed: 0')[
           ['mean_gaussian_curvature','std_gaussian_curvature','pos','neg','pos_count','neg_count']]
sal  = pd.read_csv(AR/'Saliency/Saliency.csv', index_col='Unnamed: 0')
model_based_df = vol.join(prob, how='outer').join(curv, how='outer').join(sal, how='outer')
print(f"  Model-based: {model_based_df.shape}")

X_full = radiomics_df.join(model_based_df, how='inner').dropna()
print(f"  Combined after dropna: {X_full.shape}")

# ── Load all segmentation model performances ──────────────────────────────────
print("Loading segmentation model results...")

def load_unet(path):
    df = pd.read_csv(path, index_col='Unnamed: 0')
    df.index = [i.split('-seg')[0] for i in df.index]
    return df[['WT dice', 'TC dice', 'ET dice']]

def load_nnunet(path):
    with open(path) as f: data = json.load(f)
    rows = []
    for c in data['metric_per_case']:
        name = c['reference_file'].split('/')[-1].split('.')[0]
        rows.append({'index': name,
                     'WT dice': c['metrics']['(2, 1, 3)']['Dice'],
                     'TC dice': c['metrics']['(2, 3)']['Dice'],
                     'ET dice': c['metrics']['(3,)']['Dice']})
    return pd.DataFrame(rows).set_index('index')

def load_transbts(path):
    with open(path) as f: data = json.load(f)
    rows = [{'index': k, 'WT dice': v['WT'][0], 'TC dice': v['TC'][0], 'ET dice': v['ET'][0]}
            for k, v in data.items()]
    return pd.DataFrame(rows).set_index('index')

perf = {
    'U-Net':   load_unet(RES/'Vanilla_Unet/Unet_test_dice.csv'),
    'nnU-Net': load_nnunet(RES/'nnUnet/nnUNetTrainerNoDA/summary.json'),
    'TransBTS':load_transbts(RES/'TransBTS/submission/TransBTS2023-11-03/TransBTS_summary.json'),
}
for nm, df in perf.items():
    print(f"  {nm}: {len(df)} patients, cols={df.columns.tolist()}")

# ── 5-fold CV helper ──────────────────────────────────────────────────────────
GBR_PARAMS = dict(n_estimators=150, learning_rate=0.10, max_depth=5,
                  subsample=0.75, min_samples_split=3, min_samples_leaf=2,
                  random_state=42)

def run_cv(X_df, y_s, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, r2s, mses = [], [], []
    y_true_all, y_pred_all = [], []
    for tr, te in kf.split(X_df):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_df.iloc[tr]); Xte = sc.transform(X_df.iloc[te])
        ytr, yte = y_s.iloc[tr], y_s.iloc[te]
        m = GradientBoostingRegressor(**GBR_PARAMS)
        m.fit(Xtr, ytr); yp = m.predict(Xte)
        maes.append(mean_absolute_error(yte, yp))
        r2s.append(r2_score(yte, yp))
        mses.append(mean_squared_error(yte, yp))
        y_true_all.extend(yte.tolist()); y_pred_all.extend(yp.tolist())
    return (np.mean(maes), np.std(maes),
            np.mean(r2s),  np.std(r2s),
            np.mean(mses), np.std(mses),
            np.array(y_true_all), np.array(y_pred_all))

# ── Run all 9 combinations ────────────────────────────────────────────────────
print("\nRunning 5-fold CV (3 models × 3 subregions = 9 experiments)...")
subregions = ['WT', 'TC', 'ET']
seg_models = ['U-Net', 'nnU-Net', 'TransBTS']

all_results = {}
scatter_data = {}

for seg_model in seg_models:
    all_results[seg_model] = {}
    scatter_data[seg_model] = {}
    df_perf = perf[seg_model]
    common = X_full.index.intersection(df_perf.index)
    X = X_full.loc[common]
    print(f"\n  {seg_model} ({len(common)} patients):")
    for subr in subregions:
        target_col = f'{subr} dice'
        y = df_perf.loc[common, target_col]
        # drop NaN targets (cases with zero-volume TC/ET → undefined dice)
        valid = y.notna()
        X_sub, y_sub = X.loc[valid], y.loc[valid]
        mae_m, mae_s, r2_m, r2_s, mse_m, mse_s, y_true, y_pred = run_cv(X_sub, y_sub)
        all_results[seg_model][subr] = {
            'MAE_mean': float(mae_m), 'MAE_std': float(mae_s),
            'R2_mean':  float(r2_m),  'R2_std':  float(r2_s),
            'MSE_mean': float(mse_m), 'MSE_std': float(mse_s),
            'n_features': int(X_sub.shape[1]), 'n_samples': int(valid.sum()),
        }
        scatter_data[seg_model][subr] = (y_true, y_pred)
        print(f"    {subr} (n={valid.sum()}): MAE={mae_m:.4f}±{mae_s:.4f}  R²={r2_m:.3f}±{r2_s:.3f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    'model': 'GradientBoostingRegressor',
    'hyperparameters': GBR_PARAMS,
    'feature_set': 'all_824_radiomics_plus_model_based',
    'cv': '5-fold KFold (shuffle=True, random_state=42)',
    'results': all_results,
}
json_path = JSON / 'oracle_subregion_all_features.json'
with open(json_path, 'w') as f: json.dump(output, f, indent=2)
print(f"\nJSON saved: {json_path}")

# ── Figure ────────────────────────────────────────────────────────────────────
COLORS = {'WT': '#1f77b4', 'TC': '#ff7f0e', 'ET': '#2ca02c'}
x = np.arange(len(subregions)); width = 0.25
seg_model_styles = ['steelblue', 'darkorange', 'seagreen']
seg_hatches = ['', '///', '...']

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Oracle GBR: Per-Subregion Dice Prediction\n(5-fold CV, GBR)",
             fontsize=13, fontweight='bold')

for col_i, seg_model in enumerate(seg_models):
    # ── Row 0: MAE ────────────────────────────────────────────────────────────
    ax = axes[0, col_i]
    maes = [all_results[seg_model][s]['MAE_mean'] for s in subregions]
    errs = [all_results[seg_model][s]['MAE_std']  for s in subregions]
    bar_colors = [COLORS[s] for s in subregions]
    ax.bar(x, maes, yerr=errs, color=bar_colors, capsize=5, alpha=0.85, width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(subregions, fontsize=11)
    ax.set_ylabel("MAE", fontsize=10); ax.set_title(f"{seg_model}", fontsize=11, fontweight='bold')
    for i, (m, e) in enumerate(zip(maes, errs)):
        ax.text(i, m + e + 0.001, f'{m:.4f}', ha='center', fontsize=8.5)
    if col_i == 0: ax.set_ylabel("MAE ↓", fontsize=10)

    # ── Row 1: R² ─────────────────────────────────────────────────────────────
    ax = axes[1, col_i]
    r2s  = [all_results[seg_model][s]['R2_mean'] for s in subregions]
    r2e  = [all_results[seg_model][s]['R2_std']  for s in subregions]
    ax.bar(x, r2s, yerr=r2e, color=bar_colors, capsize=5, alpha=0.85, width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(subregions, fontsize=11)
    ax.set_title(f"{seg_model}", fontsize=11, fontweight='bold')
    for i, (m, e) in enumerate(zip(r2s, r2e)):
        ax.text(i, m + e + 0.005, f'{m:.3f}', ha='center', fontsize=8.5)
    if col_i == 0: ax.set_ylabel("R² ↑", fontsize=10)

# Row labels
axes[0, 0].annotate('MAE', xy=(-0.22, 0.5), xycoords='axes fraction',
                    rotation=90, ha='center', va='center', fontsize=12, fontweight='bold')
axes[1, 0].annotate('R²', xy=(-0.22, 0.5), xycoords='axes fraction',
                    rotation=90, ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
fig_path = FIGS / 'oracle_subregion_all_features.pdf'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
print(f"Figure saved: {fig_path}")

# Also copy to paper Figs/
import shutil
paper_figs = Path('/Users/suvodeepmajumder/Document/Columbia/Papers/Brain Tumor Segmentation/Figs')
shutil.copy(fig_path, paper_figs / 'oracle_subregion_all_features.pdf')
print(f"Copied to {paper_figs / 'oracle_subregion_all_features.pdf'}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n=== RESULTS TABLE ===")
print(f"{'Seg. Model':<12} {'Subregion':<10} {'MAE':>8} {'±std':>7}  {'R²':>7} {'±std':>7}")
print("-"*58)
for seg_model in seg_models:
    for subr in subregions:
        r = all_results[seg_model][subr]
        print(f"  {seg_model:<10} {subr:<10} {r['MAE_mean']:.4f}  {r['MAE_std']:.4f}  "
              f"{r['R2_mean']:.4f}  {r['R2_std']:.4f}")
