"""
Oracle Model Improvement Exploration
Experiments:
  Exp 1: Compare 7 ML models (no HPO) on all 824 features
  Exp 2: Feature selection × top 3 models
  Exp 3: HPO (Hyperopt) for XGBoost and ExtraTrees
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path

warnings.filterwarnings('ignore')

# ── paths ───────────────────────────────────────────────────────────────────
RADIOMICS_DIR  = Path("/Users/suvodeepmajumder/Document/Columbia/Results/Analysis_Results/Radiomics/Tumor_WT")
SUMMARY_CSV    = Path("/Users/suvodeepmajumder/Document/Columbia/Results/Analysis_Results/Radiomics/summary/Tumor_WT_summary.csv")
MODEL_RESULTS  = Path("/Users/suvodeepmajumder/Document/Columbia/Results/Analysis_Results")
RESULT_JSON    = Path("/Users/suvodeepmajumder/Document/Columbia/Results/Json_summary/oracle_model_improvement.json")
FIGURE_PATH    = Path("/Users/suvodeepmajumder/Document/Columbia/Results/Figures/oracle_model_improvement.pdf")

# ── segmentation model dice scores ──────────────────────────────────────────
def load_unet(path):
    df = pd.read_csv(path, index_col='Unnamed: 0')
    df.index = [x.split('-seg')[0] for x in df.index]
    return df[['WT dice']]

def load_nnunet(path):
    with open(path) as f: data = json.load(f)
    rows = {}
    for c in data['metric_per_case']:
        name = c['reference_file'].split('/')[-1].split('.')[0]
        rows[name] = c['metrics']['(2, 1, 3)']['Dice']
    return pd.DataFrame.from_dict(rows, orient='index', columns=['WT dice'])

def load_transbts(path):
    with open(path) as f: data = json.load(f)
    rows = {k: v['WT'][0] for k, v in data.items()}
    return pd.DataFrame.from_dict(rows, orient='index', columns=['WT dice'])

print("Loading segmentation model results...")
unet_df   = load_unet("/Users/suvodeepmajumder/Document/Columbia/Results/Result/Vanilla_Unet/Unet_test_dice.csv")
nnunet_df = load_nnunet("/Users/suvodeepmajumder/Document/Columbia/Results/Result/nnUnet/nnUNetTrainerNoDA/summary.json")
trans_df  = load_transbts("/Users/suvodeepmajumder/Document/Columbia/Results/Result/TransBTS/submission/TransBTS2023-11-03/TransBTS_summary.json")
print(f"  UNet: {len(unet_df)}, nnUNet: {len(nnunet_df)}, TransBTS: {len(trans_df)}")

# ── load all radiomics features ──────────────────────────────────────────────
print("Loading radiomics features...")
import os
analysis_types = sorted([f.replace('.pkl', '') for f in os.listdir(RADIOMICS_DIR) if f.endswith('.pkl')])

frames = []
for atype in analysis_types:
    fpath = RADIOMICS_DIR / f'{atype}.pkl'
    with open(fpath, 'rb') as f:
        raw = pkl.load(f)
    n_feats = 0
    for prop_name, patient_dict in raw.items():
        df = pd.DataFrame.from_dict(patient_dict, orient='index').astype(float)
        df.columns = [f'{prop_name}_{col}_{atype}' for col in df.columns]
        frames.append(df)
        n_feats += df.shape[1]
    print(f"  {atype:15s}: {n_feats} features")

radiomics_df = frames[0]
for df in frames[1:]:
    radiomics_df = radiomics_df.join(df, how='outer')
print(f"  Total radiomics: {radiomics_df.shape}")

# ── load model-based features (CSV-based, same as Oracle_Feature_Ablation_Detailed) ─
print("Loading model-based features...")
volume_features   = ['ED', 'ET', 'NCR', 'WT_volume', 'TC_volume', 'ET_volume',
                     'TC_WT_ratio', 'ET_WT_ratio', 'ET_TC_ratio']
curvature_features = ['mean_gaussian_curvature', 'std_gaussian_curvature',
                      'pos', 'neg', 'pos_count', 'neg_count']

volume_df = pd.read_csv(MODEL_RESULTS / "volume" / "GLI-Tumor_volumns.csv",
                        index_col='Unnamed: 0')[volume_features]
prob_df   = pd.read_csv(MODEL_RESULTS / "probability" / "Probability_Tumor_boundary.csv",
                        index_col='Unnamed: 0')
curv_df   = pd.read_csv(MODEL_RESULTS / "curverature" / "curverature.csv",
                        index_col='Unnamed: 0')[curvature_features]
sal_df    = pd.read_csv(MODEL_RESULTS / "Saliency" / "Saliency.csv",
                        index_col='Unnamed: 0')

model_based_df = volume_df.join(prob_df, how='outer').join(curv_df, how='outer').join(sal_df, how='outer')
model_based_cols = list(model_based_df.columns)
print(f"  Model-based: {model_based_df.shape}")

# ── combine all features ─────────────────────────────────────────────────────
full_df = radiomics_df.join(model_based_df, how='inner')
full_df = full_df.loc[:, ~full_df.columns.duplicated()]
print(f"Full feature matrix before dropna: {full_df.shape}")

full_df.dropna(inplace=True)
# Sanitize column names for XGBoost (no [, ], <)
full_df.columns = [c.replace('[', '(').replace(']', ')').replace('<', '_') for c in full_df.columns]
print(f"Full feature matrix after dropna: {full_df.shape}")

# ── align with UNet target (primary) ────────────────────────────────────────
common_idx = full_df.index.intersection(unet_df.index)
X_full = full_df.loc[common_idx]
y_unet = unet_df.loc[common_idx, 'WT dice']
print(f"\nAligned samples (UNet): {len(common_idx)}, features: {X_full.shape[1]}")

# also align nnUNet and TransBTS
common_nn = full_df.index.intersection(nnunet_df.index)
X_nn = full_df.loc[common_nn]
y_nn = nnunet_df.loc[common_nn, 'WT dice']

common_tr = full_df.index.intersection(trans_df.index)
X_tr = full_df.loc[common_tr]
y_tr = trans_df.loc[common_tr, 'WT dice']

print(f"Aligned samples (nnUNet): {len(common_nn)}, (TransBTS): {len(common_tr)}")

# ── significance subsets ─────────────────────────────────────────────────────
print("\nBuilding significance subsets...")
summary = pd.read_csv(SUMMARY_CSV)
summary['Name'] = summary['Feature'] + '_' + summary['Modality'] + '_' + summary['Method']

def sanitize(s):
    return s.replace('[', '(').replace(']', ')').replace('<', '_')

sig_names   = set(sanitize(n) for n in summary[summary['P-value'] == '***']['Name'].unique())
large_names = set(sanitize(n) for n in summary[(summary['P-value'] == '***') &
                           (summary['Effect_Size'] == 'Large')]['Name'].unique())

radiomics_cols_in_X = [c for c in full_df.columns if c not in model_based_cols]
model_cols_in_X     = [c for c in model_based_cols if c in full_df.columns]

sig_cols   = [c for c in radiomics_cols_in_X if c in sig_names]   + model_cols_in_X
large_cols = [c for c in radiomics_cols_in_X if c in large_names] + model_cols_in_X
print(f"  Sig features: {len(sig_cols)}, Large effect: {len(large_cols)}")

# ── imports for ML ───────────────────────────────────────────────────────────
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

CV = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_scores(estimator, X, y):
    maes, r2s = [], []
    for tr, te in CV.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        estimator.fit(Xtr, ytr)
        pred = estimator.predict(Xte)
        maes.append(mean_absolute_error(yte, pred))
        r2s.append(r2_score(yte, pred))
    return np.mean(maes), np.std(maes), np.mean(r2s), np.std(r2s)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Exp 1: Model comparison — all 824 features, U-Net target
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Exp 1: Model comparison (all features, U-Net)")
print("="*60)

models_exp1 = {
    'GBR':          GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                              max_depth=4, random_state=42),
    'ExtraTrees':   ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost':      XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                 random_state=42, verbosity=0, n_jobs=-1),
    'AdaBoost':     AdaBoostRegressor(n_estimators=100, random_state=42),
    'SVR':          Pipeline([('scaler', StandardScaler()),
                               ('svr', SVR(kernel='rbf', C=1.0))]),
    'ElasticNet':   Pipeline([('scaler', StandardScaler()),
                               ('en', ElasticNet(alpha=0.01, l1_ratio=0.5,
                                                 max_iter=2000))]),
}

exp1_results = {}
for name, model in models_exp1.items():
    mae_m, mae_s, r2_m, r2_s = cv_scores(model, X_full, y_unet)
    exp1_results[name] = {'MAE_mean': float(mae_m), 'MAE_std': float(mae_s),
                          'R2_mean': float(r2_m),  'R2_std': float(r2_s)}
    print(f"  {name:15s}  MAE={mae_m:.4f}±{mae_s:.4f}  R²={r2_m:.3f}±{r2_s:.3f}")

results['exp1_model_comparison'] = exp1_results

# ═══════════════════════════════════════════════════════════════════════════════
# Exp 2: Feature selection × top 3 models (U-Net target)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Exp 2: Feature selection (U-Net)")
print("="*60)

# find best 3 from Exp 1 by MAE
sorted_models = sorted(exp1_results.items(), key=lambda x: x[1]['MAE_mean'])
top3_names = [m[0] for m in sorted_models[:3]]
print(f"  Top 3 models: {top3_names}")

def make_model(name):
    if name == 'GBR':
        return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                         max_depth=4, random_state=42)
    elif name == 'ExtraTrees':
        return ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif name == 'RandomForest':
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif name == 'XGBoost':
        return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                            random_state=42, verbosity=0, n_jobs=-1)
    elif name == 'AdaBoost':
        return AdaBoostRegressor(n_estimators=100, random_state=42)
    raise ValueError(name)

# Feature selection options
from sklearn.inspection import permutation_importance

def rf_top_pct(X, y, pct=0.25):
    """Return top pct% features by RF importance."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    thresh = imp.quantile(1 - pct)
    return imp[imp >= thresh].index.tolist()

def lasso_features(X, y):
    """Return features selected by Lasso (non-zero coef)."""
    scaler = StandardScaler()
    Xsc = scaler.fit_transform(X)
    lasso = Lasso(alpha=0.001, max_iter=5000)
    lasso.fit(Xsc, y)
    mask = lasso.coef_ != 0
    return X.columns[mask].tolist()

feature_sets = {
    'All':           list(X_full.columns),
    'KBest-F(200)':  None,   # computed below
    'KBest-MI(200)': None,
    'RF-top25%':     rf_top_pct(X_full, y_unet, 0.25),
    'Lasso':         lasso_features(X_full, y_unet),
    'Significant':   sig_cols,
    'LargeEffect':   large_cols,
}

# KBest
kb_f  = SelectKBest(f_regression, k=min(200, X_full.shape[1]))
kb_f.fit(X_full, y_unet)
feature_sets['KBest-F(200)'] = X_full.columns[kb_f.get_support()].tolist()

kb_mi = SelectKBest(mutual_info_regression, k=min(200, X_full.shape[1]))
kb_mi.fit(X_full, y_unet)
feature_sets['KBest-MI(200)'] = X_full.columns[kb_mi.get_support()].tolist()

for fs_name, cols in feature_sets.items():
    print(f"  {fs_name}: {len(cols)} features")

exp2_results = {}
for model_name in top3_names:
    exp2_results[model_name] = {}
    for fs_name, cols in feature_sets.items():
        if len(cols) == 0:
            print(f"  {model_name} × {fs_name}: SKIP (no features)")
            continue
        X_sub = X_full[cols]
        model = make_model(model_name)
        mae_m, mae_s, r2_m, r2_s = cv_scores(model, X_sub, y_unet)
        exp2_results[model_name][fs_name] = {
            'n_features': len(cols),
            'MAE_mean': float(mae_m), 'MAE_std': float(mae_s),
            'R2_mean': float(r2_m),  'R2_std': float(r2_s)
        }
        print(f"  {model_name:12s} × {fs_name:15s}  n={len(cols):4d}  "
              f"MAE={mae_m:.4f}±{mae_s:.4f}  R²={r2_m:.3f}±{r2_s:.3f}")

results['exp2_feature_selection'] = exp2_results

# ═══════════════════════════════════════════════════════════════════════════════
# Exp 3: HPO with Hyperopt (U-Net target)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Exp 3: HPO with Hyperopt (U-Net)")
print("="*60)

# Use best feature set from Exp 2 per model (by MAE)
def best_feature_set(model_name):
    if model_name not in exp2_results:
        return list(X_full.columns)
    best_fs = min(exp2_results[model_name].items(),
                  key=lambda x: x[1]['MAE_mean'])
    return feature_sets[best_fs[0]]

exp3_results = {}

# --- XGBoost HPO ---
print("\n  XGBoost HPO (50 evals)...")
xgb_cols = best_feature_set('XGBoost')
X_xgb = X_full[xgb_cols]
print(f"  Using {len(xgb_cols)} features")

space_xgb = {
    'n_estimators':     hp.quniform('n_estimators', 100, 600, 50),
    'learning_rate':    hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
    'max_depth':        hp.quniform('max_depth', 2, 8, 1),
    'subsample':        hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'gamma':            hp.loguniform('gamma', np.log(1e-5), np.log(1.0)),
    'reg_alpha':        hp.loguniform('reg_alpha', np.log(1e-5), np.log(10.0)),
    'reg_lambda':       hp.loguniform('reg_lambda', np.log(1e-5), np.log(10.0)),
}

def objective_xgb(params):
    params['n_estimators']     = int(params['n_estimators'])
    params['max_depth']        = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    model = XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1)
    mae_m, mae_s, r2_m, r2_s = cv_scores(model, X_xgb, y_unet)
    return {'loss': mae_m, 'status': STATUS_OK,
            'mae_std': mae_s, 'r2_mean': r2_m, 'r2_std': r2_s}

trials_xgb = Trials()
best_xgb = fmin(objective_xgb, space_xgb, algo=tpe.suggest,
                max_evals=50, trials=trials_xgb, rstate=np.random.default_rng(42),
                verbose=False)

best_xgb_trial = min(trials_xgb.results, key=lambda x: x['loss'])
print(f"  XGBoost best: MAE={best_xgb_trial['loss']:.4f}±{best_xgb_trial['mae_std']:.4f}"
      f"  R²={best_xgb_trial['r2_mean']:.3f}±{best_xgb_trial['r2_std']:.3f}")
exp3_results['XGBoost'] = {
    'best_params': {k: float(v) for k, v in best_xgb.items()},
    'n_features': len(xgb_cols),
    'MAE_mean': float(best_xgb_trial['loss']),
    'MAE_std':  float(best_xgb_trial['mae_std']),
    'R2_mean':  float(best_xgb_trial['r2_mean']),
    'R2_std':   float(best_xgb_trial['r2_std']),
}

# --- ExtraTrees HPO ---
print("\n  ExtraTrees HPO (50 evals)...")
et_cols = best_feature_set('ExtraTrees')
X_et = X_full[et_cols]
print(f"  Using {len(et_cols)} features")

space_et = {
    'n_estimators':    hp.quniform('n_estimators', 100, 600, 50),
    'max_depth':       hp.choice('max_depth', [None, 5, 8, 12, 16, 20]),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf':  hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features':    hp.choice('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
}

def objective_et(params):
    params['n_estimators']     = int(params['n_estimators'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf']  = int(params['min_samples_leaf'])
    model = ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
    mae_m, mae_s, r2_m, r2_s = cv_scores(model, X_et, y_unet)
    return {'loss': mae_m, 'status': STATUS_OK,
            'mae_std': mae_s, 'r2_mean': r2_m, 'r2_std': r2_s}

trials_et = Trials()
best_et = fmin(objective_et, space_et, algo=tpe.suggest,
               max_evals=50, trials=trials_et, rstate=np.random.default_rng(42),
               verbose=False)

best_et_trial = min(trials_et.results, key=lambda x: x['loss'])
print(f"  ExtraTrees best: MAE={best_et_trial['loss']:.4f}±{best_et_trial['mae_std']:.4f}"
      f"  R²={best_et_trial['r2_mean']:.3f}±{best_et_trial['r2_std']:.3f}")
exp3_results['ExtraTrees'] = {
    'best_params': {k: (float(v) if not isinstance(v, str) else v)
                    for k, v in best_et.items()},
    'n_features': len(et_cols),
    'MAE_mean': float(best_et_trial['loss']),
    'MAE_std':  float(best_et_trial['mae_std']),
    'R2_mean':  float(best_et_trial['r2_mean']),
    'R2_std':   float(best_et_trial['r2_std']),
}

# --- GBR HPO ---
print("\n  GBR HPO (50 evals)...")
gbr_cols = best_feature_set('GBR')
X_gbr = X_full[gbr_cols]
print(f"  Using {len(gbr_cols)} features")

space_gbr = {
    'n_estimators':   hp.quniform('n_estimators', 100, 600, 50),
    'learning_rate':  hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
    'max_depth':      hp.quniform('max_depth', 2, 8, 1),
    'subsample':      hp.uniform('subsample', 0.5, 1.0),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf':  hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features':   hp.choice('max_features', ['sqrt', 'log2', 0.3, 0.5]),
}

def objective_gbr(params):
    params['n_estimators']     = int(params['n_estimators'])
    params['max_depth']        = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf']  = int(params['min_samples_leaf'])
    model = GradientBoostingRegressor(**params, random_state=42)
    mae_m, mae_s, r2_m, r2_s = cv_scores(model, X_gbr, y_unet)
    return {'loss': mae_m, 'status': STATUS_OK,
            'mae_std': mae_s, 'r2_mean': r2_m, 'r2_std': r2_s}

trials_gbr = Trials()
best_gbr = fmin(objective_gbr, space_gbr, algo=tpe.suggest,
                max_evals=50, trials=trials_gbr, rstate=np.random.default_rng(42),
                verbose=False)

best_gbr_trial = min(trials_gbr.results, key=lambda x: x['loss'])
print(f"  GBR best: MAE={best_gbr_trial['loss']:.4f}±{best_gbr_trial['mae_std']:.4f}"
      f"  R²={best_gbr_trial['r2_mean']:.3f}±{best_gbr_trial['r2_std']:.3f}")
exp3_results['GBR'] = {
    'best_params': {k: (float(v) if not isinstance(v, str) else v)
                    for k, v in best_gbr.items()},
    'n_features': len(gbr_cols),
    'MAE_mean': float(best_gbr_trial['loss']),
    'MAE_std':  float(best_gbr_trial['mae_std']),
    'R2_mean':  float(best_gbr_trial['r2_mean']),
    'R2_std':   float(best_gbr_trial['r2_std']),
}

results['exp3_hpo'] = exp3_results

# ═══════════════════════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════════════════════
with open(RESULT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULT_JSON}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Oracle Model Improvement (U-Net WT Dice, 5-fold CV)", fontsize=13)

# ── Panel A: Exp 1 Model comparison ─────────────────────────────────────────
ax = axes[0]
names  = list(exp1_results.keys())
maes   = [exp1_results[n]['MAE_mean'] for n in names]
errs   = [exp1_results[n]['MAE_std']  for n in names]
colors = ['steelblue'] * len(names)
bars = ax.barh(names, maes, xerr=errs, color=colors, capsize=4)
ax.set_xlabel("MAE")
ax.set_title("(A) Model Comparison\n(all features, no HPO)")
ax.axvline(0.039, color='red', linestyle='--', linewidth=1, label='GBR baseline (paper)')
ax.legend(fontsize=8)
ax.invert_yaxis()

# ── Panel B: Exp 2 Feature selection (top 2 models) ─────────────────────────
ax = axes[1]
fs_names = list(feature_sets.keys())
x = np.arange(len(fs_names))
width = 0.25

# use top 2 models by Exp1 MAE
plot_models = top3_names[:2] if len(top3_names) >= 2 else top3_names
for i, mn in enumerate(plot_models):
    maes_fs = [exp2_results.get(mn, {}).get(fs, {}).get('MAE_mean', np.nan) for fs in fs_names]
    errs_fs = [exp2_results.get(mn, {}).get(fs, {}).get('MAE_std',  np.nan) for fs in fs_names]
    ax.bar(x + i*width, maes_fs, width, yerr=errs_fs, capsize=3, label=mn)

ax.set_xticks(x + width/2)
ax.set_xticklabels(fs_names, rotation=30, ha='right', fontsize=8)
ax.set_ylabel("MAE")
ax.set_title("(B) Feature Selection\n(top 2 models)")
ax.axhline(0.039, color='red', linestyle='--', linewidth=1, label='baseline')
ax.legend(fontsize=8)

# ── Panel C: Exp 3 HPO comparison ────────────────────────────────────────────
ax = axes[2]
hpo_models = list(exp3_results.keys())
maes_hpo  = [exp3_results[m]['MAE_mean'] for m in hpo_models]
errs_hpo  = [exp3_results[m]['MAE_std']  for m in hpo_models]
# also baseline (Exp1 no-HPO)
baseline_no_hpo = [exp1_results.get(m, {}).get('MAE_mean', np.nan) for m in hpo_models]

x = np.arange(len(hpo_models))
width = 0.35
ax.bar(x - width/2, baseline_no_hpo, width, label='No HPO', color='steelblue')
ax.bar(x + width/2, maes_hpo, width, yerr=errs_hpo, capsize=4, label='With HPO', color='darkorange')
ax.set_xticks(x)
ax.set_xticklabels(hpo_models)
ax.set_ylabel("MAE")
ax.set_title("(C) Effect of HPO\n(best feature set)")
ax.axhline(0.039, color='red', linestyle='--', linewidth=1, label='GBR baseline')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIGURE_PATH, bbox_inches='tight', dpi=150)
print(f"Figure saved to {FIGURE_PATH}")
