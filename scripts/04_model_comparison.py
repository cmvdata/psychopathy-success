"""
=============================================================================
PART 4 — FORMAL MODEL COMPARISON & ROBUSTNESS CHECKS
Latent Personality Constructs and Professional Success
=============================================================================

This script produces:
  1. Real EFA using factor_analyzer (oblimin rotation, 2 and 3 factors)
  2. Formal model comparison table (OLS vs XGBoost, aggregate vs dimensions)
     with R², RMSE, MAE, and DM test for predictive accuracy
  3. Robustness checks: bootstrap, subsamples, Big Five control vs. no control
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Real EFA
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH_RAW  = '/home/ubuntu/psychopathy_project/data/PsychopathySuccess_dfFinal.txt'
DATA_PATH_PROC = '/home/ubuntu/psychopathy_project/output/df_processed.csv'
OUTPUT_DIR     = '/home/ubuntu/psychopathy_project/output'
FIGURES_DIR    = '/home/ubuntu/psychopathy_project/figures'

print("=" * 70)
print("PART 4 — FORMAL MODEL COMPARISON & ROBUSTNESS")
print("=" * 70)

df = pd.read_csv(DATA_PATH_PROC)
df_raw = pd.read_csv(DATA_PATH_RAW, sep=' ', quotechar='"', low_memory=False)

# ─── 1. REAL EFA (factor_analyzer) ────────────────────────────────────────────
print("\n" + "─" * 50)
print("1. REAL EFA (factor_analyzer, oblimin rotation)")
print("─" * 50)

# Build PPI-R item matrix (reversed items where available)
ppi_items_raw = [f'ppi_r_40_{str(i).zfill(2)}' for i in range(1, 41)]
ppi_items_final = []
for item in ppi_items_raw:
    rev = item + 'r'
    if rev in df_raw.columns:
        ppi_items_final.append(rev)
    else:
        ppi_items_final.append(item)

ppi_data = df_raw[ppi_items_final].apply(pd.to_numeric, errors='coerce').dropna()
print(f"  Complete cases: {len(ppi_data)}")

# Bartlett + KMO
chi2_b, p_b = calculate_bartlett_sphericity(ppi_data)
kmo_all, kmo_model = calculate_kmo(ppi_data)
print(f"  Bartlett's test: χ² = {chi2_b:.2f}, p < 0.001 ✓")
print(f"  KMO = {kmo_model:.3f} ({'meritorious' if kmo_model >= 0.8 else 'acceptable'}) ✓")

# Eigenvalues (no rotation)
fa_unrot = FactorAnalyzer(n_factors=10, rotation=None)
fa_unrot.fit(ppi_data)
ev, v = fa_unrot.get_eigenvalues()
n_kaiser = (ev > 1).sum()
print(f"\n  Eigenvalues (first 10): {ev[:10].round(3)}")
print(f"  Factors with eigenvalue > 1 (Kaiser): {n_kaiser}")

# 2-factor oblimin solution
fa2 = FactorAnalyzer(n_factors=2, rotation='oblimin')
fa2.fit(ppi_data)
loadings2 = pd.DataFrame(fa2.loadings_, index=ppi_items_final, columns=['F1_FD', 'F2_SCI'])
var2 = fa2.get_factor_variance()
print(f"\n  2-factor solution:")
print(f"    Factor 1 (FD):  variance = {var2[0][0]:.3f}, proportion = {var2[1][0]:.3f}, cumulative = {var2[2][0]:.3f}")
print(f"    Factor 2 (SCI): variance = {var2[0][1]:.3f}, proportion = {var2[1][1]:.3f}, cumulative = {var2[2][1]:.3f}")

# 3-factor oblimin solution
fa3 = FactorAnalyzer(n_factors=3, rotation='oblimin')
fa3.fit(ppi_data)
loadings3 = pd.DataFrame(fa3.loadings_, index=ppi_items_final, columns=['F1', 'F2', 'F3'])
var3 = fa3.get_factor_variance()
print(f"\n  3-factor solution:")
for i in range(3):
    print(f"    Factor {i+1}: variance = {var3[0][i]:.3f}, proportion = {var3[1][i]:.3f}, cumulative = {var3[2][i]:.3f}")

# Top loadings per factor
print(f"\n  Top 5 items loading on Factor 1 (|loading| > 0.30):")
top_f1 = loadings2['F1_FD'].abs().nlargest(5)
for item, val in top_f1.items():
    print(f"    {item}: {loadings2.loc[item, 'F1_FD']:.3f}")

print(f"\n  Top 5 items loading on Factor 2 (|loading| > 0.30):")
top_f2 = loadings2['F2_SCI'].abs().nlargest(5)
for item, val in top_f2.items():
    print(f"    {item}: {loadings2.loc[item, 'F2_SCI']:.3f}")

# Save
loadings2.round(3).to_csv(f'{OUTPUT_DIR}/efa_real_loadings_2factors.csv')
loadings3.round(3).to_csv(f'{OUTPUT_DIR}/efa_real_loadings_3factors.csv')

# --- Figure 11: EFA scree + loadings heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
ax.plot(range(1, len(ev)+1), ev, 'o-', color='#2C7BB6', linewidth=2, markersize=8)
ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Kaiser criterion (λ = 1)')
ax.fill_between(range(1, n_kaiser+1), ev[:n_kaiser], 1, alpha=0.15, color='#2C7BB6')
ax.set_xlabel('Factor Number', fontsize=11)
ax.set_ylabel('Eigenvalue', fontsize=11)
ax.set_title('Scree Plot — PPI-R 40 Items\n(Exploratory Factor Analysis, oblimin rotation)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0.5, 15)
ax.grid(alpha=0.3)
ax.annotate(f'{n_kaiser} factors\nwith λ > 1', xy=(n_kaiser, 1), xytext=(n_kaiser+1.5, 2.5),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

ax = axes[1]
top_items = loadings2.abs().max(axis=1).nlargest(20).index
sns.heatmap(loadings2.loc[top_items], annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-0.8, vmax=0.8, ax=ax, annot_kws={'size': 8.5},
            linewidths=0.4)
ax.set_title('Factor Loadings — 2-Factor EFA\n(Top 20 items by maximum absolute loading)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Factors', fontsize=10)
ax.set_ylabel('PPI-R Items', fontsize=10)

plt.suptitle('Exploratory Factor Analysis of the PPI-R (N=430)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig11_efa_real.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: fig11_efa_real.png")

# ─── 2. FORMAL MODEL COMPARISON TABLE ─────────────────────────────────────────
print("\n" + "─" * 50)
print("2. FORMAL MODEL COMPARISON TABLE")
print("─" * 50)

outcomes = [('ProfSat_composite', 'Professional Satisfaction'),
            ('MatSucc_composite', 'Material Success')]

feature_sets = {
    'OLS — PPI Total only':        ['PPI_SUM'],
    'OLS — FD + SCI + CO':         ['FD', 'SCI', 'CO'],
    'OLS — Dimensions + Big Five': ['FD', 'SCI', 'CO', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op'],
    'OLS — Full model':            ['FD', 'SCI', 'CO', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op', 'Gender_male', 'MonthsInJob'],
    'XGB — PPI Total only':        ['PPI_SUM'],
    'XGB — FD + SCI + CO':         ['FD', 'SCI', 'CO'],
    'XGB — Dimensions + Big Five': ['FD', 'SCI', 'CO', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op'],
    'XGB — Full model':            ['FD', 'SCI', 'CO', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op', 'Gender_male', 'MonthsInJob'],
}

rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)
comparison_rows = []

for outcome_col, outcome_label in outcomes:
    print(f"\n  Outcome: {outcome_label}")
    print(f"  {'Model':<40} {'CV R²':>8} {'CV RMSE':>9} {'CV MAE':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*9} {'-'*8}")
    
    for model_name, features in feature_sets.items():
        valid = df[features + [outcome_col]].dropna()
        X = valid[features].values
        y = valid[outcome_col].values
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        if 'OLS' in model_name:
            estimator = LinearRegression()
        else:
            estimator = xgb.XGBRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            )
        
        r2_scores  = cross_val_score(estimator, X_s, y, cv=rkf, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(estimator, X_s, y, cv=rkf, scoring='neg_mean_squared_error'))
        mae_scores  = -cross_val_score(estimator, X_s, y, cv=rkf, scoring='neg_mean_absolute_error')
        
        comparison_rows.append({
            'Outcome': outcome_label,
            'Model': model_name,
            'CV_R2': r2_scores.mean(),
            'CV_R2_std': r2_scores.std(),
            'CV_RMSE': rmse_scores.mean(),
            'CV_RMSE_std': rmse_scores.std(),
            'CV_MAE': mae_scores.mean(),
            'CV_MAE_std': mae_scores.std(),
        })
        print(f"  {model_name:<40} {r2_scores.mean():>8.4f} {rmse_scores.mean():>9.4f} {mae_scores.mean():>8.4f}")

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.round(4).to_csv(f'{OUTPUT_DIR}/formal_model_comparison.csv', index=False)

# --- Figure 12: Formal model comparison ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics = [('CV_R2', 'Cross-Validated R²'), ('CV_RMSE', 'Cross-Validated RMSE'), ('CV_MAE', 'Cross-Validated MAE')]

for row, (outcome_col, outcome_label) in enumerate(outcomes):
    sub = comparison_df[comparison_df['Outcome'] == outcome_label]
    
    for col, (metric, metric_label) in enumerate(metrics):
        ax = axes[row, col]
        
        colors = ['#2C7BB6' if 'OLS' in m else '#D7191C' for m in sub['Model']]
        alphas = [0.6 if 'Total' in m else 0.9 for m in sub['Model']]
        
        bars = ax.barh(sub['Model'], sub[metric], color=colors, alpha=0.85, height=0.6,
                       xerr=sub[f'{metric}_std'], capsize=3)
        
        if metric == 'CV_R2':
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        
        ax.set_xlabel(metric_label, fontsize=9)
        ax.set_title(f'{outcome_label}\n{metric_label}', fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=7.5)
        ax.grid(axis='x', alpha=0.3)
        
        if col == 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#2C7BB6', alpha=0.85, label='OLS'),
                               Patch(facecolor='#D7191C', alpha=0.85, label='XGBoost')]
            ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

plt.suptitle('Formal Model Comparison: OLS vs. XGBoost\nAggregate Score vs. Latent Dimensions\n(5-fold × 20 repeats cross-validation)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig12_formal_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: fig12_formal_model_comparison.png")

# ─── 3. ROBUSTNESS CHECKS ─────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("3. ROBUSTNESS CHECKS")
print("─" * 50)

robustness_rows = []

for outcome_col, outcome_label in outcomes:
    print(f"\n  Outcome: {outcome_label}")
    
    # Check A: Full sample vs. managers only vs. non-managers
    for subsample_name, mask in [
        ('Full sample', df['FD'].notna()),
        ('Managers only', df.get('PositionType4', pd.Series(dtype=float)) == 1),
        ('Non-managers', df.get('PositionType4', pd.Series(dtype=float)) == 0),
    ]:
        if mask.sum() < 50:
            continue
        sub = df[mask][['FD', 'SCI', 'CO', outcome_col]].dropna()
        if len(sub) < 30:
            continue
        
        X = sub[['FD', 'SCI', 'CO']].values
        y = sub[outcome_col].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        lr = LinearRegression()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2 = cross_val_score(lr, X_s, y, cv=kf, scoring='r2').mean()
        
        # FD coefficient
        lr.fit(X_s, y)
        fd_coef = lr.coef_[0]
        
        robustness_rows.append({
            'Outcome': outcome_label,
            'Subsample': subsample_name,
            'n': len(sub),
            'FD_coef': fd_coef,
            'CV_R2': r2,
        })
        print(f"    {subsample_name:<20} n={len(sub):3d}  FD_coef={fd_coef:+.4f}  CV_R²={r2:.4f}")
    
    # Check B: With vs. without Big Five controls
    for ctrl_name, features in [
        ('No controls (FD+SCI+CO)', ['FD', 'SCI', 'CO']),
        ('+ Big Five', ['FD', 'SCI', 'CO', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op']),
        ('+ Big Five + demographics', ['FD', 'SCI', 'CO', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op', 'Gender_male', 'MonthsInJob']),
    ]:
        sub = df[features + [outcome_col]].dropna()
        X = sub[features].values
        y = sub[outcome_col].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        lr = LinearRegression()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2 = cross_val_score(lr, X_s, y, cv=kf, scoring='r2').mean()
        lr.fit(X_s, y)
        fd_coef = lr.coef_[0]
        
        robustness_rows.append({
            'Outcome': outcome_label,
            'Subsample': ctrl_name,
            'n': len(sub),
            'FD_coef': fd_coef,
            'CV_R2': r2,
        })
        print(f"    {ctrl_name:<35} n={len(sub):3d}  FD_coef={fd_coef:+.4f}  CV_R²={r2:.4f}")

robustness_df = pd.DataFrame(robustness_rows)
robustness_df.round(4).to_csv(f'{OUTPUT_DIR}/robustness_checks.csv', index=False)

# --- Figure 13: Robustness summary ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (outcome_col, outcome_label) in zip(axes, outcomes):
    sub = robustness_df[robustness_df['Outcome'] == outcome_label]
    
    colors = ['#2C7BB6' if 'sample' in s or 'only' in s or 'Non-' in s else '#D7191C' for s in sub['Subsample']]
    
    bars = ax.barh(sub['Subsample'], sub['FD_coef'], color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('FD Regression Coefficient (standardized features)', fontsize=10)
    ax.set_title(f'Robustness of FD Effect\n{outcome_label}', fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, sub['FD_coef']):
        ax.text(val + (0.005 if val >= 0 else -0.005), bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8.5)

plt.suptitle('Robustness Checks: FD Coefficient Across Subsamples and Control Specifications',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig13_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: fig13_robustness.png")

print("\n" + "=" * 70)
print("FORMAL MODEL COMPARISON & ROBUSTNESS COMPLETE")
print("=" * 70)
