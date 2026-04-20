"""
=============================================================================
PART 1 — REPLICATION
Latent Personality Constructs and Professional Success
Replication of Eisenbarth, Hart & Sedikides (2018)
Journal of Economic Psychology, 64, 130–139
=============================================================================

This script replicates the core analyses of the paper:
  1. Data loading and preprocessing
  2. Reliability analysis (Cronbach's Alpha)
  3. Descriptive statistics and gender comparisons
  4. Zero-order correlations (Table 3)
  5. Structural Equation Modelling via path analysis (Table 4)
  6. Visualisations

Data source: https://osf.io/tgujv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Optional: pingouin for reliability
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

# Optional: statsmodels for OLS
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = '/home/ubuntu/psychopathy_project/data/PsychopathySuccess_dfFinal.txt'
OUTPUT_DIR  = '/home/ubuntu/psychopathy_project/output'
FIGURES_DIR = '/home/ubuntu/psychopathy_project/figures'

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 70)
print("PART 1 — REPLICATION: Eisenbarth, Hart & Sedikides (2018)")
print("=" * 70)

df_raw = pd.read_csv(DATA_PATH, sep=' ', quotechar='"', low_memory=False)
print(f"\nRaw dataset: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ─── 2. SELECT AND RENAME KEY VARIABLES ───────────────────────────────────────
# Variables used in the paper (already computed in the dataset)
key_vars = {
    # Psychopathy PPI-R dimensions
    'PPIR40FD':      'FD',          # Fearless Dominance
    'PPIR40SC':      'SCI',         # Self-Centered Impulsivity
    'PPI_R_40_Co':   'CO',          # Coldheartedness
    'PPI_R_40_SUM':  'PPI_SUM',     # Total psychopathy score
    # Big Five (TIPI)
    'bf_Ex': 'BF_Ex',
    'bf_Ag': 'BF_Ag',
    'bf_Co': 'BF_Co',
    'bf_Em': 'BF_Em',
    'bf_Op': 'BF_Op',
    # Professional Satisfaction indicators
    'CareerSa': 'CareerSat',
    'PromSat':  'PromSat',
    'SalSat':   'SalSat',
    # Material Success indicators
    'AnnSalary': 'AnnSalary',
    'PromFreq':  'PromFreq',
    'OwnOffice': 'OwnOffice',
    'CarAccess': 'CarAccess',
    'Budget':    'Budget',
    'Employee':  'Employee',
    # Composites already in dataset
    'SubjectiveSuccess':  'ProfSat_composite',
    'ObjectiveSuccess':   'MatSucc_composite',
    'ProfStd':            'ProfStd',
    # Controls
    'Whatisyourgender': 'Gender',
    'MonthsInJob':      'MonthsInJob',
    'Whatisyourage':    'Age',
    # Impression Management
    'IM': 'IM',
}

df = df_raw[list(key_vars.keys())].rename(columns=key_vars).copy()

# Recode gender: 1 = male, 0 = female (as in paper)
df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')
# In the dataset: 1=female, 2=male → recode to 0/1 (male=1 as in paper)
df['Gender_male'] = (df['Gender'] == 2).astype(int)

# Ensure numeric
numeric_cols = [c for c in df.columns if c not in ['Gender']]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Apply exclusion criteria from paper:
# (1) failed to complete all measures (n=20), (2) prior participation (n=16),
# (3) age < 18 or non-English first language (n=2)
# The dataset provided is already the final cleaned sample (N=439)
print(f"Final sample N = {len(df)}")

# ─── 3. RELIABILITY ANALYSIS (CRONBACH'S ALPHA) ───────────────────────────────
print("\n" + "─" * 50)
print("3. RELIABILITY ANALYSIS (Cronbach's Alpha)")
print("─" * 50)

def cronbach_alpha(df_items):
    """Compute Cronbach's alpha for a set of items."""
    df_items = df_items.dropna()
    n = df_items.shape[1]
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    alpha = (n / (n - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

# Professional Satisfaction (3 items)
alpha_profsat = cronbach_alpha(df[['CareerSat', 'PromSat', 'SalSat']])
print(f"  Professional Satisfaction (CareerSat, PromSat, SalSat): α = {alpha_profsat:.3f}")
print(f"  Paper reports: α = 0.78 ✓" if abs(alpha_profsat - 0.78) < 0.05 else f"  Paper reports: α = 0.78")

# Material Success (3 items from ProfStd + AnnSalary + PromFreq)
alpha_matsuc = cronbach_alpha(df[['ProfStd', 'AnnSalary', 'PromFreq']])
print(f"  Material Success (ProfStd, AnnSalary, PromFreq):        α = {alpha_matsuc:.3f}")
print(f"  Paper reports: α = 0.48 ✓" if abs(alpha_matsuc - 0.48) < 0.05 else f"  Paper reports: α = 0.48")

# PPI-R subscales
alpha_fd  = cronbach_alpha(df_raw[['ppi_r_40_07r','ppi_r_40_17r','ppi_r_40_24r',
                                    'ppi_r_40_26r','ppi_r_40_39r']].apply(pd.to_numeric, errors='coerce'))
print(f"  PPI-R Coldheartedness (5 items):                        α = {alpha_fd:.3f}")
print(f"  Paper reports: α = 0.73 ✓" if abs(alpha_fd - 0.73) < 0.05 else f"  Paper reports: α = 0.73")

# ─── 4. DESCRIPTIVE STATISTICS ────────────────────────────────────────────────
print("\n" + "─" * 50)
print("4. DESCRIPTIVE STATISTICS")
print("─" * 50)

desc_vars = ['FD', 'SCI', 'CO', 'PPI_SUM', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op',
             'CareerSat', 'PromSat', 'SalSat', 'ProfSat_composite',
             'AnnSalary', 'PromFreq', 'ProfStd', 'MatSucc_composite']

desc_df = df[desc_vars].describe().T[['mean', 'std', 'min', 'max']]
desc_df.columns = ['Mean', 'SD', 'Min', 'Max']
print(desc_df.round(2).to_string())

# Gender split
n_women = (df['Gender'] == 1).sum()
n_men   = (df['Gender'] == 2).sum()
print(f"\n  Women: n = {n_women} ({n_women/len(df)*100:.1f}%)")
print(f"  Men:   n = {n_men} ({n_men/len(df)*100:.1f}%)")
print(f"  Mean age: {df['Age'].mean():.2f} (SD = {df['Age'].std():.2f})")

# ─── 5. ZERO-ORDER CORRELATIONS (TABLE 3) ─────────────────────────────────────
print("\n" + "─" * 50)
print("5. ZERO-ORDER CORRELATIONS (replication of Table 3)")
print("─" * 50)

corr_predictors = ['PPI_SUM', 'FD', 'SCI', 'CO', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op']
corr_outcomes   = ['ProfSat_composite', 'MatSucc_composite']

results_corr = []
for pred in corr_predictors:
    row = {'Predictor': pred}
    for out in corr_outcomes:
        valid = df[[pred, out]].dropna()
        r, p = pearsonr(valid[pred], valid[out])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        row[out] = f"{r:.2f}{sig}"
    results_corr.append(row)

corr_table = pd.DataFrame(results_corr).set_index('Predictor')
corr_table.columns = ['Prof. Satisfaction', 'Material Success']
print(corr_table.to_string())

print("\n  Paper Table 3 reference values:")
print("  FD  → Prof.Sat: 0.12**,  Mat.Succ: 0.15***")
print("  SCI → Prof.Sat: -0.13**, Mat.Succ: -0.01")
print("  CO  → Prof.Sat: -0.05,   Mat.Succ: 0.00")

# ─── 6. OLS REGRESSION (approximation of SEM Table 4) ────────────────────────
print("\n" + "─" * 50)
print("6. OLS REGRESSION (approximation of SEM — Table 4)")
print("─" * 50)

# Standardise the satisfaction composite (z-score, as in paper)
df['ProfSat_z'] = (df['ProfSat_composite'] - df['ProfSat_composite'].mean()) / df['ProfSat_composite'].std()
df['MatSucc_z'] = (df['MatSucc_composite'] - df['MatSucc_composite'].mean()) / df['MatSucc_composite'].std()

# Model 1 (small): FD + SCI + CO → Professional Satisfaction
m1_sat = smf.ols('ProfSat_z ~ FD + SCI + CO', data=df).fit()
print("\n  Model 1a — Professional Satisfaction (FD + SCI + CO):")
print(f"    FD  B = {m1_sat.params['FD']:.4f}, p = {m1_sat.pvalues['FD']:.3f}")
print(f"    SCI B = {m1_sat.params['SCI']:.4f}, p = {m1_sat.pvalues['SCI']:.3f}")
print(f"    CO  B = {m1_sat.params['CO']:.4f}, p = {m1_sat.pvalues['CO']:.3f}")
print(f"    R² = {m1_sat.rsquared:.4f}")

# Model 1 (small): FD + SCI + CO → Material Success
m1_mat = smf.ols('MatSucc_z ~ FD + SCI + CO', data=df).fit()
print("\n  Model 1b — Material Success (FD + SCI + CO):")
print(f"    FD  B = {m1_mat.params['FD']:.4f}, p = {m1_mat.pvalues['FD']:.3f}")
print(f"    SCI B = {m1_mat.params['SCI']:.4f}, p = {m1_mat.pvalues['SCI']:.3f}")
print(f"    CO  B = {m1_mat.params['CO']:.4f}, p = {m1_mat.pvalues['CO']:.3f}")
print(f"    R² = {m1_mat.rsquared:.4f}")

# Model 2 (large): + Big Five + Gender + MonthsInJob → Professional Satisfaction
m2_sat = smf.ols('ProfSat_z ~ FD + SCI + CO + BF_Ex + BF_Ag + BF_Co + BF_Em + BF_Op + Gender_male + MonthsInJob',
                  data=df).fit()
print("\n  Model 2a — Professional Satisfaction (+ Big Five + controls):")
for var in ['FD', 'SCI', 'CO', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op', 'Gender_male', 'MonthsInJob']:
    sig = '***' if m2_sat.pvalues[var] < 0.001 else ('**' if m2_sat.pvalues[var] < 0.01 else ('*' if m2_sat.pvalues[var] < 0.05 else ''))
    print(f"    {var:12s} B = {m2_sat.params[var]:+.4f}, p = {m2_sat.pvalues[var]:.3f} {sig}")
print(f"    R² = {m2_sat.rsquared:.4f}")

# Model 2 (large): + Big Five + Gender + MonthsInJob → Material Success
m2_mat = smf.ols('MatSucc_z ~ FD + SCI + CO + BF_Ex + BF_Ag + BF_Co + BF_Em + BF_Op + Gender_male + MonthsInJob',
                  data=df).fit()
print("\n  Model 2b — Material Success (+ Big Five + controls):")
for var in ['FD', 'SCI', 'CO', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op', 'Gender_male', 'MonthsInJob']:
    sig = '***' if m2_mat.pvalues[var] < 0.001 else ('**' if m2_mat.pvalues[var] < 0.01 else ('*' if m2_mat.pvalues[var] < 0.05 else ''))
    print(f"    {var:12s} B = {m2_mat.params[var]:+.4f}, p = {m2_mat.pvalues[var]:.3f} {sig}")
print(f"    R² = {m2_mat.rsquared:.4f}")

# ─── 7. VISUALISATIONS ────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("7. GENERATING VISUALISATIONS")
print("─" * 50)

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ['#2C7BB6', '#D7191C', '#1A9641']

# --- Figure 1: Correlation heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr_vars = ['FD', 'SCI', 'CO', 'PPI_SUM', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op',
             'ProfSat_composite', 'MatSucc_composite']
labels = ['FD', 'SCI', 'CO', 'PPI Total', 'Extraversion', 'Agreeableness',
          'Conscientiousness', 'Emot. Stability', 'Openness',
          'Prof. Satisfaction', 'Material Success']
corr_matrix = df[corr_vars].corr()
corr_matrix.index = labels
corr_matrix.columns = labels

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-0.6, vmax=0.6, ax=ax,
            annot_kws={'size': 9}, linewidths=0.5)
ax.set_title('Zero-Order Correlations\nEisenbarth et al. (2018) — Replication', 
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig1_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: fig1_correlation_heatmap.png")

# --- Figure 2: Regression coefficients comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models_data = [
    ('Professional Satisfaction', m1_sat, m2_sat),
    ('Material Success', m1_mat, m2_mat),
]

ppi_vars = ['FD', 'SCI', 'CO']
colors_m1 = '#2C7BB6'
colors_m2 = '#D7191C'

for ax, (title, m_small, m_large) in zip(axes, models_data):
    y_pos = np.arange(len(ppi_vars))
    
    b_small = [m_small.params[v] for v in ppi_vars]
    ci_small = [(m_small.conf_int().loc[v, 0], m_small.conf_int().loc[v, 1]) for v in ppi_vars]
    
    b_large = [m_large.params[v] for v in ppi_vars]
    ci_large = [(m_large.conf_int().loc[v, 0], m_large.conf_int().loc[v, 1]) for v in ppi_vars]
    
    # Model 1 (without controls)
    for i, (b, ci) in enumerate(zip(b_small, ci_small)):
        ax.errorbar(b, y_pos[i] + 0.15, xerr=[[b - ci[0]], [ci[1] - b]],
                    fmt='o', color=colors_m1, markersize=8, capsize=4, linewidth=2,
                    label='Without controls' if i == 0 else '')
    
    # Model 2 (with controls)
    for i, (b, ci) in enumerate(zip(b_large, ci_large)):
        ax.errorbar(b, y_pos[i] - 0.15, xerr=[[b - ci[0]], [ci[1] - b]],
                    fmt='s', color=colors_m2, markersize=8, capsize=4, linewidth=2,
                    label='With Big Five + controls' if i == 0 else '')
    
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(['Fearless Dominance (FD)', 'Self-Centered Impulsivity (SCI)', 'Coldheartedness (CO)'],
                       fontsize=11)
    ax.set_xlabel('Unstandardized Regression Coefficient (B)', fontsize=10)
    ax.set_title(f'Predictors of\n{title}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Replication of Table 4 — Eisenbarth et al. (2018)', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig2_regression_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: fig2_regression_coefficients.png")

# --- Figure 3: Distribution of PPI-R dimensions by gender ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
ppi_dims = [('FD', 'Fearless Dominance'), ('SCI', 'Self-Centered Impulsivity'), ('CO', 'Coldheartedness')]

for ax, (var, label) in zip(axes, ppi_dims):
    women = df[df['Gender'] == 1][var].dropna()
    men   = df[df['Gender'] == 2][var].dropna()
    
    ax.hist(women, bins=20, alpha=0.6, color='#E8A0BF', label=f'Women (n={len(women)})', density=True)
    ax.hist(men,   bins=20, alpha=0.6, color='#4472C4', label=f'Men (n={len(men)})', density=True)
    
    t_stat, p_val = ttest_ind(women, men, equal_var=False)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
    
    ax.set_title(f'{label}\nt({len(women)+len(men)-2}) = {t_stat:.2f}, p {sig}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9)

plt.suptitle('Distribution of Psychopathic Trait Dimensions by Gender', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig3_ppi_distributions_gender.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: fig3_ppi_distributions_gender.png")

# ─── 8. SAVE REPLICATION SUMMARY ──────────────────────────────────────────────
print("\n" + "─" * 50)
print("8. SAVING REPLICATION SUMMARY")
print("─" * 50)

# Save correlation table
corr_full = df[corr_vars].corr()
corr_full.index = labels
corr_full.columns = labels
corr_full.round(3).to_csv(f'{OUTPUT_DIR}/replication_correlation_matrix.csv')

# Save regression results
reg_results = {
    'Model': ['1a: Prof.Sat (no controls)', '1b: Mat.Succ (no controls)',
              '2a: Prof.Sat (+ Big Five)', '2b: Mat.Succ (+ Big Five)'],
    'FD_B': [m1_sat.params['FD'], m1_mat.params['FD'], m2_sat.params['FD'], m2_mat.params['FD']],
    'FD_p': [m1_sat.pvalues['FD'], m1_mat.pvalues['FD'], m2_sat.pvalues['FD'], m2_mat.pvalues['FD']],
    'SCI_B': [m1_sat.params['SCI'], m1_mat.params['SCI'], m2_sat.params['SCI'], m2_mat.params['SCI']],
    'SCI_p': [m1_sat.pvalues['SCI'], m1_mat.pvalues['SCI'], m2_sat.pvalues['SCI'], m2_mat.pvalues['SCI']],
    'CO_B': [m1_sat.params['CO'], m1_mat.params['CO'], m2_sat.params['CO'], m2_mat.params['CO']],
    'CO_p': [m1_sat.pvalues['CO'], m1_mat.pvalues['CO'], m2_sat.pvalues['CO'], m2_mat.pvalues['CO']],
    'R2': [m1_sat.rsquared, m1_mat.rsquared, m2_sat.rsquared, m2_mat.rsquared],
}
pd.DataFrame(reg_results).round(4).to_csv(f'{OUTPUT_DIR}/replication_regression_results.csv', index=False)

# Save processed dataframe for next scripts
df.to_csv(f'{OUTPUT_DIR}/df_processed.csv', index=False)

print(f"\n  Saved: replication_correlation_matrix.csv")
print(f"  Saved: replication_regression_results.csv")
print(f"  Saved: df_processed.csv")

print("\n" + "=" * 70)
print("REPLICATION COMPLETE")
print("=" * 70)
print(f"\nKey findings:")
print(f"  FD  → Prof.Sat: r = {df[['FD','ProfSat_composite']].dropna().pipe(lambda x: pearsonr(x['FD'], x['ProfSat_composite'])[0]):.3f}")
print(f"  SCI → Prof.Sat: r = {df[['SCI','ProfSat_composite']].dropna().pipe(lambda x: pearsonr(x['SCI'], x['ProfSat_composite'])[0]):.3f}")
print(f"  FD  → Mat.Succ: r = {df[['FD','MatSucc_composite']].dropna().pipe(lambda x: pearsonr(x['FD'], x['MatSucc_composite'])[0]):.3f}")
print(f"  SCI → Mat.Succ: r = {df[['SCI','MatSucc_composite']].dropna().pipe(lambda x: pearsonr(x['SCI'], x['MatSucc_composite'])[0]):.3f}")
