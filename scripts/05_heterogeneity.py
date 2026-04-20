"""
=============================================================================
PART 5 — HETEROGENEOUS TREATMENT EFFECTS
Latent Personality Constructs and Professional Success
=============================================================================

This script estimates the ATT of high Fearless Dominance (FD) on
Professional Satisfaction across subgroups:
  - By gender (male vs. female)
  - By experience level (above/below median months in job)
  - By age group (above/below median age)

Method: Propensity Score Matching (1:1 NN) within each subgroup.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

DATA_PATH  = '/home/ubuntu/psychopathy_project/output/df_processed.csv'
OUTPUT_DIR = '/home/ubuntu/psychopathy_project/output'
FIGURES_DIR = '/home/ubuntu/psychopathy_project/figures'

print("=" * 70)
print("PART 5 — HETEROGENEOUS TREATMENT EFFECTS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

# ─── Setup ────────────────────────────────────────────────────────────────────
outcome    = 'ProfSat_composite'
confounders = ['Age', 'Gender_male', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op', 'MonthsInJob']

fd_p75 = df['FD'].quantile(0.75)
df['T_FD'] = (df['FD'] > fd_p75).astype(int)

analysis_cols = ['T_FD', 'FD', outcome, 'SCI', 'CO'] + confounders
df_ci = df[analysis_cols].dropna().copy()

# ─── PSM function ─────────────────────────────────────────────────────────────
def run_psm(data, confounders, treatment_col, outcome_col, n_boot=1000, seed=42):
    """Run 1:1 NN PSM and return ATT with bootstrap CI."""
    data = data.copy().reset_index(drop=True)
    
    X = data[confounders].values
    T = data[treatment_col].values
    Y = data[outcome_col].values
    
    if T.sum() < 5 or (1 - T).sum() < 5:
        return {'att': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan, 'n_treated': T.sum(), 'n_control': (1-T).sum()}
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(max_iter=1000, random_state=seed)
    ps_model.fit(X_s, T)
    data['ps'] = ps_model.predict_proba(X_s)[:, 1]
    
    # Common support
    ps_t = data[data[treatment_col] == 1]['ps']
    ps_c = data[data[treatment_col] == 0]['ps']
    overlap_min = max(ps_t.min(), ps_c.min())
    overlap_max = min(ps_t.max(), ps_c.max())
    data = data[(data['ps'] >= overlap_min) & (data['ps'] <= overlap_max)].copy().reset_index(drop=True)
    
    treated_df = data[data[treatment_col] == 1].reset_index(drop=True)
    control_df = data[data[treatment_col] == 0].reset_index(drop=True)
    
    if len(treated_df) < 5 or len(control_df) < 5:
        return {'att': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan, 'n_treated': len(treated_df), 'n_control': len(control_df)}
    
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control_df[['ps']].values)
    _, indices = nn.kneighbors(treated_df[['ps']].values)
    
    matched_control = control_df.iloc[indices.flatten()].reset_index(drop=True)
    matched_treated = treated_df.reset_index(drop=True)
    
    att = matched_treated[outcome_col].mean() - matched_control[outcome_col].mean()
    _, p_val = ttest_ind(matched_treated[outcome_col], matched_control[outcome_col], equal_var=False)
    
    # Bootstrap CI
    np.random.seed(seed)
    boot_att = []
    for _ in range(n_boot):
        idx = np.random.choice(len(matched_treated), len(matched_treated), replace=True)
        boot_att.append(
            matched_treated[outcome_col].iloc[idx].mean() - matched_control[outcome_col].iloc[idx].mean()
        )
    
    return {
        'att': att,
        'ci_lower': np.percentile(boot_att, 2.5),
        'ci_upper': np.percentile(boot_att, 97.5),
        'p_value': p_val,
        'n_treated': len(matched_treated),
        'n_control': len(matched_control),
    }

# ─── 1. FULL SAMPLE (reference) ───────────────────────────────────────────────
print("\n" + "─" * 50)
print("1. FULL SAMPLE (reference)")
print("─" * 50)

full_result = run_psm(df_ci, confounders, 'T_FD', outcome)
print(f"  ATT = {full_result['att']:.3f} (95% CI [{full_result['ci_lower']:.3f}, {full_result['ci_upper']:.3f}], p = {full_result['p_value']:.4f})")
print(f"  n_treated = {full_result['n_treated']}, n_control = {full_result['n_control']}")

# ─── 2. HETEROGENEITY BY GENDER ───────────────────────────────────────────────
print("\n" + "─" * 50)
print("2. HETEROGENEITY BY GENDER")
print("─" * 50)

conf_no_gender = [c for c in confounders if c != 'Gender_male']

gender_results = {}
for gender_val, gender_label in [(1, 'Male'), (0, 'Female')]:
    sub = df_ci[df_ci['Gender_male'] == gender_val].copy()
    result = run_psm(sub, conf_no_gender, 'T_FD', outcome)
    gender_results[gender_label] = result
    print(f"  {gender_label}: n={len(sub)}, ATT = {result['att']:.3f} "
          f"(95% CI [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}], p = {result['p_value']:.4f})")

# ─── 3. HETEROGENEITY BY EXPERIENCE ───────────────────────────────────────────
print("\n" + "─" * 50)
print("3. HETEROGENEITY BY EXPERIENCE (Months in Job)")
print("─" * 50)

median_exp = df_ci['MonthsInJob'].median()
conf_no_exp = [c for c in confounders if c != 'MonthsInJob']

exp_results = {}
for exp_label, mask in [
    (f'High experience (>{median_exp:.0f} months)', df_ci['MonthsInJob'] > median_exp),
    (f'Low experience (≤{median_exp:.0f} months)', df_ci['MonthsInJob'] <= median_exp),
]:
    sub = df_ci[mask].copy()
    result = run_psm(sub, conf_no_exp, 'T_FD', outcome)
    exp_results[exp_label] = result
    print(f"  {exp_label}: n={len(sub)}, ATT = {result['att']:.3f} "
          f"(95% CI [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}], p = {result['p_value']:.4f})")

# ─── 4. HETEROGENEITY BY AGE ──────────────────────────────────────────────────
print("\n" + "─" * 50)
print("4. HETEROGENEITY BY AGE")
print("─" * 50)

median_age = df_ci['Age'].median()
conf_no_age = [c for c in confounders if c != 'Age']

age_results = {}
for age_label, mask in [
    (f'Older (>{median_age:.0f} years)', df_ci['Age'] > median_age),
    (f'Younger (≤{median_age:.0f} years)', df_ci['Age'] <= median_age),
]:
    sub = df_ci[mask].copy()
    result = run_psm(sub, conf_no_age, 'T_FD', outcome)
    age_results[age_label] = result
    print(f"  {age_label}: n={len(sub)}, ATT = {result['att']:.3f} "
          f"(95% CI [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}], p = {result['p_value']:.4f})")

# ─── 5. COMPILE RESULTS ───────────────────────────────────────────────────────
all_results = []

all_results.append({
    'Subgroup': 'Full Sample', 'Category': 'Reference',
    **full_result
})
for label, res in gender_results.items():
    all_results.append({'Subgroup': label, 'Category': 'Gender', **res})
for label, res in exp_results.items():
    all_results.append({'Subgroup': label, 'Category': 'Experience', **res})
for label, res in age_results.items():
    all_results.append({'Subgroup': label, 'Category': 'Age', **res})

hte_df = pd.DataFrame(all_results)
hte_df.round(4).to_csv(f'{OUTPUT_DIR}/heterogeneous_treatment_effects.csv', index=False)

# ─── 6. VISUALISATION ─────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("5. GENERATING VISUALISATION")
print("─" * 50)

plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(11, 7))

categories = ['Reference', 'Gender', 'Experience', 'Age']
cat_colors = {'Reference': '#555555', 'Gender': '#2C7BB6', 'Experience': '#1A9641', 'Age': '#D7191C'}
cat_markers = {'Reference': 'D', 'Gender': 'o', 'Experience': 's', 'Age': '^'}

y_labels = []
y_positions = []
y = 0
category_breaks = []

for cat in categories:
    sub = hte_df[hte_df['Category'] == cat]
    if y_labels:
        y += 0.5  # spacing between categories
        category_breaks.append(y)
    for _, row in sub.iterrows():
        if not np.isnan(row['att']):
            sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else ''))
            label = f"{row['Subgroup']}  (n={int(row['n_treated'])+int(row['n_control'])}) {sig}"
            ax.errorbar(
                row['att'], y,
                xerr=[[row['att'] - row['ci_lower']], [row['ci_upper'] - row['att']]],
                fmt=cat_markers[cat], color=cat_colors[cat],
                markersize=10, capsize=5, linewidth=2.2, elinewidth=1.8
            )
            y_labels.append(label)
            y_positions.append(y)
            y += 1

ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)

# Add reference line for full sample ATT
ax.axvline(full_result['att'], color='#555555', linewidth=1, linestyle=':', alpha=0.5,
           label=f"Full sample ATT = {full_result['att']:.3f}")

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=9.5)
ax.set_xlabel('ATT Estimate (95% Bootstrap CI)\nOutcome: Professional Satisfaction', fontsize=11)
ax.set_title('Heterogeneous Treatment Effects of High Fearless Dominance\n'
             'Treatment: FD > 75th percentile vs. FD ≤ 75th percentile\n'
             '(Propensity Score Matching, 1:1 NN, within-subgroup)',
             fontsize=12, fontweight='bold')

# Category labels on the right
for cat in categories:
    sub = hte_df[hte_df['Category'] == cat]
    valid = sub[~sub['att'].isna()]
    if len(valid) > 0:
        y_mid = y_positions[hte_df[hte_df['Category'] == cat].index[0] - hte_df.index[0]]

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=cat_colors['Reference'], label='Full Sample'),
    Patch(facecolor=cat_colors['Gender'], label='By Gender'),
    Patch(facecolor=cat_colors['Experience'], label='By Experience'),
    Patch(facecolor=cat_colors['Age'], label='By Age'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
ax.grid(axis='x', alpha=0.3)

# Significance note
ax.text(0.01, 0.01, '* p<0.05  ** p<0.01  *** p<0.001',
        transform=ax.transAxes, fontsize=8, color='gray', va='bottom')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig14_heterogeneous_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: fig14_heterogeneous_effects.png")

print("\n" + "=" * 70)
print("HETEROGENEOUS TREATMENT EFFECTS COMPLETE")
print("=" * 70)
print(hte_df[['Subgroup', 'att', 'ci_lower', 'ci_upper', 'p_value', 'n_treated']].round(3).to_string(index=False))
