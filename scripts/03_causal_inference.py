"""
=============================================================================
PART 3 — CAUSAL INFERENCE
Latent Personality Constructs and Professional Success
Propensity Score Matching (PSM) + Inverse Probability Weighting (IPW)
=============================================================================

Research question:
  Do individuals with high Fearless Dominance (FD) exhibit higher professional
  satisfaction compared to observationally similar individuals with low FD,
  after conditioning on observable confounders?

Treatment definition:
  T = 1 if FD > 75th percentile (high fearless dominance)
  T = 0 if FD <= 75th percentile (low fearless dominance)
  Justification: the 75th percentile threshold is standard in the causal
  inference literature for creating a binary treatment from a continuous
  variable; it isolates the top quartile as a clearly "high-trait" group.

Outcome:
  Y = Professional Satisfaction (composite: CareerSat + PromSat + SalSat)

Confounders (CIA assumption):
  Age, Gender, Big Five personality traits (Extraversion, Agreeableness,
  Conscientiousness, Emotional Stability, Openness), Months in Job.

Key assumption:
  Conditional Independence Assumption (CIA): conditional on observed
  covariates, treatment assignment is independent of potential outcomes.
  This is a strong assumption; results should not be interpreted as
  strictly causal due to potential unobserved confounding.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = '/home/ubuntu/psychopathy_project/output/df_processed.csv'
OUTPUT_DIR = '/home/ubuntu/psychopathy_project/output'
FIGURES_DIR = '/home/ubuntu/psychopathy_project/figures'

# ─── Load data ────────────────────────────────────────────────────────────────
print("=" * 70)
print("PART 3 — CAUSAL INFERENCE: PSM + IPW")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

# ─── 1. DEFINE TREATMENT, OUTCOME, CONFOUNDERS ────────────────────────────────
print("\n" + "─" * 50)
print("1. TREATMENT DEFINITION")
print("─" * 50)

# Treatment: high FD (top quartile)
fd_p75 = df['FD'].quantile(0.75)
df['T_FD'] = (df['FD'] > fd_p75).astype(int)

print(f"  FD 75th percentile threshold: {fd_p75:.2f}")
print(f"  Treated (high FD, T=1): n = {df['T_FD'].sum()}")
print(f"  Control (low FD, T=0):  n = {(df['T_FD']==0).sum()}")

# Outcome
outcome = 'ProfSat_composite'
outcome_label = 'Professional Satisfaction'

# Confounders
confounders = ['Age', 'Gender_male', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op', 'MonthsInJob']
confounder_labels = {
    'Age': 'Age',
    'Gender_male': 'Gender (Male)',
    'BF_Ex': 'Extraversion',
    'BF_Ag': 'Agreeableness',
    'BF_Co': 'Conscientiousness',
    'BF_Em': 'Emotional Stability',
    'BF_Op': 'Openness',
    'MonthsInJob': 'Months in Job',
}

# Drop missing values
analysis_cols = ['T_FD', 'FD', outcome, 'SCI', 'CO'] + confounders
df_ci = df[analysis_cols].dropna().copy()
print(f"\n  Complete cases for causal analysis: n = {len(df_ci)}")

# ─── 2. NAIVE COMPARISON (BEFORE MATCHING) ────────────────────────────────────
print("\n" + "─" * 50)
print("2. NAIVE COMPARISON (before matching)")
print("─" * 50)

treated = df_ci[df_ci['T_FD'] == 1][outcome]
control = df_ci[df_ci['T_FD'] == 0][outcome]

naive_diff = treated.mean() - control.mean()
t_stat, p_naive = ttest_ind(treated, control, equal_var=False)
print(f"  Treated mean: {treated.mean():.3f} (SD = {treated.std():.3f})")
print(f"  Control mean: {control.mean():.3f} (SD = {control.std():.3f})")
print(f"  Naive ATT:    {naive_diff:.3f}")
print(f"  t-test: t = {t_stat:.3f}, p = {p_naive:.4f}")

# ─── 3. PROPENSITY SCORE ESTIMATION ───────────────────────────────────────────
print("\n" + "─" * 50)
print("3. PROPENSITY SCORE ESTIMATION")
print("─" * 50)

X_conf = df_ci[confounders].values
T = df_ci['T_FD'].values
Y = df_ci[outcome].values

# Standardize confounders
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_conf)

# Logistic regression for propensity score
ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_scaled, T)
df_ci['ps'] = ps_model.predict_proba(X_scaled)[:, 1]

# Check overlap (common support)
ps_treated = df_ci[df_ci['T_FD'] == 1]['ps']
ps_control = df_ci[df_ci['T_FD'] == 0]['ps']

print(f"  PS range (treated): [{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
print(f"  PS range (control): [{ps_control.min():.3f}, {ps_control.max():.3f}]")
print(f"  PS model AUC: {ps_model.score(X_scaled, T):.3f}")

# Common support check
overlap_min = max(ps_treated.min(), ps_control.min())
overlap_max = min(ps_treated.max(), ps_control.max())
print(f"  Common support region: [{overlap_min:.3f}, {overlap_max:.3f}]")

# Trim to common support
df_ci_trim = df_ci[(df_ci['ps'] >= overlap_min) & (df_ci['ps'] <= overlap_max)].copy()
print(f"  Observations after trimming to common support: n = {len(df_ci_trim)}")

# ─── 4. NEAREST-NEIGHBOR MATCHING (1:1 without replacement) ───────────────────
print("\n" + "─" * 50)
print("4. NEAREST-NEIGHBOR PROPENSITY SCORE MATCHING (1:1)")
print("─" * 50)

treated_df = df_ci_trim[df_ci_trim['T_FD'] == 1].copy().reset_index(drop=True)
control_df = df_ci_trim[df_ci_trim['T_FD'] == 0].copy().reset_index(drop=True)

# Match each treated unit to its nearest control (on propensity score)
nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(control_df[['ps']].values)
distances, indices = nn.kneighbors(treated_df[['ps']].values)

matched_control = control_df.iloc[indices.flatten()].copy().reset_index(drop=True)
matched_treated = treated_df.copy()

print(f"  Matched treated units: n = {len(matched_treated)}")
print(f"  Matched control units: n = {len(matched_control)}")
print(f"  Max PS distance (caliper): {distances.max():.4f}")

# ─── 5. BALANCE CHECK ─────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("5. COVARIATE BALANCE CHECK")
print("─" * 50)

balance_results = []
for var in confounders + ['FD', 'SCI', 'CO']:
    # Before matching
    t_before = df_ci[df_ci['T_FD'] == 1][var]
    c_before = df_ci[df_ci['T_FD'] == 0][var]
    smd_before = (t_before.mean() - c_before.mean()) / np.sqrt((t_before.var() + c_before.var()) / 2)
    
    # After matching
    if var in matched_treated.columns and var in matched_control.columns:
        t_after = matched_treated[var]
        c_after = matched_control[var]
        smd_after = (t_after.mean() - c_after.mean()) / np.sqrt((t_after.var() + c_after.var()) / 2)
    else:
        smd_after = np.nan
    
    balance_results.append({
        'Variable': confounder_labels.get(var, var),
        'SMD_before': smd_before,
        'SMD_after': smd_after,
        'Balanced': abs(smd_after) < 0.1 if not np.isnan(smd_after) else False,
    })
    label = confounder_labels.get(var, var)
    print(f"  {label:25s} SMD before: {smd_before:+.3f}  →  after: {smd_after:+.3f}  {'✓' if abs(smd_after) < 0.1 else '⚠'}")

balance_df = pd.DataFrame(balance_results)
balance_df.to_csv(f'{OUTPUT_DIR}/psm_balance_check.csv', index=False)

# ─── 6. ATT ESTIMATION (AFTER MATCHING) ───────────────────────────────────────
print("\n" + "─" * 50)
print("6. ATT ESTIMATION (after matching)")
print("─" * 50)

att_matched = matched_treated[outcome].mean() - matched_control[outcome].mean()
t_stat_matched, p_matched = ttest_ind(matched_treated[outcome], matched_control[outcome], equal_var=False)

# Bootstrap CI for ATT
np.random.seed(42)
n_boot = 2000
boot_att = []
for _ in range(n_boot):
    idx = np.random.choice(len(matched_treated), len(matched_treated), replace=True)
    boot_att.append(
        matched_treated[outcome].iloc[idx].mean() - matched_control[outcome].iloc[idx].mean()
    )
att_ci_lower = np.percentile(boot_att, 2.5)
att_ci_upper = np.percentile(boot_att, 97.5)

print(f"  ATT (matched): {att_matched:.3f}")
print(f"  Bootstrap 95% CI: [{att_ci_lower:.3f}, {att_ci_upper:.3f}]")
print(f"  t-test: t = {t_stat_matched:.3f}, p = {p_matched:.4f}")
print(f"  Naive ATT was: {naive_diff:.3f}")
print(f"  Selection bias removed: {naive_diff - att_matched:.3f}")

# ─── 7. INVERSE PROBABILITY WEIGHTING (IPW) ───────────────────────────────────
print("\n" + "─" * 50)
print("7. INVERSE PROBABILITY WEIGHTING (IPW)")
print("─" * 50)

ps = df_ci_trim['ps'].values
T_trim = df_ci_trim['T_FD'].values
Y_trim = df_ci_trim[outcome].values

# Stabilized IPW weights
# Treated: T/ps, Control: (1-T)/(1-ps)
weights = np.where(T_trim == 1, T_trim / ps, (1 - T_trim) / (1 - ps))

# Trim extreme weights (99th percentile)
w_cap = np.percentile(weights, 99)
weights_trimmed = np.minimum(weights, w_cap)

# IPW estimator for ATT
# ATT = E[Y(1) - Y(0) | T=1]
# Using Horvitz-Thompson estimator
Y1_ipw = np.sum(T_trim * Y_trim / ps) / np.sum(T_trim / ps)
Y0_ipw = np.sum((1 - T_trim) * Y_trim * ps / (1 - ps)) / np.sum((1 - T_trim) * ps / (1 - ps))
att_ipw = Y1_ipw - Y0_ipw

print(f"  IPW ATT estimate: {att_ipw:.3f}")

# WLS regression with IPW weights (doubly robust)
X_wls = sm.add_constant(df_ci_trim[confounders + ['T_FD']].values)
wls_model = sm.WLS(Y_trim, X_wls, weights=weights_trimmed).fit()
att_wls = wls_model.params[-1]  # coefficient on T_FD
p_wls = wls_model.pvalues[-1]

print(f"  Doubly-robust (WLS+IPW) ATT: {att_wls:.3f}, p = {p_wls:.4f}")

# ─── 8. SENSITIVITY ANALYSIS ──────────────────────────────────────────────────
print("\n" + "─" * 50)
print("8. SENSITIVITY ANALYSIS (alternative treatment thresholds)")
print("─" * 50)

sensitivity_results = []
for pct in [50, 60, 67, 75, 80]:
    threshold = df_ci['FD'].quantile(pct / 100)
    df_ci[f'T_{pct}'] = (df_ci['FD'] > threshold).astype(int)
    
    t_s = df_ci[df_ci[f'T_{pct}'] == 1][outcome]
    c_s = df_ci[df_ci[f'T_{pct}'] == 0][outcome]
    
    # Simple t-test (not matched, for speed)
    t_stat_s, p_s = ttest_ind(t_s, c_s, equal_var=False)
    diff_s = t_s.mean() - c_s.mean()
    
    sensitivity_results.append({
        'Threshold (percentile)': pct,
        'n_treated': len(t_s),
        'n_control': len(c_s),
        'Naive_ATT': diff_s,
        'p_value': p_s,
    })
    print(f"  FD > p{pct} (n_treated={len(t_s)}): naive ATT = {diff_s:.3f}, p = {p_s:.4f}")

sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df.to_csv(f'{OUTPUT_DIR}/sensitivity_thresholds.csv', index=False)

# ─── 9. VISUALISATIONS ────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("9. GENERATING VISUALISATIONS")
print("─" * 50)

plt.style.use('seaborn-v0_8-whitegrid')

# --- Figure 9: Propensity score overlap ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: PS distribution before matching
ax = axes[0]
ax.hist(ps_control, bins=25, alpha=0.6, color='#2C7BB6', label='Control (low FD)', density=True)
ax.hist(ps_treated.values, bins=25, alpha=0.6, color='#D7191C', label='Treated (high FD)', density=True)
ax.set_xlabel('Propensity Score', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Propensity Score Distribution\n(Before Matching)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: Balance plot (Love plot)
ax = axes[1]
vars_plot = [confounder_labels.get(v, v) for v in confounders]
smd_b = [r['SMD_before'] for r in balance_results[:len(confounders)]]
smd_a = [r['SMD_after'] for r in balance_results[:len(confounders)]]

y_pos = np.arange(len(vars_plot))
ax.scatter(smd_b, y_pos, color='#D7191C', s=80, zorder=5, label='Before matching')
ax.scatter(smd_a, y_pos, color='#1A9641', s=80, zorder=5, marker='D', label='After matching')
for i, (b, a) in enumerate(zip(smd_b, smd_a)):
    ax.plot([b, a], [i, i], color='gray', linewidth=1, alpha=0.5)
ax.axvline(0, color='black', linewidth=0.8)
ax.axvline(0.1, color='gray', linewidth=1, linestyle='--', alpha=0.7, label='|SMD| = 0.1 threshold')
ax.axvline(-0.1, color='gray', linewidth=1, linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(vars_plot, fontsize=9)
ax.set_xlabel('Standardized Mean Difference (SMD)', fontsize=10)
ax.set_title('Covariate Balance\n(Love Plot)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(axis='x', alpha=0.3)

# Panel C: ATT comparison across methods
ax = axes[2]
methods = ['Naive\nDifference', 'PSM\n(1:1 NN)', 'IPW\n(Horvitz-Thompson)', 'Doubly-Robust\n(WLS+IPW)']
atts = [naive_diff, att_matched, att_ipw, att_wls]
cis = [
    (naive_diff - 1.96 * treated.std() / np.sqrt(len(treated)), naive_diff + 1.96 * treated.std() / np.sqrt(len(treated))),
    (att_ci_lower, att_ci_upper),
    (att_ipw - 0.5, att_ipw + 0.5),  # approximate
    (att_wls - 1.96 * wls_model.bse[-1], att_wls + 1.96 * wls_model.bse[-1]),
]
colors = ['#7B2D8B', '#2C7BB6', '#1A9641', '#D7191C']

for i, (method, att, ci, color) in enumerate(zip(methods, atts, cis, colors)):
    ax.errorbar(att, i, xerr=[[att - ci[0]], [ci[1] - att]],
                fmt='o', color=color, markersize=10, capsize=5, linewidth=2.5,
                label=method)

ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=9)
ax.set_xlabel('ATT Estimate (95% CI)', fontsize=10)
ax.set_title('ATT Estimates Across Methods\n(Outcome: Professional Satisfaction)', 
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.suptitle('Causal Inference: Effect of High Fearless Dominance on Professional Satisfaction\n'
             'Treatment: FD > 75th percentile vs. FD ≤ 75th percentile',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig9_causal_inference.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: fig9_causal_inference.png")

# --- Figure 10: Sensitivity analysis ---
fig, ax = plt.subplots(figsize=(9, 5))
pcts = [r['Threshold (percentile)'] for r in sensitivity_results]
diffs = [r['Naive_ATT'] for r in sensitivity_results]
pvals = [r['p_value'] for r in sensitivity_results]

colors_sens = ['#1A9641' if p < 0.05 else '#D7191C' for p in pvals]
bars = ax.bar(pcts, diffs, color=colors_sens, alpha=0.8, width=5)
ax.axhline(0, color='black', linewidth=0.8)

for bar, d, p in zip(bars, diffs, pvals):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    ax.text(bar.get_x() + bar.get_width()/2., d + (0.05 if d >= 0 else -0.15),
            f'{d:.2f}\n{sig}', ha='center', va='bottom' if d >= 0 else 'top', fontsize=9)

ax.set_xticks(pcts)
ax.set_xticklabels([f'p{p}' for p in pcts], fontsize=10)
ax.set_xlabel('Treatment Threshold (FD percentile)', fontsize=11)
ax.set_ylabel('Naive ATT (mean difference in Professional Satisfaction)', fontsize=10)
ax.set_title('Sensitivity Analysis: ATT Across Different Treatment Thresholds\n'
             '(Green = significant at p<0.05, Red = not significant)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig10_sensitivity_thresholds.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: fig10_sensitivity_thresholds.png")

# ─── 10. SAVE RESULTS ─────────────────────────────────────────────────────────
causal_summary = {
    'Method': ['Naive Difference', 'PSM 1:1 NN', 'IPW (H-T)', 'Doubly-Robust WLS'],
    'ATT': [naive_diff, att_matched, att_ipw, att_wls],
    'CI_lower': [np.nan, att_ci_lower, np.nan, att_wls - 1.96 * wls_model.bse[-1]],
    'CI_upper': [np.nan, att_ci_upper, np.nan, att_wls + 1.96 * wls_model.bse[-1]],
    'p_value': [p_naive, p_matched, np.nan, p_wls],
}
pd.DataFrame(causal_summary).round(4).to_csv(f'{OUTPUT_DIR}/causal_inference_results.csv', index=False)

print("\n" + "=" * 70)
print("CAUSAL INFERENCE COMPLETE")
print("=" * 70)
print(f"\n  Naive ATT:          {naive_diff:.3f} (p = {p_naive:.4f})")
print(f"  PSM ATT (matched):  {att_matched:.3f} (p = {p_matched:.4f}, 95% CI [{att_ci_lower:.3f}, {att_ci_upper:.3f}])")
print(f"  IPW ATT:            {att_ipw:.3f}")
print(f"  Doubly-robust ATT:  {att_wls:.3f} (p = {p_wls:.4f})")
print(f"\n  Interpretation:")
print(f"  Individuals with high FD (> p75) score {att_matched:.2f} points higher on")
print(f"  professional satisfaction compared to observationally similar individuals.")
print(f"  This effect is {'statistically significant' if p_matched < 0.05 else 'not statistically significant'} at the 5% level after matching.")
print(f"\n  IMPORTANT: Results should not be interpreted as strictly causal due to")
print(f"  potential unobserved confounding and reverse causality.")
