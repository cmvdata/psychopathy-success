"""
=============================================================================
PART 6 — FD CONSTRUCT REDUNDANCY ANALYSIS
=============================================================================

How much (linearly and informationally) is FD redundant with Big Five?

BLOCK 1 — Linear redundancy (Pearson r, OLS, VIF)
BLOCK 2 — Information-theoretic redundancy (KSG mutual information, entropy)

Outputs:
  output/fd_redundancy_analysis.csv       Block 1 metrics
  output/fd_redundancy_models.txt         Full OLS regression summaries
  output/fd_redundancy_information.csv    Block 2 metrics (MI in bits, entropy)
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from scipy.special import digamma
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / 'output' / 'df_processed.csv'
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
SEED      = 42
LN2       = np.log(2)
N_BINS_FD = 10   # quantile bins for FD entropy
K_KSG     = 3    # k-NN parameter for KSG estimator (sklearn default)

BF_VARS = ['BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op']
BF_LABELS = {
    'BF_Ex': 'Extraversion',
    'BF_Ag': 'Agreeableness',
    'BF_Co': 'Conscientiousness',
    'BF_Em': 'Emotional Stability',
    'BF_Op': 'Openness',
}
CONTROL_VARS = ['Age', 'Gender_male', 'MonthsInJob']
PREDICTORS   = BF_VARS + CONTROL_VARS

# ═════════════════════════════════════════════════════════════════════════════
# Load
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 6 — FD CONSTRUCT REDUNDANCY ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df_clean = df[['FD'] + PREDICTORS].dropna().reset_index(drop=True)
print(f"\nSample: n = {len(df_clean)}")
print(f"FD: mean = {df_clean['FD'].mean():.2f}, sd = {df_clean['FD'].std():.2f}")

results_b1, results_b2, log_lines = [], [], []

# ═════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — LINEAR REDUNDANCY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("BLOCK 1 — LINEAR REDUNDANCY (Pearson, OLS, VIF)")
print("═" * 70)

# 1.1 Pearson
print("\n─── 1.1 Pearson correlations FD vs Big Five ───")
log_lines.append("=" * 70)
log_lines.append("BLOCK 1.1 — Pearson correlations FD vs Big Five")
log_lines.append("=" * 70)
for v in BF_VARS:
    r, p = pearsonr(df_clean['FD'], df_clean[v])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    if   abs(r) > 0.5:  interp = 'strong'
    elif abs(r) > 0.3:  interp = 'moderate'
    elif abs(r) > 0.1:  interp = 'weak'
    else:               interp = 'negligible'
    print(f"  FD ↔ {BF_LABELS[v]:25s} r = {r:+.3f}  p = {p:.3e}  [{interp} {sig}]")
    log_lines.append(f"  FD ↔ {BF_LABELS[v]:25s} r = {r:+.3f}  p = {p:.3e}  ({sig})")
    results_b1.append({
        'metric': 'Pearson_r', 'variable': BF_LABELS[v],
        'value': round(r, 4),
        'interpretation': f'{interp} ({sig}, p={p:.2e})',
    })

# 1.2 OLS full
print("\n─── 1.2 OLS full: FD ~ Age + Gender + Big Five + MonthsInJob ───")
log_lines.append("\n" + "=" * 70)
log_lines.append("BLOCK 1.2 — OLS full")
log_lines.append("=" * 70)

cont_pred = [v for v in PREDICTORS if v != 'Gender_male']
df_z = df_clean.copy()
df_z[cont_pred] = StandardScaler().fit_transform(df_clean[cont_pred])
df_z['FD_z']    = (df_clean['FD'] - df_clean['FD'].mean()) / df_clean['FD'].std()

X_full = sm.add_constant(df_z[PREDICTORS])
m_full = sm.OLS(df_z['FD_z'], X_full).fit()
print(m_full.summary())
log_lines.append(str(m_full.summary()))
print(f"\n  R² = {m_full.rsquared:.4f}")
print(f"  R²_adj = {m_full.rsquared_adj:.4f}")
print(f"  F = {m_full.fvalue:.2f}, p = {m_full.f_pvalue:.3e}")
print(f"  N = {int(m_full.nobs)}")

results_b1.append({'metric': 'OLS_full_R2', 'variable': 'all_predictors',
                   'value': round(m_full.rsquared, 4),
                   'interpretation': f'{m_full.rsquared*100:.1f}% of FD variance explained linearly'})
results_b1.append({'metric': 'OLS_full_R2_adj', 'variable': 'all_predictors',
                   'value': round(m_full.rsquared_adj, 4),
                   'interpretation': 'adjusted R²'})
results_b1.append({'metric': 'OLS_full_F', 'variable': 'all_predictors',
                   'value': round(m_full.fvalue, 4),
                   'interpretation': f'F-stat (p = {m_full.f_pvalue:.2e})'})

print("\n  Standardized β (Big Five only):")
for v in BF_VARS:
    b = m_full.params[v]; p = m_full.pvalues[v]
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    print(f"    {BF_LABELS[v]:25s} β = {b:+.3f}  p = {p:.3e}  ({sig})")
    results_b1.append({'metric': 'OLS_full_std_beta', 'variable': BF_LABELS[v],
                       'value': round(b, 4),
                       'interpretation': f'p = {p:.2e} ({sig})'})

# 1.3 OLS reduced
print("\n─── 1.3 OLS reduced: FD ~ BF_Ex + BF_Em ───")
log_lines.append("\n" + "=" * 70)
log_lines.append("BLOCK 1.3 — OLS reduced: FD ~ BF_Ex + BF_Em")
log_lines.append("=" * 70)
m_red = sm.OLS(df_z['FD_z'], sm.add_constant(df_z[['BF_Ex', 'BF_Em']])).fit()
print(m_red.summary())
log_lines.append(str(m_red.summary()))
frac_full = m_red.rsquared / m_full.rsquared * 100 if m_full.rsquared > 0 else np.nan
print(f"\n  R² = {m_red.rsquared:.4f}, R²_adj = {m_red.rsquared_adj:.4f}")
print(f"  Reduced (BF_Ex+BF_Em) recovers {frac_full:.1f}% of full R²")
results_b1.append({'metric': 'OLS_reduced_R2', 'variable': 'BF_Ex+BF_Em',
                   'value': round(m_red.rsquared, 4),
                   'interpretation': f'{frac_full:.1f}% of full-model R² captured by 2 predictors'})

# 1.4 VIF
print("\n─── 1.4 VIF (multicollinearity) ───")
log_lines.append("\n" + "=" * 70)
log_lines.append("BLOCK 1.4 — VIF in full model")
log_lines.append("=" * 70)
# Add constant column so each VIF regression has an intercept (required for
# correct VIF; without it the implicit regressions inflate R^2 due to non-zero means).
X_vif = sm.add_constant(df_clean[PREDICTORS]).values
for i, v in enumerate(PREDICTORS):
    vif = variance_inflation_factor(X_vif, i + 1)  # +1 to skip the constant column
    label = BF_LABELS.get(v, v)
    if   vif > 10:    interp = 'severe (>10)'
    elif vif > 5:     interp = 'moderate (5-10)'
    elif vif > 2.5:   interp = 'mild (2.5-5)'
    else:             interp = 'minimal (<2.5)'
    print(f"  {label:25s} VIF = {vif:.2f}  [{interp}]")
    log_lines.append(f"  {label:25s} VIF = {vif:.2f}  ({interp})")
    if v in BF_VARS:
        results_b1.append({'metric': 'VIF', 'variable': BF_LABELS[v],
                           'value': round(vif, 4), 'interpretation': interp})

pd.DataFrame(results_b1).to_csv(OUTPUT_DIR / 'fd_redundancy_analysis.csv', index=False)
with open(OUTPUT_DIR / 'fd_redundancy_models.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
print(f"\n  Saved: fd_redundancy_analysis.csv")
print(f"  Saved: fd_redundancy_models.txt")

# ═════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — INFORMATION-THEORETIC REDUNDANCY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("BLOCK 2 — INFORMATION-THEORETIC REDUNDANCY (MI, entropy)")
print("═" * 70)

# Standardize for KSG (Chebyshev distance is scale-sensitive)
df_zall = df_clean.copy()
df_zall[['FD'] + BF_VARS] = StandardScaler().fit_transform(df_clean[['FD'] + BF_VARS])
FD_arr = df_zall['FD'].values.reshape(-1, 1)
BF_arr = df_zall[BF_VARS].values
n_obs  = len(df_zall)


def ksg_mi(X, Y, k=K_KSG):
    """
    Kraskov-Stögbauer-Grassberger (KSG-1) MI estimator. Phys. Rev. E (2004).
    X: (n, dx), Y: (n, dy). Returns MI in nats (>= 0). Uses Chebyshev (max) norm.
    """
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    if X.ndim == 1: X = X.reshape(-1, 1)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    n  = X.shape[0]
    XY = np.hstack([X, Y])

    tree_xy = cKDTree(XY)
    eps = tree_xy.query(XY, k=k+1, p=np.inf)[0][:, k]   # distance to k-th NN

    tree_x = cKDTree(X); tree_y = cKDTree(Y)
    nx = np.array([len(tree_x.query_ball_point(X[i], eps[i] - 1e-12, p=np.inf)) - 1
                   for i in range(n)])
    ny = np.array([len(tree_y.query_ball_point(Y[i], eps[i] - 1e-12, p=np.inf)) - 1
                   for i in range(n)])
    nx = np.maximum(nx, 1); ny = np.maximum(ny, 1)

    return max(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)), 0.0)


# 2.1 Univariate MI
print("\n─── 2.1 Univariate MI: I(FD ; BF_k) (sklearn KSG, k=3) ───")
mi_uni = mutual_info_regression(BF_arr, FD_arr.ravel(),
                                  random_state=SEED, n_neighbors=K_KSG) / LN2
mi_uni_dict = dict(zip(BF_VARS, mi_uni))
for v, mi in sorted(mi_uni_dict.items(), key=lambda x: -x[1]):
    print(f"  I(FD ; {BF_LABELS[v]:25s}) = {mi:.4f} bits")
    results_b2.append({'metric': 'MI_univariate', 'variable': BF_LABELS[v],
                       'value_bits': round(mi, 4),
                       'interpretation': 'sklearn mutual_info_regression (KSG, k=3)'})

# 2.2 Joint MI
print("\n─── 2.2 Joint MI: I(FD ; Big Five vector) (KSG-1) ───")
mi_joint    = ksg_mi(FD_arr, BF_arr) / LN2
mi_uni_sum  = float(mi_uni.sum())
print(f"  I(FD ; BF_full)  = {mi_joint:.4f} bits")
print(f"  Σ univariate MIs = {mi_uni_sum:.4f} bits")
print(f"  Joint / Σ_uni    = {mi_joint / mi_uni_sum:.3f}  (<<1 → BFs share information)")
results_b2.append({'metric': 'MI_joint_full', 'variable': 'BF_full(5_vars)',
                   'value_bits': round(mi_joint, 4),
                   'interpretation': 'ceiling: total info BF carries about FD'})
results_b2.append({'metric': 'MI_sum_univariate', 'variable': 'BF_full(5_vars)',
                   'value_bits': round(mi_uni_sum, 4),
                   'interpretation': 'sum of univariate MIs (>= joint due to BF overlap)'})

# 2.3 Incremental MI
print("\n─── 2.3 Incremental MI: I(FD ; BF_k | rest) ≈ I_full - I_without_k ───")
for k_idx, v in enumerate(BF_VARS):
    rest = BF_arr[:, [j for j in range(len(BF_VARS)) if j != k_idx]]
    mi_rest = ksg_mi(FD_arr, rest) / LN2
    inc     = max(0.0, mi_joint - mi_rest)
    print(f"  ΔI from {BF_LABELS[v]:25s} = {inc:.4f} bits  "
          f"(MI without it = {mi_rest:.4f})")
    results_b2.append({'metric': 'MI_incremental', 'variable': BF_LABELS[v],
                       'value_bits': round(inc, 4),
                       'interpretation': 'unique info given the other 4 BFs'})

# 2.4 Entropy and corrected redundancy ratio
print("\n─── 2.4 Entropy and corrected redundancy ratio ───")
fd_bins = pd.qcut(df_clean['FD'], q=N_BINS_FD, labels=False, duplicates='drop')
n_bins  = fd_bins.nunique()
probs   = fd_bins.value_counts(normalize=True).values
H_FD    = float(-(probs * np.log2(probs)).sum())
print(f"  H(FD_disc) [{n_bins} quantile bins] = {H_FD:.4f} bits  "
      f"(max = log2({n_bins}) = {np.log2(n_bins):.4f})")

H_cond = max(0.0, H_FD - mi_joint)
print(f"  H(FD_disc) - I(FD ; BF) = {H_cond:.4f} bits  "
      f"(NOT a redundancy ratio: mixes discrete H with continuous MI)")

# Honest comparison: continuous-MI vs continuous differential entropy.
# For standardized FD (sigma=1) approximately Gaussian, H_diff = 0.5 * log2(2*pi*e).
H_diff_FD            = 0.5 * np.log2(2 * np.pi * np.e)
redundancy_corrected = max(0.0, 1.0 - mi_joint / H_diff_FD)
shared_pct           = (1.0 - redundancy_corrected) * 100
print(f"  H_diff(FD) for N(0,1)         = {H_diff_FD:.4f} bits")
print(f"  Redundancy ratio (corrected)  = 1 - I(FD;BF)/H_diff(FD) = {redundancy_corrected:.3f}")
print(f"    → Big Five carries {shared_pct:.0f}% of FD's information;")
print(f"      remaining {redundancy_corrected*100:.0f}% is idiosyncratic / noise.")

results_b2.append({'metric': 'H_FD_disc', 'variable': f'{n_bins}_quantile_bins',
                   'value_bits': round(H_FD, 4),
                   'interpretation': 'entropy of discretized FD'})
results_b2.append({'metric': 'H_FD_given_BF_approx', 'variable': 'BF_full(5_vars)',
                   'value_bits': round(H_cond, 4),
                   'interpretation': 'H(FD_disc) - I(FD_cont ; BF_full); not a clean ratio numerator'})
results_b2.append({'metric': 'H_diff_FD', 'variable': 'standardized_FD',
                   'value_bits': round(H_diff_FD, 4),
                   'interpretation': 'differential entropy 0.5*log2(2*pi*e) for N(0,1) Gaussian'})
results_b2.append({'metric': 'Redundancy_ratio_corrected', 'variable': 'BF_full(5_vars)',
                   'value_bits': round(redundancy_corrected, 4),
                   'interpretation': (
                       f'1 - I(FD;BF)/H_diff(FD); {shared_pct:.0f}% of FD information '
                       f'shared with Big Five, the rest {100-shared_pct:.0f}% idiosyncratic/noise'
                   )})

# 2.5 Ranking comparison: MI vs |β|
print("\n─── 2.5 Ranking: univariate MI vs |β| from full OLS ───")
beta_abs = {v: abs(m_full.params[v]) for v in BF_VARS}
print(f"  {'Big Five':<25s} {'|β|':>8s} {'MI(bits)':>10s}")
for v in BF_VARS:
    print(f"  {BF_LABELS[v]:<25s} {beta_abs[v]:>8.3f} {mi_uni_dict[v]:>10.4f}")
mi_rank   = sorted(BF_VARS, key=lambda x: -mi_uni_dict[x])
beta_rank = sorted(BF_VARS, key=lambda x: -beta_abs[x])
agree     = sum(int(a == b) for a, b in zip(mi_rank, beta_rank))
print(f"\n  MI ranking:   {[BF_LABELS[v] for v in mi_rank]}")
print(f"  |β| ranking:  {[BF_LABELS[v] for v in beta_rank]}")
print(f"  Position-by-position agreement: {agree}/5")
print(f"  → {'Linear OLS captures the relationship' if agree == 5 else 'Some non-linear MI dependence not captured by OLS ranking'}")
results_b2.append({'metric': 'MI_vs_beta_rank_agreement', 'variable': 'all_BF',
                   'value_bits': agree,
                   'interpretation': f'{agree}/5 positions match between MI and |β| rankings'})

pd.DataFrame(results_b2).to_csv(OUTPUT_DIR / 'fd_redundancy_information.csv', index=False)
print(f"\n  Saved: fd_redundancy_information.csv")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FD REDUNDANCY ANALYSIS COMPLETE")
print("=" * 70)
print(f"Linear:    OLS full R² = {m_full.rsquared:.3f}  "
      f"(reduced BF_Ex+BF_Em R² = {m_red.rsquared:.3f})")
print(f"Joint MI:  I(FD ; BF_full) = {mi_joint:.3f} bits")
print(f"Redundancy (corrected): 1 - I/H_diff(FD) = {redundancy_corrected:.3f}  "
      f"→ Big Five carries {shared_pct:.0f}% of FD information")
