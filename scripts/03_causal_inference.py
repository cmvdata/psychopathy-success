"""
=============================================================================
PART 3 — CAUSAL INFERENCE (REFACTORED)
Latent Personality Constructs and Professional Success
=============================================================================

Estimators of the ATT of high Fearless Dominance on Professional Satisfaction:
  1. Caliper PSM 1:1 NN without replacement (caliper = 0.2 * SD(logit PS))
  2. IPW with ATT weights and p1/p99 trimming
  3. Augmented IPW (Robins-Rotnitzky-Zhao, doubly robust)

Main analysis: T = 1 if FD > P50 (median).
Sensitivity:   thresholds P50, P67, P75, P80 x 3 estimators.

Propensity score (statsmodels.Logit, 16 parameters total):
  - 8 linear:    Age, Gender, Big Five (Ex/Ag/Co/Em/Op), Months in Job
  - 7 quadratic: Age^2, Big Five^2, MonthsInJob^2  (continuous covariates only)
  - 0 interactions (removed; see PS_INTERACTIONS comment for rationale)

Outcome model for AIPW (mu_0):
  - Linear-only OLS on the 8 confounders. Deliberately simpler than the PS
    so the two models do not share the same nonlinear misspecification —
    preserving the doubly-robust property: if either model is correct, the
    AIPW estimator is consistent.

Bootstrap (N=500) re-fits the propensity-score Logit (and mu_0 in AIPW)
in each replicate for IPW and AIPW. Caliper PSM uses paired-difference
bootstrap on matched pairs (Abadie & Imbens 2008 caveat: bootstrap with
re-fit + re-match is invalid for matching estimators).
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'output' / 'df_processed.csv'
OUTPUT_DIR  = ROOT / 'output'
FIGURES_DIR = ROOT / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
OUTCOME = 'ProfSat_composite'
CONTINUOUS_CONFOUNDERS = ['Age', 'BF_Ex', 'BF_Ag', 'BF_Co', 'BF_Em', 'BF_Op', 'MonthsInJob']
BINARY_CONFOUNDERS     = ['Gender_male']
CONFOUNDERS = CONTINUOUS_CONFOUNDERS + BINARY_CONFOUNDERS

CONFOUNDER_LABELS = {
    'Age':         'Age',
    'BF_Ex':       'Extraversion',
    'BF_Ag':       'Agreeableness',
    'BF_Co':       'Conscientiousness',
    'BF_Em':       'Emotional Stability',
    'BF_Op':       'Openness',
    'MonthsInJob': 'Months in Job',
    'Gender_male': 'Gender (Male)',
}

# Pre-specified PS interactions: REMOVED (16-parameter spec).
# Diagnostic at P50 showed adding 3 theory-justified interactions changed
# McFadden Pseudo-R^2 by only +0.007 (0.447 -> 0.455) and reduced obs with
# ps in [0.05, 0.95] by 11 (308 -> 297). The PS bimodality and overlap
# failure are structural — driven by FD's strong correlation with Big Five
# (r ~ 0.65 with Emotional Stability, r ~ 0.49 with Extraversion) — not by
# interaction-induced overfitting on n=409, ~200 treated. The simpler
# 16-parameter spec is preferred.
PS_INTERACTIONS = []

THRESHOLDS     = [50, 67, 75, 80]
MAIN_THRESHOLD = 50
N_BOOT         = 500
SEED           = 42
CALIPER_FACTOR = 0.2


# ─── Helpers ──────────────────────────────────────────────────────────────────
def standardized_mean_difference(x_t, x_c, w_t=None, w_c=None):
    """SMD with optional weights (for IPW-weighted balance assessment)."""
    if w_t is None:
        m_t, v_t = float(np.mean(x_t)), float(np.var(x_t, ddof=1))
    else:
        m_t = float(np.average(x_t, weights=w_t))
        v_t = float(np.average((x_t - m_t) ** 2, weights=w_t))
    if w_c is None:
        m_c, v_c = float(np.mean(x_c)), float(np.var(x_c, ddof=1))
    else:
        m_c = float(np.average(x_c, weights=w_c))
        v_c = float(np.average((x_c - m_c) ** 2, weights=w_c))
    pool = np.sqrt((v_t + v_c) / 2)
    return 0.0 if pool == 0 else (m_t - m_c) / pool


def build_ps_design(data):
    """
    PS design matrix:
      - 8 linear    (continuous + binary Gender)
      - 7 quadratic (continuous only, after standardization)
      - 0 interactions (see PS_INTERACTIONS comment for the rationale)
    Total: 16 parameters incl. intercept.
    """
    scaler = StandardScaler()
    cont = pd.DataFrame(
        scaler.fit_transform(data[CONTINUOUS_CONFOUNDERS]),
        columns=CONTINUOUS_CONFOUNDERS,
        index=data.index,
    )
    binr = data[BINARY_CONFOUNDERS]

    feats = {}
    for c in CONTINUOUS_CONFOUNDERS:
        feats[c]          = cont[c]
        feats[f'{c}__sq'] = cont[c] ** 2
    for c in BINARY_CONFOUNDERS:
        feats[c] = binr[c]
    for c1, c2 in PS_INTERACTIONS:
        feats[f'{c1}__x__{c2}'] = cont[c1] * cont[c2]

    return sm.add_constant(pd.DataFrame(feats, index=data.index))


def build_outcome_design(data):
    """
    Linear-only design for the AIPW outcome model mu_0.
    Deliberately simpler than the PS to preserve specification diversity:
    if both PS and mu_0 carry the same nonlinearities, their misspecification
    errors covary and the doubly-robust property degrades.
    """
    return sm.add_constant(data[CONFOUNDERS].copy())


def estimate_propensity(data, T):
    """Fit Logit and return clipped PS, logit(PS), and the model."""
    X = build_ps_design(data)
    model = sm.Logit(T, X).fit(disp=0, maxiter=200)
    ps = np.clip(model.predict(X).values, 1e-6, 1 - 1e-6)
    logit_ps = np.log(ps / (1 - ps))
    return ps, logit_ps, model


# ─── Caliper Matching ─────────────────────────────────────────────────────────
def caliper_match_indices(T, ps, logit_ps, factor=CALIPER_FACTOR, seed=SEED):
    """
    Greedy 1:1 NN matching on logit(PS) without replacement.
    Caliper = factor * SD(logit_ps).
    Treated units with no admissible match (within caliper) are dropped.
    """
    sd_logit = float(np.std(logit_ps, ddof=1))
    caliper  = factor * sd_logit

    treated = np.where(T == 1)[0]
    control = list(np.where(T == 0)[0])

    rng = np.random.default_rng(seed)
    # Random tie-breaking, then sort by descending PS (hardest first).
    order = rng.permutation(treated)
    order = order[np.argsort(-ps[order], kind='stable')]

    matched_t, matched_c = [], []
    for ti in order:
        if not control:
            break
        diffs = np.abs(logit_ps[control] - logit_ps[ti])
        j = int(np.argmin(diffs))
        if diffs[j] <= caliper:
            matched_t.append(int(ti))
            matched_c.append(int(control[j]))
            control.pop(j)
    return np.array(matched_t, dtype=int), np.array(matched_c, dtype=int), caliper


def att_caliper_psm(T, Y, ps, logit_ps, n_boot=N_BOOT, seed=SEED):
    mt, mc, cal = caliper_match_indices(T, ps, logit_ps, seed=seed)
    n_t_total   = int(T.sum())
    n_dropped   = n_t_total - len(mt)

    if len(mt) < 5:
        return {'att': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan, 'n_matched': len(mt),
                'n_dropped_treated': n_dropped, 'caliper': cal,
                'matched_t': mt, 'matched_c': mc}

    yt, yc = Y[mt], Y[mc]
    diff = yt - yc
    att  = float(np.mean(diff))

    # Paired t-test for p-value (analytical, exact under matching with fixed PS)
    p_val = float(ttest_rel(yt, yc).pvalue)

    # Bootstrap CI on matched-pair differences (PS held fixed; standard
    # practice — bootstrap with re-fit + re-match is invalid in general,
    # Abadie & Imbens 2008).
    rng = np.random.default_rng(seed + 1)
    n = len(diff)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = float(np.mean(diff[idx]))
    ci_l, ci_u = np.percentile(boot, [2.5, 97.5])

    return {'att': att, 'ci_lower': float(ci_l), 'ci_upper': float(ci_u),
            'p_value': p_val, 'n_matched': len(mt),
            'n_dropped_treated': n_dropped, 'caliper': float(cal),
            'matched_t': mt, 'matched_c': mc}


# ─── IPW (ATT weights, p1/p99 trim, full PS re-fit in bootstrap) ─────────────
def _ipw_point_estimate(T, Y, ps):
    """ATT-IPW point estimate with p1/p99 trim. Returns (ATT, keep_mask)."""
    p1, p99 = np.percentile(ps, [1, 99])
    keep = (ps >= p1) & (ps <= p99)
    Tk, Yk, psk = T[keep], Y[keep], ps[keep]
    if Tk.sum() < 5 or (1 - Tk).sum() < 5:
        return np.nan, keep
    y1 = float(np.mean(Yk[Tk == 1]))
    w_c = psk[Tk == 0] / (1 - psk[Tk == 0])
    if w_c.sum() == 0:
        return np.nan, keep
    y0 = float(np.average(Yk[Tk == 0], weights=w_c))
    return y1 - y0, keep


def att_ipw(df_clean, T, Y, ps, n_boot=N_BOOT, seed=SEED):
    """ATT via IPW with p1/p99 trim. Bootstrap re-fits the PS Logit per replicate."""
    att, keep = _ipw_point_estimate(T, Y, ps)
    if np.isnan(att):
        return {'att': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan, 'n_kept': int(keep.sum())}

    rng = np.random.default_rng(seed + 2)
    n   = len(df_clean)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        df_b = df_clean.iloc[idx].reset_index(drop=True)
        T_b  = T[idx]
        Y_b  = Y[idx]
        if T_b.sum() < 10 or (1 - T_b).sum() < 10:
            continue
        try:
            ps_b, _, _ = estimate_propensity(df_b, T_b)
        except Exception:
            continue
        att_b, _ = _ipw_point_estimate(T_b, Y_b, ps_b)
        if not np.isnan(att_b):
            boot.append(att_b)

    if len(boot) < 100:
        return {'att': att, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan, 'n_kept': int(keep.sum())}

    ci_l, ci_u = np.percentile(boot, [2.5, 97.5])
    se = float(np.std(boot, ddof=1))
    z  = att / se if se > 0 else np.nan
    pv = float(2 * (1 - norm.cdf(abs(z)))) if not np.isnan(z) else np.nan

    return {'att': att, 'ci_lower': float(ci_l), 'ci_upper': float(ci_u),
            'p_value': pv, 'n_kept': int(keep.sum())}


# ─── Augmented IPW (Robins-Rotnitzky-Zhao) ───────────────────────────────────
def _aipw_point_estimate(df_clean, T, Y, ps):
    """AIPW (RRZ) ATT with p1/p99 trim and linear-only mu_0."""
    p1, p99 = np.percentile(ps, [1, 99])
    keep = (ps >= p1) & (ps <= p99)
    sub  = df_clean[keep].reset_index(drop=True)
    Tk, Yk, psk = T[keep], Y[keep], ps[keep]
    if Tk.sum() < 5 or (1 - Tk).sum() < 10:
        return np.nan, keep

    Xk = build_outcome_design(sub).values
    try:
        mu0     = LinearRegression().fit(Xk[Tk == 0], Yk[Tk == 0])
        mu0_hat = mu0.predict(Xk)
    except Exception:
        return np.nan, keep

    n1 = int(Tk.sum())
    contrib = (Tk * (Yk - mu0_hat)
               - (1 - Tk) * (psk / (1 - psk)) * (Yk - mu0_hat))
    return float(contrib.sum() / n1), keep


def att_aipw(df_clean, T, Y, ps, n_boot=N_BOOT, seed=SEED):
    """
    AIPW ATT. Bootstrap re-fits BOTH the PS Logit and the linear mu_0 OLS
    in each replicate (full doubly-robust uncertainty propagation).
    """
    att, keep = _aipw_point_estimate(df_clean, T, Y, ps)
    if np.isnan(att):
        return {'att': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan, 'n_kept': int(keep.sum())}

    rng = np.random.default_rng(seed + 3)
    n = len(df_clean)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        df_b = df_clean.iloc[idx].reset_index(drop=True)
        T_b  = T[idx]
        Y_b  = Y[idx]
        if T_b.sum() < 10 or (1 - T_b).sum() < 10:
            continue
        try:
            ps_b, _, _ = estimate_propensity(df_b, T_b)
        except Exception:
            continue
        att_b, _ = _aipw_point_estimate(df_b, T_b, Y_b, ps_b)
        if not np.isnan(att_b):
            boot.append(att_b)

    if len(boot) < 100:
        return {'att': att, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'p_value': np.nan, 'n_kept': int(keep.sum())}

    ci_l, ci_u = np.percentile(boot, [2.5, 97.5])
    se = float(np.std(boot, ddof=1))
    z  = att / se if se > 0 else np.nan
    pv = float(2 * (1 - norm.cdf(abs(z)))) if not np.isnan(z) else np.nan

    return {'att': att, 'ci_lower': float(ci_l), 'ci_upper': float(ci_u),
            'p_value': pv, 'n_kept': int(keep.sum())}


# ─── Balance reporting ────────────────────────────────────────────────────────
def balance_unweighted(data, T):
    rows = []
    for v in CONFOUNDERS:
        x = data[v].values
        rows.append({
            'Variable':   CONFOUNDER_LABELS[v],
            'SMD_before': standardized_mean_difference(x[T == 1], x[T == 0]),
        })
    return pd.DataFrame(rows)


def balance_after_caliper(data, mt, mc):
    rows = []
    for v in CONFOUNDERS:
        x = data[v].values
        rows.append({
            'Variable':  CONFOUNDER_LABELS[v],
            'SMD_after': standardized_mean_difference(x[mt], x[mc]),
        })
    return pd.DataFrame(rows)


def balance_after_ipw(data, T, ps):
    """ATT-weighted SMDs (control weight = ps/(1-ps); treated weight = 1) after p1/p99 trim."""
    p1, p99 = np.percentile(ps, [1, 99])
    keep = (ps >= p1) & (ps <= p99)
    sub  = data[keep]
    Tk, psk = T[keep], ps[keep]
    w_t = np.ones(int(Tk.sum()))
    w_c = psk[Tk == 0] / (1 - psk[Tk == 0])
    rows = []
    for v in CONFOUNDERS:
        x = sub[v].values
        rows.append({
            'Variable':  CONFOUNDER_LABELS[v],
            'SMD_after': standardized_mean_difference(
                x[Tk == 1], x[Tk == 0], w_t, w_c
            ),
        })
    return pd.DataFrame(rows)


# ─── Love plot ────────────────────────────────────────────────────────────────
def love_plot(balance_df, title, outfile):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = np.arange(len(balance_df))
    ax.scatter(balance_df['SMD_before'], y, color='#D7191C', s=90, label='Before', zorder=3)
    ax.scatter(balance_df['SMD_after'],  y, color='#1A9641', s=90, marker='D',
               label='After', zorder=3)
    for i, (b, a) in enumerate(zip(balance_df['SMD_before'], balance_df['SMD_after'])):
        ax.plot([b, a], [i, i], color='gray', alpha=0.5, linewidth=1)
    ax.axvline(0, color='black', linewidth=0.8)
    for x in (-0.1, 0.1):
        ax.axvline(x, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(balance_df['Variable'], fontsize=10)
    ax.set_xlabel('Standardized Mean Difference', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Pipeline per threshold ───────────────────────────────────────────────────
def run_threshold(df_clean, pct):
    threshold_value = float(df_clean['FD'].quantile(pct / 100))
    T = (df_clean['FD'] > threshold_value).astype(int).values
    Y = df_clean[OUTCOME].values

    if T.sum() < 10 or (1 - T).sum() < 10:
        return None

    ps, logit_ps, _ = estimate_propensity(df_clean, T)

    res_psm  = att_caliper_psm(T, Y, ps, logit_ps)
    res_ipw  = att_ipw(df_clean, T, Y, ps)
    res_aipw = att_aipw(df_clean, T, Y, ps)

    return {
        'threshold':  pct,
        'fd_cutoff':  threshold_value,
        'n_treated':  int(T.sum()),
        'n_control':  int((1 - T).sum()),
        'ps':         ps,
        'logit_ps':   logit_ps,
        'T':          T,
        'Y':          Y,
        'caliper':    res_psm,
        'ipw':        res_ipw,
        'aipw':       res_aipw,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 3 — CAUSAL INFERENCE (Caliper PSM + IPW + AIPW)")
print("=" * 70)

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
analysis_cols = [OUTCOME, 'FD'] + CONFOUNDERS
df_clean = df[analysis_cols].dropna().reset_index(drop=True).copy()
print(f"Complete cases:  n = {len(df_clean)}")
print(f"Outcome:         {OUTCOME}")
print(f"Confounders:     {CONFOUNDERS}")
print(f"PS interactions: {PS_INTERACTIONS}")
print(f"Bootstrap:       N = {N_BOOT} (full PS re-fit per replicate)")

# ─── 2. MAIN ANALYSIS (P50) ───────────────────────────────────────────────────
print("\n" + "─" * 60)
print(f"2. MAIN ANALYSIS — Treatment: FD > P{MAIN_THRESHOLD} (median)")
print("─" * 60)

main_result = run_threshold(df_clean, MAIN_THRESHOLD)
if main_result is None:
    raise RuntimeError("Main threshold has insufficient sample sizes.")

print(f"  FD cutoff (P{MAIN_THRESHOLD}): {main_result['fd_cutoff']:.3f}")
print(f"  Treated: n = {main_result['n_treated']}")
print(f"  Control: n = {main_result['n_control']}")

for name, key in [('Caliper PSM 1:1 NN',   'caliper'),
                  ('IPW (ATT, p1/p99)',    'ipw'),
                  ('AIPW (Doubly Robust)', 'aipw')]:
    r = main_result[key]
    if np.isnan(r['att']):
        print(f"  {name:24s}: ATT = NA")
        continue
    print(f"  {name:24s}: ATT = {r['att']:+.3f}, "
          f"95% CI [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}], "
          f"p = {r['p_value']:.4f}")

# ─── 3. BALANCE — MAIN ANALYSIS ───────────────────────────────────────────────
print("\n" + "─" * 60)
print("3. COVARIATE BALANCE — MAIN ANALYSIS (P50)")
print("─" * 60)

bal_pre = balance_unweighted(df_clean, main_result['T'])

# Caliper
bal_cal_after = balance_after_caliper(df_clean,
                                      main_result['caliper']['matched_t'],
                                      main_result['caliper']['matched_c'])
bal_caliper = bal_pre.merge(bal_cal_after, on='Variable')
bal_caliper['Balanced_after'] = bal_caliper['SMD_after'].abs() < 0.10
bal_caliper.to_csv(OUTPUT_DIR / 'psm_caliper_balance.csv', index=False)
print("\n  Caliper PSM SMDs (unweighted, matched pairs):")
for _, r in bal_caliper.iterrows():
    flag = 'OK' if r['Balanced_after'] else '**'
    print(f"    {r['Variable']:25s} before = {r['SMD_before']:+.3f}  "
          f"after = {r['SMD_after']:+.3f}  [{flag}]")

# IPW
bal_ipw_after = balance_after_ipw(df_clean, main_result['T'], main_result['ps'])
bal_ipw = bal_pre.merge(bal_ipw_after, on='Variable')
bal_ipw['Balanced_after'] = bal_ipw['SMD_after'].abs() < 0.10
bal_ipw.to_csv(OUTPUT_DIR / 'ipw_balance.csv', index=False)
print("\n  IPW SMDs (ATT-weighted controls, p1/p99 trim):")
for _, r in bal_ipw.iterrows():
    flag = 'OK' if r['Balanced_after'] else '**'
    print(f"    {r['Variable']:25s} before = {r['SMD_before']:+.3f}  "
          f"after = {r['SMD_after']:+.3f}  [{flag}]")

# Love plots
love_plot(bal_caliper,
          f'Covariate Balance — Caliper PSM (P{MAIN_THRESHOLD})\n'
          f'caliper = {CALIPER_FACTOR}*SD(logit PS) = {main_result["caliper"]["caliper"]:.3f}, '
          f'matched n = {main_result["caliper"]["n_matched"]}, '
          f'dropped treated = {main_result["caliper"]["n_dropped_treated"]}',
          FIGURES_DIR / 'love_plot_caliper.png')
print("\n  Saved: love_plot_caliper.png")

love_plot(bal_ipw,
          f'Covariate Balance — IPW ATT (P{MAIN_THRESHOLD})\n'
          f'p1/p99 trimmed, n_kept = {main_result["ipw"]["n_kept"]}',
          FIGURES_DIR / 'love_plot_ipw.png')
print("  Saved: love_plot_ipw.png")

# ─── 4. att_main.csv ──────────────────────────────────────────────────────────
main_rows = []
for name, key in [('Caliper PSM',          'caliper'),
                  ('IPW',                  'ipw'),
                  ('AIPW (Doubly Robust)', 'aipw')]:
    r = main_result[key]
    main_rows.append({
        'Estimator':  name,
        'Threshold':  MAIN_THRESHOLD,
        'ATT':        r['att'],
        'CI_lower':   r['ci_lower'],
        'CI_upper':   r['ci_upper'],
        'p_value':    r['p_value'],
        'n_used':     r.get('n_matched', r.get('n_kept', np.nan)),
    })
pd.DataFrame(main_rows).round(4).to_csv(OUTPUT_DIR / 'att_main.csv', index=False)
print("  Saved: att_main.csv")

# ─── 5. SENSITIVITY ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("5. SENSITIVITY — Thresholds x Estimators")
print("─" * 60)

sens_rows = []
for pct in THRESHOLDS:
    res = main_result if pct == MAIN_THRESHOLD else run_threshold(df_clean, pct)
    if res is None:
        continue
    for name, key in [('Caliper PSM',          'caliper'),
                      ('IPW',                  'ipw'),
                      ('AIPW (Doubly Robust)', 'aipw')]:
        r = res[key]
        sens_rows.append({
            'Threshold': pct,
            'Estimator': name,
            'n_treated': res['n_treated'],
            'ATT':       r['att'],
            'CI_lower':  r['ci_lower'],
            'CI_upper':  r['ci_upper'],
            'p_value':   r['p_value'],
        })
        if not np.isnan(r['att']):
            print(f"  P{pct:>2d}  {name:24s} ATT = {r['att']:+.3f} "
                  f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}], "
                  f"p = {r['p_value']:.4f}")

sens_df = pd.DataFrame(sens_rows)
sens_df.round(4).to_csv(OUTPUT_DIR / 'att_sensitivity.csv', index=False)
print("\n  Saved: att_sensitivity.csv")

# ─── 6. SENSITIVITY FOREST PLOT ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
estimator_order  = ['Caliper PSM', 'IPW', 'AIPW (Doubly Robust)']
estimator_color  = {'Caliper PSM': '#2C7BB6', 'IPW': '#1A9641',
                    'AIPW (Doubly Robust)': '#D7191C'}
estimator_offset = {'Caliper PSM': -0.25, 'IPW': 0.0,
                    'AIPW (Doubly Robust)': 0.25}

for i, pct in enumerate(THRESHOLDS):
    for est in estimator_order:
        sub = sens_df[(sens_df['Threshold'] == pct) & (sens_df['Estimator'] == est)]
        if sub.empty or pd.isna(sub.iloc[0]['ATT']):
            continue
        row = sub.iloc[0]
        ax.errorbar(
            row['ATT'], i + estimator_offset[est],
            xerr=[[row['ATT'] - row['CI_lower']],
                  [row['CI_upper'] - row['ATT']]],
            fmt='o', color=estimator_color[est], capsize=4,
            markersize=8, linewidth=2,
            label=est if i == 0 else None,
        )

ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_yticks(range(len(THRESHOLDS)))
ax.set_yticklabels([f'P{p}' for p in THRESHOLDS], fontsize=11)
ax.set_xlabel('ATT on Professional Satisfaction (95% bootstrap CI)', fontsize=11)
ax.set_ylabel('Treatment threshold (FD percentile)', fontsize=11)
ax.set_title('Sensitivity: ATT across thresholds and estimators',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'sensitivity_forest_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sensitivity_forest_plot.png")

# ─── 7. SUMMARY ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CAUSAL INFERENCE COMPLETE")
print("=" * 70)
print(f"\nMain analysis (FD > P{MAIN_THRESHOLD}):")
for name, key in [('Caliper PSM', 'caliper'),
                  ('IPW',         'ipw'),
                  ('AIPW',        'aipw')]:
    r = main_result[key]
    if np.isnan(r['att']):
        continue
    print(f"  {name:12s} ATT = {r['att']:+.3f}, "
          f"95% CI [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}], "
          f"p = {r['p_value']:.4f}")

n_unbalanced_cal = int((~bal_caliper['Balanced_after']).sum())
n_unbalanced_ipw = int((~bal_ipw['Balanced_after']).sum())
print(f"\nBalance (|SMD| < 0.10) on {len(CONFOUNDERS)} confounders:")
print(f"  Caliper PSM: {len(CONFOUNDERS) - n_unbalanced_cal}/{len(CONFOUNDERS)} balanced")
print(f"  IPW:         {len(CONFOUNDERS) - n_unbalanced_ipw}/{len(CONFOUNDERS)} balanced")
print(f"\nCaliper PSM dropped {main_result['caliper']['n_dropped_treated']} "
      f"of {main_result['n_treated']} treated units (no admissible match).")
