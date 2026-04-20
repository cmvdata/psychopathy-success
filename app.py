import streamlit as st
import pandas as pd
import numpy as np

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Psychopathy & Professional Success",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {background-color: #1a1a2e;}
    [data-testid="stSidebar"] * {color: #e0e0e0 !important;}

    /* Main background */
    .main {background-color: #f8f9fa;}

    /* Section headers */
    h1 {color: #1a1a2e; font-family: 'Georgia', serif;}
    h2 {color: #16213e; border-bottom: 2px solid #e94560; padding-bottom: 6px;}
    h3 {color: #0f3460;}

    /* Key finding box */
    .finding-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 20px 25px;
        border-radius: 10px;
        border-left: 5px solid #e94560;
        margin: 15px 0;
        font-size: 1.05em;
        line-height: 1.6;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .metric-value {font-size: 1.8em; font-weight: bold; color: #e94560;}
    .metric-label {font-size: 0.85em; color: #6c757d; margin-top: 4px;}

    /* CV bullet */
    .cv-bullet {
        background: #f0f4ff;
        border-left: 4px solid #0f3460;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.95em;
        line-height: 1.5;
    }

    /* Warning box */
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.93em;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Navigation")
    st.markdown("---")
    section = st.radio(
        "Go to section:",
        [
            "🏠 Overview",
            "1️⃣  Replication",
            "2️⃣  Factor Analysis (EFA)",
            "3️⃣  Model Comparison",
            "4️⃣  Causal Inference",
            "5️⃣  Heterogeneous Effects",
            "6️⃣  Economic Interpretation",
            "📄 CV Bullets",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    **Data source**  
    Eisenbarth, Hart & Sedikides (2018)  
    *Journal of Economic Psychology*  
    [OSF Repository ↗](https://osf.io/tgujv)
    """)
    st.markdown("---")
    st.markdown("""
    **Pipeline**  
    `01_replication.py`  
    `02_extension.py`  
    `03_causal_inference.py`  
    `04_model_comparison.py`  
    `05_heterogeneity.py`
    """)

# ─── OVERVIEW ─────────────────────────────────────────────────────────────────
if section == "🏠 Overview":
    st.title("Psychopathy & Professional Success")
    st.subheader("A Causal & Predictive Extension of Eisenbarth, Hart & Sedikides (2018)")

    st.markdown("""
    This project replicates and extends the peer-reviewed article *"Do psychopathic traits predict professional success?"*
    using **classical psychometrics**, **machine learning**, and **causal inference** on the original dataset (N = 439).
    """)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">439</div>
            <div class="metric-label">Participants</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">Analysis Scripts</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">14</div>
            <div class="metric-label">Figures</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">Causal Estimators</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="finding-box">
    <b>Core Contribution:</b> This project demonstrates that <b>aggregation bias</b> and <b>selection on observables</b>
    jointly explain the apparent positive relationship between Fearless Dominance and professional success.
    The naive positive effect (+0.63) disappears after propensity score matching (−0.97) and turns negative
    under doubly-robust estimation (−1.54), revealing that the correlation is driven by confounding
    personality traits — not by the psychopathic trait itself.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Narrative Arc")
    cols = st.columns(6)
    steps = [
        ("📊", "Replication", "Exact match with original paper"),
        ("🔬", "EFA", "Validate construct structure"),
        ("⚖️", "ML Models", "Aggregate vs. dimensions"),
        ("🎯", "Causal", "PSM + IPW + ATT"),
        ("👥", "Heterogeneity", "By gender & experience"),
        ("💼", "Economics", "Signaling vs. productivity"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)

# ─── SECTION 1: REPLICATION ───────────────────────────────────────────────────
elif section == "1️⃣  Replication":
    st.header("1. Replication — Classical Psychometrics")

    st.markdown("""
    Using the original data from the [Open Science Framework](https://osf.io/tgujv), we replicated
    the core findings of Eisenbarth et al. (2018) with exact numerical precision.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">α = 0.779</div>
            <div class="metric-label">Professional Satisfaction<br>(paper: 0.78 ✓)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">α = 0.483</div>
            <div class="metric-label">Material Success<br>(paper: 0.48 ✓)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">α = 0.732</div>
            <div class="metric-label">PPI-R Total<br>(paper: 0.73 ✓)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Zero-Order Correlation Matrix")
    st.image("figures/fig1_correlation_heatmap.png", use_container_width=True)
    st.caption("Replication of Table 3 from Eisenbarth et al. (2018). FD correlates positively with professional satisfaction (r = 0.13**), while SCI correlates negatively (r = −0.14**).")

    st.markdown("---")
    st.subheader("Regression Coefficients (SEM Approximation)")
    st.image("figures/fig2_regression_coefficients.png", use_container_width=True)
    st.caption("Replication of Table 4. When controlling for the Big Five, the effects of psychopathic traits are significantly attenuated.")

# ─── SECTION 2: EFA ───────────────────────────────────────────────────────────
elif section == "2️⃣  Factor Analysis (EFA)":
    st.header("2. Exploratory Factor Analysis (EFA)")

    st.markdown("""
    We applied a true EFA using `factor_analyzer` with **oblimin rotation** on all 40 items of the PPI-R.
    This tests whether the instrument's structure supports the three-dimensional model claimed by the authors.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">χ² = 4821.99</div>
            <div class="metric-label">Bartlett's test (p < 0.001)<br>Factor structure confirmed ✓</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">KMO = 0.807</div>
            <div class="metric-label">Sampling adequacy<br>Meritorious ✓</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">10 factors</div>
            <div class="metric-label">Eigenvalue > 1 (Kaiser)<br>Clearly multidimensional</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.image("figures/fig11_efa_real.png", use_container_width=True)
    st.caption("Left: Scree plot showing eigenvalues for the 40 PPI-R items. Right: Factor loadings heatmap for the 2-factor oblimin solution (top 20 items).")

    st.markdown("""
    <div class="finding-box">
    <b>Key finding:</b> The PPI-R is <b>not unidimensional</b>. The scree plot and factor loadings confirm
    that Fearless Dominance (FD) and Self-Centered Impulsivity (SCI) load on separate, orthogonal factors.
    Aggregating them into a single "psychopathy score" is therefore a form of <b>feature engineering error</b>:
    it mixes signals with opposite directions and destroys predictive capacity.
    </div>
    """, unsafe_allow_html=True)

# ─── SECTION 3: MODEL COMPARISON ──────────────────────────────────────────────
elif section == "3️⃣  Model Comparison":
    st.header("3. Formal Model Comparison")

    st.markdown("""
    We trained OLS and XGBoost models with **repeated cross-validation** (5-fold × 20 repeats)
    to compare the predictive power of the aggregated PPI score vs. the separated latent dimensions.
    """)

    try:
        df_comp = pd.read_csv("formal_model_comparison.csv")

        for outcome_label in df_comp['Outcome'].unique():
            st.subheader(f"Outcome: {outcome_label}")
            sub = df_comp[df_comp['Outcome'] == outcome_label][['Model', 'CV_R2', 'CV_RMSE', 'CV_MAE']].copy()
            sub.columns = ['Model', 'CV R²', 'CV RMSE', 'CV MAE']
            sub = sub.round(4)

            # Highlight best row
            best_r2 = sub['CV R²'].idxmax()
            best_rmse = sub['CV RMSE'].idxmin()

            def highlight_best(row):
                styles = [''] * len(row)
                if row.name == best_r2:
                    styles[1] = 'background-color: #d4edda; font-weight: bold'
                if row.name == best_rmse:
                    styles[2] = 'background-color: #d4edda; font-weight: bold'
                return styles

            st.dataframe(
                sub.style.apply(highlight_best, axis=1),
                use_container_width=True,
                hide_index=True,
            )
    except Exception:
        st.warning("Model comparison CSV not found. Run 04_model_comparison.py first.")

    st.markdown("---")
    st.image("figures/fig12_formal_model_comparison.png", use_container_width=True)
    st.caption("Model comparison across all specifications. Green highlights indicate best performance per metric.")

    st.markdown("""
    <div class="finding-box">
    <b>Key finding:</b> Separating the three PPI-R dimensions (FD, SCI, CO) consistently yields
    <b>lower RMSE and higher R²</b> than using the aggregated total score — across both OLS and XGBoost.
    This is the empirical proof of aggregation bias: the aggregate score is a noisier predictor
    than its components.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Aggregation Bias — The Core Problem")
    st.image("figures/fig7_aggregation_bias.png", use_container_width=True)
    st.caption("FD (r = +0.13) and SCI (r = −0.14) cancel each other when summed, producing a spurious near-zero correlation for the total score (r = −0.03).")

# ─── SECTION 4: CAUSAL INFERENCE ──────────────────────────────────────────────
elif section == "4️⃣  Causal Inference":
    st.header("4. Causal Inference: Propensity Score Matching")

    st.markdown("""
    To move beyond correlational analysis, we implemented a causal inference framework
    to estimate the **Average Treatment Effect on the Treated (ATT)** of having high
    Fearless Dominance on Professional Satisfaction.
    """)

    with st.expander("📐 Treatment Definition & Assumptions", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Treatment (T)**  
            T = 1 if FD > 75th percentile  
            T = 0 if FD ≤ 75th percentile  
            *(n_treated = 95, n_control = 344)*

            **Outcome (Y)**  
            Professional Satisfaction composite score
            """)
        with col2:
            st.markdown("""
            **Confounders (X)**  
            Age, Gender, Big Five personality traits (Extraversion, Agreeableness,
            Conscientiousness, Emotional Stability, Openness), Months in Job

            **Key Assumption**  
            Conditional Independence Assumption (CIA): treatment is independent
            of potential outcomes conditional on X.
            """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#f0ad4e">+0.634</div>
            <div class="metric-label">Naive ATT<br>(p = 0.347, n.s.)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#e94560">−0.966</div>
            <div class="metric-label">PSM ATT (matched)<br>(p = 0.212, n.s.)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#e94560">−1.543</div>
            <div class="metric-label">Doubly-Robust ATT<br>(p = 0.011 *)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.image("figures/fig9_causal_inference.png", use_container_width=True)
    st.caption("Left: Propensity score overlap (common support confirmed). Centre: Love plot showing covariate balance before/after matching. Right: ATT estimates across all three methods with 95% CIs.")

    st.markdown("""
    <div class="finding-box">
    <b>Key finding:</b> The naive positive effect of high FD (+0.63) <b>disappears after matching</b> (−0.97)
    and turns negative under doubly-robust estimation (−1.54). This means the apparent advantage of
    "boldness" in the raw data is entirely explained by selection: high-FD individuals are also
    higher in Extraversion and Emotional Stability — and <i>those</i> traits drive the career satisfaction,
    not the psychopathic component itself.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Sensitivity Analysis: Alternative Treatment Thresholds")
    st.image("figures/fig10_sensitivity_thresholds.png", use_container_width=True)
    st.caption("The naive ATT is only significant at the median split (p50), and disappears at higher thresholds. This confirms the result is not robust to the treatment definition.")

# ─── SECTION 5: HETEROGENEITY ─────────────────────────────────────────────────
elif section == "5️⃣  Heterogeneous Effects":
    st.header("5. Heterogeneous Treatment Effects")

    st.markdown("""
    We estimated the ATT within subgroups (by gender, experience, and age) to identify
    whether the effect of high FD is heterogeneous across the population.
    """)

    try:
        hte = pd.read_csv("heterogeneous_treatment_effects.csv")
        hte_display = hte[['Subgroup', 'Category', 'att', 'ci_lower', 'ci_upper', 'p_value', 'n_treated', 'n_control']].copy()
        hte_display.columns = ['Subgroup', 'Category', 'ATT', 'CI Lower', 'CI Upper', 'p-value', 'n treated', 'n control']
        hte_display = hte_display.round(3)

        def color_pvalue(val):
            if pd.isna(val):
                return ''
            if val < 0.05:
                return 'background-color: #d4edda; font-weight: bold'
            return ''

        st.dataframe(
            hte_display.style.applymap(color_pvalue, subset=['p-value']),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Green = statistically significant at p < 0.05. No subgroup shows a significant effect.")
    except Exception:
        st.warning("Heterogeneity CSV not found. Run 05_heterogeneity.py first.")

    st.markdown("---")
    st.image("figures/fig14_heterogeneous_effects.png", use_container_width=True)
    st.caption("ATT estimates with 95% bootstrap CIs across subgroups. The dashed vertical line marks zero; the dotted line marks the full-sample ATT.")

    st.markdown("""
    <div class="finding-box">
    <b>Key finding:</b> No subgroup shows a statistically significant ATT. The male subgroup shows
    a positive point estimate (+2.0, p = 0.11) that is worth noting — it is marginally non-significant
    and could reflect a gender-specific signaling dynamic in labor markets. However, the wide confidence
    interval [−0.14, +4.52] prevents strong conclusions. The null result is consistent across
    experience levels and age groups.
    </div>
    """, unsafe_allow_html=True)

# ─── SECTION 6: ECONOMIC INTERPRETATION ──────────────────────────────────────
elif section == "6️⃣  Economic Interpretation":
    st.header("6. Economic Interpretation & Threats to Identification")

    st.subheader("Economic Interpretation")
    st.markdown("""
    The naive results suggest that labor markets may reward traits associated with confidence and
    dominance (FD) in raw correlational terms, while penalizing impulsivity (SCI). This is consistent
    with **signaling and productivity-based explanations** in labor economics: boldness may act as a
    positive signal during hiring or promotion negotiations, whereas impulsivity negatively impacts
    actual productivity.

    However, our causal inference analysis reveals a critical nuance: once we control for selection
    into the "high FD" group based on observables — especially Extraversion and Emotional Stability —
    the positive naive effect disappears. This suggests that the apparent success of individuals with
    high Fearless Dominance is largely driven by **confounding personality traits** rather than the
    psychopathic component itself.

    In labor market terms: **what gets rewarded is confidence and emotional stability, not psychopathy**.
    The PPI-R's Fearless Dominance subscale captures both, and failing to separate them leads to
    a misattribution of the effect.
    """)

    st.markdown("---")
    st.subheader("Threats to Identification")

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Results should not be interpreted as strictly causal</b> due to the following threats:
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Omitted Variable Bias**  
        We cannot observe cognitive ability (IQ), socioeconomic background, or specific industry
        contexts — all of which likely affect both personality expression and career success.
        """)
    with col2:
        st.markdown("""
        **Reverse Causality**  
        Professional success and the accumulation of wealth/power may causally *increase*
        an individual's self-reported Fearless Dominance over time.
        """)
    with col3:
        st.markdown("""
        **Measurement Error**  
        Self-reported psychometric scales are subject to social desirability bias,
        particularly among successful professionals who may under-report impulsivity.
        """)

    st.markdown("---")
    st.subheader("Robustness Checks")
    st.image("figures/fig13_robustness.png", use_container_width=True)
    st.caption("FD coefficient stability across control specifications. The coefficient drops from +0.72 (no controls) to +0.55 (+ Big Five + demographics), consistent with the causal inference result.")

# ─── SECTION 7: CV BULLETS ────────────────────────────────────────────────────
elif section == "📄 CV Bullets":
    st.header("CV Bullet Points")
    st.markdown("Ready-to-copy bullets for your CV, LinkedIn, or job application.")
    st.markdown("---")

    bullets = [
        ("🎯 Causal Inference",
         "Implemented Propensity Score Matching (1:1 NN) and Inverse Probability Weighting (IPW) "
         "to estimate treatment effects of latent personality traits on professional outcomes, "
         "addressing selection bias under observational data constraints."),
        ("🔬 Psychometrics & Feature Engineering",
         "Conducted Exploratory Factor Analysis (EFA) with oblimin rotation to validate construct "
         "dimensionality, empirically demonstrating how aggregation bias (mixing orthogonal signals) "
         "destroys predictive capacity."),
        ("📈 Predictive Modeling",
         "Built and evaluated XGBoost and OLS models with repeated cross-validation (5-fold × 20), "
         "utilizing SHAP values to interpret non-linear feature contributions."),
        ("💻 Statistical Programming",
         "Developed a reproducible Python pipeline (pandas, scikit-learn, statsmodels, factor_analyzer) "
         "for end-to-end econometric and machine learning analysis on published behavioral data."),
    ]

    for icon_title, text in bullets:
        st.markdown(f"""
        <div class="cv-bullet">
        <b>{icon_title}:</b> {text}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("One-liner for interviews")
    st.markdown("""
    <div class="finding-box">
    "I replicated a peer-reviewed paper on personality and career success, then extended it with
    propensity score matching and SHAP-interpreted XGBoost models — showing that the naive positive
    effect of boldness on job satisfaction is entirely explained by confounding with Extraversion."
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Project links")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original paper data:** [OSF Repository](https://osf.io/tgujv)")
    with col2:
        st.markdown("**Paper:** Eisenbarth, Hart & Sedikides (2018), *Journal of Economic Psychology*")
