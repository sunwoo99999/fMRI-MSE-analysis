"""
bold_asl_04_mlm.py
Multilevel Modeling (MLM) analysis -- Section 2.7 of McDonough et al. 2019.

Paper's specification (lme4 equivalent):
  mse_diff ~ Timescale + MSE_pre + (1 + Timescale | subject)

  - Random intercept + random slope for Timescale (per subject)
  - Maximum likelihood estimation (not REML)
  - All timescales modeled simultaneously (long format)
  - Covariates: MSE_pre (available), Sex/IQ (not available in our dataset)

  Note: AR(1) autocorrelation within subject is approximated by the random
  slope (captures systematic within-subject timescale trends). True AR(1)
  requires lme4/nlme in R; statsmodels MixedLM does not support it natively.

Models implemented (matching Table 2 structure):
  Within-modality (BOLD or ASL, subjects with REST1+REST2):
    Model 1: MSE_REST2-REST1 ~ Timescale + MSE_REST1 + (1+Timescale|subject)
    Model 2: MSE_REST1       ~ Timescale               + (1+Timescale|subject)
    Model 3: MSE_REST2       ~ Timescale + MSE_REST1   + (1+Timescale|subject)

  Cross-modality (BOLD vs ASL at REST1, our adaptation):
    Model A: MSE_BOLD-ASL    ~ Timescale + MSE_ASL_pre + (1+Timescale|subject)
    Model B: MSE_ASL_REST1   ~ Timescale               + (1+Timescale|subject)
    Model C: MSE_BOLD_REST1  ~ Timescale + MSE_ASL_pre + (1+Timescale|subject)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCALE_COLS   = [f"scale_{s}" for s in range(1, 7)]

RSN_NAMES = [
    "medial_visual", "occipital_pole", "lateral_visual", "default_mode",
    "cerebellum", "sensorimotor", "auditory", "executive_ctrl",
    "right_frontoparietal", "left_frontoparietal",
]
DMN_IDX = 3


# ── helpers ──────────────────────────────────────────────────────────────────
def _average_directions(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["subject", "session", "modality", "rsn_idx", "rsn_name"],
                       as_index=False)[SCALE_COLS].mean())


def _to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Wide → long: one row per (subject, session, modality, rsn, timescale)."""
    return df_wide.melt(
        id_vars=["subject", "session", "modality", "rsn_idx", "rsn_name"],
        value_vars=SCALE_COLS,
        var_name="scale_str",
        value_name="mse",
    ).assign(timescale=lambda d: d["scale_str"].str.replace("scale_", "").astype(int)
    ).drop(columns="scale_str")


def _fit_mlm(endog, exog_df, groups, method="ml"):
    """
    Fit MLM with random intercept + random slope for Timescale.
    Falls back to random-intercept-only if random slope causes convergence failure.
    Returns (result, random_slope_used).
    """
    exog_re = exog_df[["intercept", "timescale"]]
    try:
        model  = MixedLM(endog, exog_df, groups=groups, exog_re=exog_re)
        result = model.fit(method=method, reml=False, disp=False)
        if not result.converged:
            raise ValueError("did not converge")
        return result, True
    except Exception:
        # fallback: random intercept only
        model  = MixedLM(endog, exog_df, groups=groups)
        result = model.fit(method=method, reml=False, disp=False)
        return result, False


def _extract_fixed(result, term: str):
    """Return (coef, se, z, p) for a fixed-effect term."""
    try:
        idx  = result.fe_params.index.get_loc(term)
        coef = result.fe_params.iloc[idx]
        se   = result.bse_fe.iloc[idx]
        z    = result.tvalues.iloc[idx]
        p    = result.pvalues.iloc[idx]
        return coef, se, z, p
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ── Within-modality MLM (Models 1–3) ─────────────────────────────────────────
def mlm_within_modality(df_avg: pd.DataFrame) -> pd.DataFrame:
    """
    For each modality × RSN: fit Models 1, 2, 3 following Table 2 structure.

    Model 1 DV  = REST2 - REST1  (post-pre difference)
    Model 2 DV  = REST1
    Model 3 DV  = REST2
    Covariate   = MSE_REST1 (for Models 1 and 3)
    Fixed pred  = Timescale
    """
    rows = []

    for modality in ["BOLD", "ASL"]:
        sub   = df_avg[df_avg.modality == modality]
        subs_both = (sub.groupby("subject")["session"]
                     .nunique().loc[lambda s: s == 2].index.tolist())
        if len(subs_both) < 3:
            print(f"  {modality}: only {len(subs_both)} subjects with REST1+REST2 -- need >=3, skipping")
            continue

        sub = sub[sub.subject.isin(subs_both)]

        for rsn_i in range(10):
            rsn_df = sub[sub.rsn_idx == rsn_i]
            r1 = (rsn_df[rsn_df.session == "REST1"]
                  .sort_values("subject").reset_index(drop=True))
            r2 = (rsn_df[rsn_df.session == "REST2"]
                  .sort_values("subject").reset_index(drop=True))

            # Build long-format diff table
            r1_long = _to_long(r1).rename(columns={"mse": "mse_r1"})
            r2_long = _to_long(r2).rename(columns={"mse": "mse_r2"})
            long    = r1_long.merge(
                r2_long[["subject", "timescale", "mse_r2"]],
                on=["subject", "timescale"]
            ).dropna(subset=["mse_r1", "mse_r2"])
            long["mse_diff"]    = long["mse_r2"] - long["mse_r1"]
            long["intercept"]   = 1.0
            long["timescale_z"] = (long["timescale"] - long["timescale"].mean()
                                   ) / long["timescale"].std(ddof=1)
            long["mse_r1_z"]    = (long["mse_r1"] - long["mse_r1"].mean()
                                   ) / long["mse_r1"].std(ddof=1)

            groups = long["subject"]

            for model_id, dv_col, include_pre in [
                ("Model1_diff",  "mse_diff", True),
                ("Model2_REST1", "mse_r1",   False),
                ("Model3_REST2", "mse_r2",   True),
            ]:
                endog  = long[dv_col]
                if include_pre and model_id != "Model2_REST1":
                    exog = long[["intercept", "timescale_z", "mse_r1_z"]]
                    exog.columns = ["intercept", "timescale", "mse_pre"]
                else:
                    exog = long[["intercept", "timescale_z"]]
                    exog.columns = ["intercept", "timescale"]

                res, rslope = _fit_mlm(endog, exog.copy(), groups)

                rec = dict(modality=modality, rsn_idx=rsn_i,
                           rsn_name=RSN_NAMES[rsn_i], model=model_id,
                           n_subjects=len(subs_both), random_slope=rslope,
                           converged=res.converged)
                for term in ["intercept", "timescale", "mse_pre"]:
                    c, se, z, p = _extract_fixed(res, term)
                    rec[f"β_{term}"]  = round(c,  4) if not np.isnan(c) else np.nan
                    rec[f"se_{term}"] = round(se, 4) if not np.isnan(se) else np.nan
                    rec[f"p_{term}"]  = round(p,  4) if not np.isnan(p) else np.nan
                rows.append(rec)

    return pd.DataFrame(rows)


# ── Cross-modality MLM (BOLD vs ASL) ─────────────────────────────────────────
def mlm_cross_modality(df_avg: pd.DataFrame) -> pd.DataFrame:
    """
    Adaptation: BOLD − ASL at REST1 ~ Timescale + (1+Timescale|subject).
    Mirrors paper's Model 1 structure; "Modality" replaces "Age".
    """
    rest1 = df_avg[df_avg.session == "REST1"]
    rows  = []

    for rsn_i in range(10):
        rsn_df = rest1[rest1.rsn_idx == rsn_i]
        bold   = rsn_df[rsn_df.modality == "BOLD"].sort_values("subject")
        asl    = rsn_df[rsn_df.modality == "ASL"].sort_values("subject")
        subs   = sorted(set(bold.subject) & set(asl.subject))
        if len(subs) < 3:
            if rsn_i == 0:
                print(f"  Cross-modality RSN{rsn_i}: only {len(subs)} paired subjects -- need >=3, skipping all RSNs")
            continue

        bold = bold[bold.subject.isin(subs)].sort_values("subject").reset_index(drop=True)
        asl  = asl[asl.subject.isin(subs)].sort_values("subject").reset_index(drop=True)

        # Long-format BOLD-ASL difference
        b_long  = _to_long(bold).rename(columns={"mse": "mse_bold"})
        a_long  = _to_long(asl).rename(columns={"mse": "mse_asl"})
        long    = b_long.merge(
            a_long[["subject", "timescale", "mse_asl"]],
            on=["subject", "timescale"]
        ).dropna(subset=["mse_bold", "mse_asl"])
        long["mse_diff"]    = long["mse_bold"] - long["mse_asl"]
        long["intercept"]   = 1.0
        long["timescale_z"] = (long["timescale"] - long["timescale"].mean()
                               ) / long["timescale"].std(ddof=1)
        long["mse_asl_z"]   = (long["mse_asl"] - long["mse_asl"].mean()
                               ) / long["mse_asl"].std(ddof=1)

        groups = long["subject"]

        for model_id, dv_col, include_pre in [
            ("ModelA_diff",      "mse_diff", True),
            ("ModelB_ASL_only",  "mse_asl",  False),
            ("ModelC_BOLD_only", "mse_bold", True),
        ]:
            endog  = long[dv_col]
            if include_pre and model_id != "ModelB_ASL_only":
                exog = long[["intercept", "timescale_z", "mse_asl_z"]]
                exog.columns = ["intercept", "timescale", "mse_pre"]
            else:
                exog = long[["intercept", "timescale_z"]]
                exog.columns = ["intercept", "timescale"]

            res, rslope = _fit_mlm(endog, exog.copy(), groups)

            rec = dict(rsn_idx=rsn_i, rsn_name=RSN_NAMES[rsn_i],
                       model=model_id, n_subjects=len(subs),
                       random_slope=rslope, converged=res.converged)
            for term in ["intercept", "timescale", "mse_pre"]:
                c, se, z, p = _extract_fixed(res, term)
                rec[f"β_{term}"]  = round(c,  4) if not np.isnan(c) else np.nan
                rec[f"se_{term}"] = round(se, 4) if not np.isnan(se) else np.nan
                rec[f"p_{term}"]  = round(p,  4) if not np.isnan(p) else np.nan
            rows.append(rec)

    return pd.DataFrame(rows)


# ── Visualization -- marginal plot (Figure 3 equivalent) ──────────────────────
def plot_marginal_diff(df_long_diff: pd.DataFrame,
                       title: str, fname: str) -> None:
    """
    Scatter + regression line of MSE diff (y) vs timescale (x),
    one line per subject.  Equivalent to paper's Figure 3.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for sub, grp in df_long_diff.groupby("subject"):
        ax.plot(grp["timescale"], grp["mse_diff"],
                color="steelblue", alpha=0.3, linewidth=0.8)
    # group mean
    mean = df_long_diff.groupby("timescale")["mse_diff"].mean()
    ax.plot(mean.index, mean.values, color="steelblue",
            linewidth=2.5, label="Group mean")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Timescale", fontsize=12)
    ax.set_ylabel("MSE Difference", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(sorted(df_long_diff["timescale"].unique()))
    ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Console Table 2 equivalent ───────────────────────────────────────────────
def print_table2(within_df: pd.DataFrame,
                 cross_df: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("TABLE 2 EQUIVALENT -- Fixed Effects (DMN only)")
    print("="*70)

    if not within_df.empty and "rsn_idx" in within_df.columns:
        dmn_w = within_df[within_df.rsn_idx == DMN_IDX]
        if not dmn_w.empty:
            print("\n--- Within-modality (BOLD & ASL, REST2-REST1) ---")
            cols = ["modality", "model", "n_subjects", "β_timescale", "se_timescale",
                    "p_timescale", "β_mse_pre", "p_mse_pre", "converged"]
            print(dmn_w[[c for c in cols if c in dmn_w.columns]].to_string(index=False))
        else:
            print("\n  Within-modality: no DMN results available")
    else:
        print("\n  Within-modality: no results (need >=3 subjects with REST1+REST2)")

    if not cross_df.empty and "rsn_idx" in cross_df.columns:
        dmn_c = cross_df[cross_df.rsn_idx == DMN_IDX]
        if not dmn_c.empty:
            print("\n--- Cross-modality (BOLD - ASL at REST1) ---")
            cols = ["model", "n_subjects", "β_timescale", "se_timescale",
                    "p_timescale", "β_mse_pre", "p_mse_pre", "converged"]
            print(dmn_c[[c for c in cols if c in dmn_c.columns]].to_string(index=False))
        else:
            print("\n  Cross-modality: no DMN results available")
    else:
        print("\n  Cross-modality: no results (need >=3 paired subjects)")


# ── Entry point ───────────────────────────────────────────────────────────────
def run_mlm(csv_path: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw = pd.read_csv(csv_path)
    df  = _average_directions(raw)

    print("\n--- MLM: Within-modality (REST1 vs REST2) ---")
    within_df = mlm_within_modality(df)
    if not within_df.empty:
        wp = os.path.join(RESULTS_DIR, "bold_asl_mlm_within.csv")
        within_df.to_csv(wp, index=False)
        print(f"  Saved: {wp}")

        # Marginal plots per modality
        for mod in ["BOLD", "ASL"]:
            sub_avg  = df[(df.modality == mod)]
            subs_all = sub_avg.groupby("subject")["session"].nunique()
            subs_both = subs_all[subs_all == 2].index
            if len(subs_both) < 2:
                continue
            sub_avg = sub_avg[sub_avg.subject.isin(subs_both)]
            r1_long = _to_long(sub_avg[sub_avg.session == "REST1"
                                       ][sub_avg.rsn_idx == DMN_IDX]
                               ).rename(columns={"mse": "mse_r1"})
            r2_long = _to_long(sub_avg[sub_avg.session == "REST2"
                                       ][sub_avg.rsn_idx == DMN_IDX]
                               ).rename(columns={"mse": "mse_r2"})
            diff_long = r1_long.merge(
                r2_long[["subject", "timescale", "mse_r2"]],
                on=["subject", "timescale"]
            ).assign(mse_diff=lambda d: d["mse_r2"] - d["mse_r1"])
            plot_marginal_diff(diff_long,
                               f"{mod}: REST2−REST1 DMN complexity (Figure 3 equiv.)",
                               f"bold_asl_mlm_marginal_{mod.lower()}.png")

    print("\n--- MLM: Cross-modality (BOLD vs ASL at REST1) ---")
    cross_df = mlm_cross_modality(df)
    if not cross_df.empty:
        cp = os.path.join(RESULTS_DIR, "bold_asl_mlm_cross.csv")
        cross_df.to_csv(cp, index=False)
        print(f"  Saved: {cp}")

        # Marginal plot for cross-modality
        rest1    = df[df.session == "REST1"]
        bold_dmn = _to_long(rest1[(rest1.modality == "BOLD") & (rest1.rsn_idx == DMN_IDX)]
                            ).rename(columns={"mse": "mse_bold"})
        asl_dmn  = _to_long(rest1[(rest1.modality == "ASL") & (rest1.rsn_idx == DMN_IDX)]
                            ).rename(columns={"mse": "mse_asl"})
        diff_cross = bold_dmn.merge(
            asl_dmn[["subject", "timescale", "mse_asl"]],
            on=["subject", "timescale"]
        ).assign(mse_diff=lambda d: d["mse_bold"] - d["mse_asl"])
        plot_marginal_diff(diff_cross,
                           "BOLD − ASL: DMN complexity at REST1 (Figure 3 equiv.)",
                           "bold_asl_mlm_marginal_cross.png")

    print_table2(within_df if not within_df.empty else pd.DataFrame(),
                 cross_df  if not cross_df.empty  else pd.DataFrame())
    print("\nMLM done.")
