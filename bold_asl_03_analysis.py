"""
bold_asl_03_analysis.py
Statistical analysis and visualization.

Maps to McDonough et al. 2019 Sections 2.7 / 3:
  - MSE curves per modality × session  (→ Figure 2 equivalent)
  - Within-modality: REST2 − REST1 difference (primary DV in paper)
  - Cross-modality : BOLD vs ASL at REST1 (our adaptation)
  - Paired t-tests (adaptation of MLM; no Age/Memory covariates available)
  - Focus on DMN (index 3) as primary RSN
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from bold_asl_02_rsn_mse import RSN_NAMES, DMN_IDX

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCALE_COLS  = [f"scale_{s}" for s in range(1, 7)]


# ── helpers ──────────────────────────────────────────────────────────────────
def _load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _average_directions(df: pd.DataFrame) -> pd.DataFrame:
    """Average AP+PA (BOLD) / LR+RL (ASL) for each subject–session–modality."""
    return (df.groupby(["subject", "session", "modality", "rsn_idx", "rsn_name"],
                       as_index=False)[SCALE_COLS].mean())


def _dmn(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["rsn_idx"] == DMN_IDX].copy()


# ── Figure 2 equivalent ───────────────────────────────────────────────────────
def plot_mse_curves(df_avg: pd.DataFrame) -> None:
    """
    Mean MSE ± 95 % CI across subjects per (modality, session).
    One panel per modality.  Saves to results/bold_asl_mse_curves.png
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dmn = _dmn(df_avg)
    conditions = [
        ("BOLD", "REST1", "steelblue",  "-"),
        ("BOLD", "REST2", "steelblue",  "--"),
        ("ASL",  "REST1", "darkorange", "-"),
        ("ASL",  "REST2", "darkorange", "--"),
    ]
    scales = np.arange(1, 7)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mod, ses, color, ls in conditions:
        subset = dmn[(dmn.modality == mod) & (dmn.session == ses)]
        if subset.empty:
            continue
        vals = subset[SCALE_COLS].values      # (n_subjects, 6)
        mean = np.nanmean(vals, axis=0)
        se   = stats.sem(vals, axis=0, nan_policy="omit")
        ci   = 1.96 * se
        ax.plot(scales, mean, color=color, linestyle=ls,
                label=f"{mod} {ses}", linewidth=2)
        ax.fill_between(scales, mean - ci, mean + ci,
                        color=color, alpha=0.15)

    ax.set_xlabel("Timescale", fontsize=12)
    ax.set_ylabel("Network Complexity (SampEn)", fontsize=12)
    ax.set_title("MSE in Default Mode Network — BOLD vs ASL", fontsize=13)
    ax.set_xticks(scales)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "bold_asl_mse_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Within-modality REST1 vs REST2 (mirrors paper's main DV) ─────────────────
def compare_sessions(df_avg: pd.DataFrame) -> pd.DataFrame:
    """
    For each modality: paired t-test REST2 − REST1 per RSN per scale.
    Only subjects with both sessions are included.
    Returns a summary DataFrame.
    """
    rows = []
    for mod in ["BOLD", "ASL"]:
        sub  = df_avg[df_avg.modality == mod]
        # subjects with both sessions
        subs_both = (sub.groupby("subject")["session"]
                     .nunique().loc[lambda s: s == 2].index)
        if len(subs_both) < 2:
            continue
        sub = sub[sub.subject.isin(subs_both)]

        for rsn_i in range(10):
            rsn_df = sub[sub.rsn_idx == rsn_i]
            r1 = rsn_df[rsn_df.session == "REST1"][SCALE_COLS].values
            r2 = rsn_df[rsn_df.session == "REST2"][SCALE_COLS].values
            # same subject ordering guaranteed by sort in average_directions
            diff = r2 - r1
            for si, sc in enumerate(SCALE_COLS):
                d = diff[:, si][~np.isnan(diff[:, si])]
                if len(d) < 2:
                    continue
                t, p = stats.ttest_1samp(d, 0)
                rows.append(dict(
                    modality=mod, rsn_idx=rsn_i, rsn_name=RSN_NAMES[rsn_i],
                    scale=si + 1, mean_diff=d.mean(), t=t, p=p, n=len(d)
                ))

    return pd.DataFrame(rows)


# ── Cross-modality BOLD vs ASL at REST1 ──────────────────────────────────────
def compare_modalities(df_avg: pd.DataFrame) -> pd.DataFrame:
    """
    Paired t-test BOLD − ASL at REST1 (same subject, within-session).
    Returns a summary DataFrame.
    """
    rest1 = df_avg[df_avg.session == "REST1"]
    rows  = []
    for rsn_i in range(10):
        rsn_df = rest1[rest1.rsn_idx == rsn_i]
        bold   = rsn_df[rsn_df.modality == "BOLD"].sort_values("subject")
        asl    = rsn_df[rsn_df.modality == "ASL"].sort_values("subject")
        # intersect subjects
        subs   = sorted(set(bold.subject) & set(asl.subject))
        if len(subs) < 2:
            continue
        b_vals = bold[bold.subject.isin(subs)].sort_values("subject")[SCALE_COLS].values
        a_vals = asl[asl.subject.isin(subs)].sort_values("subject")[SCALE_COLS].values
        diff   = b_vals - a_vals
        for si, sc in enumerate(SCALE_COLS):
            d = diff[:, si][~np.isnan(diff[:, si])]
            if len(d) < 2:
                continue
            t, p = stats.ttest_1samp(d, 0)
            rows.append(dict(
                rsn_idx=rsn_i, rsn_name=RSN_NAMES[rsn_i],
                scale=si + 1, mean_diff=d.mean(), t=t, p=p, n=len(d)
            ))

    return pd.DataFrame(rows)


# ── Visualization: difference plots ──────────────────────────────────────────
def plot_session_diff(session_df: pd.DataFrame) -> None:
    """Bar chart of REST2−REST1 MSE difference per scale for DMN."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dmn = session_df[session_df.rsn_idx == DMN_IDX]
    if dmn.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mod in zip(axes, ["BOLD", "ASL"]):
        sub = dmn[dmn.modality == mod]
        if sub.empty:
            ax.set_title(f"{mod} (no data)")
            continue
        colors = ["steelblue" if p < 0.05 else "lightgray"
                  for p in sub["p"]]
        ax.bar(sub["scale"], sub["mean_diff"], color=colors, edgecolor="k")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Timescale")
        ax.set_ylabel("REST2 − REST1 MSE")
        ax.set_title(f"{mod} — DMN session difference\n(blue = p<0.05)")
        ax.set_xticks(sub["scale"])

    plt.suptitle("Within-modality REST1→REST2 change in Network Complexity",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "bold_asl_session_diff.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_modality_diff(modality_df: pd.DataFrame) -> None:
    """Bar chart of BOLD − ASL MSE difference per scale for DMN."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dmn = modality_df[modality_df.rsn_idx == DMN_IDX]
    if dmn.empty:
        return

    colors = ["steelblue" if p < 0.05 else "lightgray" for p in dmn["p"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dmn["scale"], dmn["mean_diff"], color=colors, edgecolor="k")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Timescale")
    ax.set_ylabel("BOLD − ASL MSE")
    ax.set_title("Cross-modality comparison at REST1 — DMN\n(blue = p<0.05)")
    ax.set_xticks(dmn["scale"])
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "bold_asl_modality_diff.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Console summary ───────────────────────────────────────────────────────────
def print_summary(session_df: pd.DataFrame, modality_df: pd.DataFrame) -> None:
    print("\n=== Within-modality REST1 vs REST2 (DMN) ===")
    dmn_s = session_df[session_df.rsn_idx == DMN_IDX][
        ["modality", "scale", "mean_diff", "t", "p", "n"]]
    print(dmn_s.to_string(index=False))

    print("\n=== Cross-modality BOLD vs ASL at REST1 (DMN) ===")
    dmn_m = modality_df[modality_df.rsn_idx == DMN_IDX][
        ["scale", "mean_diff", "t", "p", "n"]]
    print(dmn_m.to_string(index=False))


def run_analysis(csv_path: str) -> None:
    """Full analysis pipeline given the cached MSE CSV."""
    print("\n--- Loading MSE results ---")
    raw  = _load(csv_path)
    df   = _average_directions(raw)

    print("--- Plotting MSE curves (Figure 2 equivalent) ---")
    plot_mse_curves(df)

    print("--- Within-modality session comparison ---")
    ses_df = compare_sessions(df)
    plot_session_diff(ses_df)
    ses_path = os.path.join(RESULTS_DIR, "bold_asl_session_stats.csv")
    ses_df.to_csv(ses_path, index=False)
    print(f"  Saved: {ses_path}")

    print("--- Cross-modality comparison ---")
    mod_df = compare_modalities(df)
    plot_modality_diff(mod_df)
    mod_path = os.path.join(RESULTS_DIR, "bold_asl_modality_stats.csv")
    mod_df.to_csv(mod_path, index=False)
    print(f"  Saved: {mod_path}")

    print_summary(ses_df, mod_df)
