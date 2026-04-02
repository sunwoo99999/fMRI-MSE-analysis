"""
bold_asl_run.py
Full pipeline entry point — BOLD vs ASL network complexity analysis.

Applies McDonough et al. 2019 (Entropy) methodology to multi-modal
resting-state fMRI data (BOLD + ASL, same 19 subjects).

Steps
-----
1. Build scan inventory
2. For each scan: extract 10 RSN time series (Smith 2009 ICA maps via nilearn)
3. Compute MSE (m=2, r=0.5, scales 1-6) for every RSN
4. Cache results to results/bold_asl_mse_raw.csv
5. Run analysis & generate plots

Usage
-----
    # Full run (all 19 subjects, ~1-2 hours)
    python bold_asl_run.py

    # Fast test (first 2 subjects only)
    python bold_asl_run.py --fast

    # Skip to analysis (if raw CSV already exists)
    python bold_asl_run.py --analysis_only
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── Auto-detect R (nlme AR(1) support) ───────────────────────────────────────
# winget installs R to %LOCALAPPDATA%\Programs\R\<version>\bin by default.
# If Rscript is not already on PATH, probe the known location and prepend it.
import shutil as _shutil
if _shutil.which("Rscript") is None:
    _r_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "R")
    if os.path.isdir(_r_base):
        for _ver in sorted(os.listdir(_r_base), reverse=True):
            _rbin = os.path.join(_r_base, _ver, "bin")
            if os.path.exists(os.path.join(_rbin, "Rscript.exe")):
                os.environ["PATH"] = _rbin + os.pathsep + os.environ.get("PATH", "")
                break
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_PATH  = os.path.join(RESULTS_DIR, "bold_asl_mse_raw.csv")

# MSE / rcMSE parameters (McDonough et al. 2019, Section 2.6)
# method='rcmse' (Wu et al. 2014) is the default — recommended for short
# ASL series (~63 TR) where standard MSE yields NaN at scales 4-6.
M          = 2
R_FACTOR   = 0.5
MAX_SCALE  = 6   # common upper bound for BOLD (333tp) and ASL (63tp)
METHOD     = "rcmse"  # 'rcmse' | 'mse'

SCALE_COLS = [f"scale_{s}" for s in range(1, MAX_SCALE + 1)]


def _import_modules():
    """Lazy import to keep startup fast."""
    import importlib.util, sys

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    loader   = _load("bold_asl_01_loader",  os.path.join(BASE_DIR, "bold_asl_01_loader.py"))
    rsn_mse  = _load("bold_asl_02_rsn_mse", os.path.join(BASE_DIR, "bold_asl_02_rsn_mse.py"))
    analysis = _load("bold_asl_03_analysis",os.path.join(BASE_DIR, "bold_asl_03_analysis.py"))
    return loader, rsn_mse, analysis


def compute_all(inventory: pd.DataFrame,
                rsn_mse_mod,
                max_scale: int = MAX_SCALE,
                method: str = METHOD) -> pd.DataFrame:
    """
    For every scan in inventory: extract RSN time series → compute MSE.
    Returns a DataFrame with one row per (scan × RSN).
    """
    from bold_asl_02_rsn_mse import RSN_NAMES
    rows    = []
    n_scans = len(inventory)

    for i, row in inventory.iterrows():
        tag = f"[{i+1}/{n_scans}] {row.modality} {row.subject} {row.session} {row.direction}"
        print(f"  {tag} ... ", end="", flush=True)

        try:
            ts  = rsn_mse_mod.extract_rsn_timeseries(row.filepath)
            mse = rsn_mse_mod.compute_mse_all_rsns(ts,
                                                    max_scale=max_scale,
                                                    m=M,
                                                    r_factor=R_FACTOR,
                                                    method=method)
            for rsn_i in range(mse.shape[0]):
                rec = dict(
                    subject=row.subject,
                    session=row.session,
                    modality=row.modality,
                    direction=row.direction,
                    rsn_idx=rsn_i,
                    rsn_name=RSN_NAMES[rsn_i],
                )
                for si, val in enumerate(mse[rsn_i]):
                    rec[f"scale_{si+1}"] = val
                rows.append(rec)
            print("done")

        except Exception as e:
            print(f"ERROR: {e}")

    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(
        description="BOLD/ASL network complexity pipeline (McDonough 2019 methodology)")
    p.add_argument("--fast",          action="store_true",
                   help="Use first 2 subjects only (quick test)")
    p.add_argument("--analysis_only", action="store_true",
                   help="Skip computation, run analysis on existing cache")
    p.add_argument("--max_scale",     type=int, default=MAX_SCALE,
                   help=f"Maximum MSE scale (default {MAX_SCALE})")
    p.add_argument("--method",         type=str, default=METHOD,
                   choices=["rcmse", "mse"],
                   help="Entropy algorithm: rcmse (default, short series) or mse (Costa 2002)")
    return p.parse_args()


def main():
    args    = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    loader, rsn_mse, analysis = _import_modules()

    import importlib.util
    def _load_mlm():
        spec = importlib.util.spec_from_file_location(
            "bold_asl_04_mlm", os.path.join(BASE_DIR, "bold_asl_04_mlm.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    mlm = _load_mlm()


    # ── Step 1: inventory ────────────────────────────────────────────────────
    print("=== Step 1: Building scan inventory ===")
    inv = loader.build_inventory()
    if args.fast:
        subs = sorted(inv["subject"].unique())[:2]
        inv  = inv[inv["subject"].isin(subs)].reset_index(drop=True)
        print(f"  Fast mode: {subs}")
    print(f"  {len(inv)} scan files -- "
          f"{inv['subject'].nunique()} subjects, "
          f"{inv['modality'].value_counts().to_dict()}")

    # ── Step 2-3: compute MSE (with caching) ─────────────────────────────────
    if args.analysis_only and os.path.exists(CACHE_PATH):
        print(f"\n=== Skipping computation -- loading cache: {CACHE_PATH} ===")
    else:
        print("\n=== Steps 2-3: RSN extraction + MSE computation ===")
        print(f"  Parameters: m={M}, r={R_FACTOR}*SD, scales 1-{args.max_scale}, method={args.method}")
        raw_df = compute_all(inv, rsn_mse, max_scale=args.max_scale,
                             method=args.method)
        raw_df.to_csv(CACHE_PATH, index=False)
        print(f"\n  Cached: {CACHE_PATH}  ({len(raw_df)} rows)")

    # ── Step 4: analysis ─────────────────────────────────────────────────────
    print("\n=== Step 4: Analysis + visualization ===")
    analysis.run_analysis(CACHE_PATH)

    # ── Step 5: Section 2.7 MLM ──────────────────────────────────────────────
    print("\n=== Step 5: MLM (Section 2.7) ===")
    mlm.run_mlm(CACHE_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
