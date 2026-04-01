"""
bold_asl_02_rsn_mse.py
RSN time-series extraction + MSE / rcMSE computation.

Follows McDonough et al. 2019 (Entropy) Sections 2.5–2.6:

  Section 2.5  — Dual regression using Smith et al. 2009 10-RSN spatial maps.
                 Here we use nilearn NiftiMapsMasker as the Python equivalent
                 (spatial ICA regression without requiring FSL).

  Section 2.6  — MSE with m=2, r=0.5 * SD(original signal), scales 1..max_scale.
                 Upper bound: N/25 (paper used 175/25=7 scales).
                 For fair BOLD–ASL comparison we cap at scale 6 (63/10 ≈ 6).

  rcMSE extension (Wu et al. 2014, Front. Neuroinform.):
                 For ASL data (~63 TR), standard MSE yields NaN at scales 4-6
                 because each coarse-grained series has only 10-15 points and
                 pattern counts collapse to zero.  rcMSE creates τ offset
                 coarse-grained series per scale and pools their template match
                 counts before taking the log — recovering ~τ-fold more matches
                 with no additional assumptions.  This is the default method.
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── RSN metadata (Smith et al. 2009 rsn10 ordering) ─────────────────────────
RSN_NAMES = [
    "medial_visual",       # 0
    "occipital_pole",      # 1
    "lateral_visual",      # 2
    "default_mode",        # 3  ← DMN (primary network in paper)
    "cerebellum",          # 4
    "sensorimotor",        # 5
    "auditory",            # 6
    "executive_ctrl",      # 7
    "right_frontoparietal",# 8
    "left_frontoparietal", # 9
]
N_RSN  = len(RSN_NAMES)
DMN_IDX = 3


# ── RSN extraction ───────────────────────────────────────────────────────────
_rsn_atlas = None


def _get_rsn_atlas():
    global _rsn_atlas
    if _rsn_atlas is None:
        from nilearn import datasets
        _rsn_atlas = datasets.fetch_atlas_smith_2009().maps
    return _rsn_atlas


def extract_rsn_timeseries(nii_path: str, standardize: bool = True) -> np.ndarray:
    """
    Extract (T, 10) RSN time series via dual regression Step 1.

    Implements FSL dual_regression Step 1 exactly:
      T = (S^T S)^{-1} S^T Y^T
    where S = (V, K) group RSN spatial maps,
          Y = (T, V) 4D BOLD/ASL data (brain voxels only).

    All 10 RSN maps enter the GLM simultaneously, removing shared spatial
    variance — matching the paper's Section 2.5 procedure.

    Parameters
    ----------
    nii_path    : path to a preprocessed 4D NIfTI in MNI 2 mm space
    standardize : z-score each RSN time series after extraction

    Returns
    -------
    ts : np.ndarray  shape (T, 10)
    """
    import nibabel as nib
    from nilearn.image import resample_to_img
    from nilearn.masking import compute_brain_mask, apply_mask

    # ── Load atlas and resample to data space if needed ──────────────────────
    atlas_img = nib.load(_get_rsn_atlas())
    data_img  = nib.load(nii_path)
    # Use first volume as reference for mask computation
    import nilearn.image as nli
    ref_img = nli.index_img(data_img, 0)

    # Resample atlas to data space
    atlas_res = resample_to_img(atlas_img, ref_img, interpolation="continuous")

    # ── Build brain mask from 4D data ─────────────────────────────────────────
    mask_img = compute_brain_mask(data_img)

    # ── Extract masked voxel matrices ─────────────────────────────────────────
    # Y : (T, V)
    Y = apply_mask(data_img, mask_img).astype(np.float64)
    # S : (V, K=10) — spatial maps masked to same voxels
    S = apply_mask(atlas_res, mask_img).astype(np.float64).T  # (V, K)

    # ── Dual regression Step 1: T = (S^T S)^{-1} S^T Y^T → shape (K, T) ─────
    # Solve via least squares for numerical stability
    ts, _, _, _ = np.linalg.lstsq(S, Y.T, rcond=None)   # (K, T)
    ts = ts.T                                              # (T, K)

    # ── Optional z-score ──────────────────────────────────────────────────────
    if standardize:
        mu  = ts.mean(axis=0, keepdims=True)
        sd  = ts.std(axis=0, keepdims=True, ddof=1)
        sd[sd == 0] = 1.0
        ts  = (ts - mu) / sd

    return ts


# ── MSE / rcMSE computation ──────────────────────────────────────────────────
def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    """Non-overlapping window average starting at offset 0 (Costa et al. 2002)."""
    n = len(x) - (len(x) % scale)
    return x[:n].reshape(-1, scale).mean(axis=1)


def _coarse_grain_k(x: np.ndarray, scale: int, k: int) -> np.ndarray:
    """
    k-th offset coarse-grained sequence (0-indexed k, 0 <= k < scale).

    y_j = mean( x[k + j*scale : k + (j+1)*scale] )

    Used by rcMSE to generate τ offset sequences per scale, recovering
    ~τ-fold more template pairs from the same short signal.
    """
    x_off = x[k:]
    n     = len(x_off) - (len(x_off) % scale)
    if n == 0:
        return np.array([], dtype=np.float64)
    return x_off[:n].reshape(-1, scale).mean(axis=1)


def _sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    """
    Sample entropy with template length m and absolute tolerance r.
    Vectorized Chebyshev distance over upper-triangle template pairs.
    """
    N = len(x)
    if N < m + 2:
        return np.nan

    idx = np.arange(N - m)
    tm  = x[idx[:, None] + np.arange(m)]        # (N-m, m)
    tm1 = x[idx[:, None] + np.arange(m + 1)]    # (N-m, m+1)

    diff_m  = np.abs(tm[:, None, :]  - tm[None, :, :])    # (N-m, N-m, m)
    diff_m1 = np.abs(tm1[:, None, :] - tm1[None, :, :])   # (N-m, N-m, m+1)

    cheb_m  = diff_m.max(axis=2)
    cheb_m1 = diff_m1.max(axis=2)

    mask = np.triu(np.ones((N - m, N - m), dtype=bool), k=1)
    B = int(np.sum(cheb_m[mask]  <= r))
    A = int(np.sum(cheb_m1[mask] <= r))

    if B == 0:
        return np.nan
    return float(-np.log(A / B)) if A > 0 else np.nan


def _template_counts(cg: np.ndarray, m: int, r: float):
    """Return (A, B) for a single coarse-grained sequence."""
    N = len(cg)
    if N < m + 2:
        return 0, 0
    idx  = np.arange(N - m)
    tm   = cg[idx[:, None] + np.arange(m)]
    tm1  = cg[idx[:, None] + np.arange(m + 1)]
    cheb_m  = np.abs(tm[:, None, :] - tm[None, :, :]).max(axis=2)
    cheb_m1 = np.abs(tm1[:, None, :] - tm1[None, :, :]).max(axis=2)
    mask = np.triu(np.ones((N - m, N - m), dtype=bool), k=1)
    return int(np.sum(cheb_m1[mask] <= r)), int(np.sum(cheb_m[mask] <= r))


def compute_mse(x: np.ndarray, max_scale: int,
                m: int = 2, r_factor: float = 0.5) -> np.ndarray:
    """
    Standard MSE curve for 1-D time series x (Costa et al. 2002).

    r is fixed as r_factor * SD(original signal) — not the coarse-grained
    signal — following McDonough & Nashiro 2014 (ref 28).

    Returns
    -------
    mse : np.ndarray  shape (max_scale,), NaN where scale too coarse
    """
    r   = r_factor * float(np.std(x, ddof=1))
    mse = np.full(max_scale, np.nan)
    for s in range(1, max_scale + 1):
        cg = _coarse_grain(x, s)
        if len(cg) >= max(10, m + 2):
            mse[s - 1] = _sample_entropy(cg, m, r)
    return mse


def compute_rcmse(x: np.ndarray, max_scale: int,
                  m: int = 2, r_factor: float = 0.5) -> np.ndarray:
    """
    Refined Composite MSE (rcMSE) — Wu et al. 2014, Front. Neuroinform.

    At scale τ, creates τ offset coarse-grained sequences (k = 0 … τ-1) and
    pools their template match counts before computing entropy:

        rcMSE(τ) = -log( ΣA_k / ΣB_k )

    This recovers ~τ-fold more template pairs from the same short signal,
    dramatically reducing NaN rates for time series with N < 100 (e.g.
    ASL scans with ~63 TR where standard MSE collapses at scales 4-6).

    At scale 1 rcMSE is identical to standard sample entropy.

    r is fixed as r_factor * SD(original signal) — same convention as
    compute_mse() for cross-scale and cross-modality comparability.

    Returns
    -------
    rcmse : np.ndarray  shape (max_scale,), NaN only when ΣB_k == 0
            after pooling all τ offset sequences.
    """
    r     = r_factor * float(np.std(x, ddof=1))
    rcmse = np.full(max_scale, np.nan)
    for s in range(1, max_scale + 1):
        A_total = 0
        B_total = 0
        for k in range(s):
            cg = _coarse_grain_k(x, s, k)
            A_k, B_k = _template_counts(cg, m, r)
            A_total += A_k
            B_total += B_k
        if B_total > 0:
            rcmse[s - 1] = float(-np.log(A_total / B_total)) if A_total > 0 else np.nan
    return rcmse


def compute_mse_all_rsns(ts_matrix: np.ndarray,
                          max_scale: int = 6,
                          m: int = 2,
                          r_factor: float = 0.5,
                          method: str = "rcmse") -> np.ndarray:
    """
    Compute MSE or rcMSE for every RSN time series.

    Parameters
    ----------
    ts_matrix : (T, 10)
    method    : 'rcmse' (default, recommended for ASL / short series) or
                'mse'   (standard Costa 2002, for reference / long BOLD)

    Returns
    -------
    out : (10, max_scale)
    """
    if method not in ("rcmse", "mse"):
        raise ValueError(f"method must be 'rcmse' or 'mse', got {method!r}")
    _fn   = compute_rcmse if method == "rcmse" else compute_mse
    n_rsn = ts_matrix.shape[1]
    out   = np.full((n_rsn, max_scale), np.nan)
    for i in range(n_rsn):
        out[i] = _fn(ts_matrix[:, i], max_scale, m, r_factor)
    return out
