# fMRI Network Complexity: BOLD vs ASL Multiscale Entropy Analysis

Resting-state fMRI network complexity analysis following **McDonough et al. 2019** (_Entropy_ 21, 1072).  
Applies Multiscale Entropy (MSE) to BOLD and ASL data from the same 19 subjects and compares network complexity across modalities and sessions.

---

## Reference

> McDonough, I.M., Letang, S.K., Erwin, H.B., & Kana, R.K. (2019).  
> Evidence for Maintained Post-Encoding Memory Consolidation Across the Adult Lifespan Revealed by Network Complexity.  
> _Entropy_, 21(11), 1072. https://doi.org/10.3390/e21111072

---

## Data

| Path                | Contents                                                                               |
| ------------------- | -------------------------------------------------------------------------------------- |
| `data/BOLD/`        | BOLD fMRI NIfTI — `{sub}_rfMRI_{REST1\|REST2}_{AP\|PA}.nii.gz`, shape (91,109,91,333)  |
| `data/ASL/`         | ASL fMRI NIfTI — `swrdr63real_{sub}_{REST1\|REST2}_{LR\|RL}.nii`, shape (91,109,91,63) |
| `data/SUB*_hrf.mat` | HRF data (not used in this pipeline)                                                   |

- **19 subjects** (IDs: 1222, 5185, 5700, 5743, 5762, 5765, 5877, 5898, 6077, 6103, 6116, 6142, 6183, 6395, 6607, 6634, 6645, 6733, 6738)
- **9 subjects** have both REST1 and REST2; all 19 have REST1
- Both modalities share the same subject IDs

---

## File Structure

```
MSE_apply/
├── README.md
│
├── bold_asl_01_loader.py       # Scan inventory builder
├── bold_asl_02_rsn_mse.py      # Dual regression (Sec 2.5) + MSE (Sec 2.6)
├── bold_asl_03_analysis.py     # Paired t-tests + visualization
├── bold_asl_04_mlm.py          # Multilevel modeling (Sec 2.7)
├── bold_asl_run.py             # Full pipeline entry point
│
└── results/
    ├── bold_asl_mse_raw.csv            # MSE values per subject/session/modality/RSN/scale
    ├── bold_asl_mse_curves.png         # Figure 2 equivalent — MSE curves
    ├── bold_asl_session_diff.png       # REST1 vs REST2 MSE difference
    ├── bold_asl_session_stats.csv      # Paired t-test results (within-modality)
    ├── bold_asl_modality_diff.png      # BOLD vs ASL MSE difference
    ├── bold_asl_modality_stats.csv     # Paired t-test results (cross-modality)
    ├── bold_asl_mlm_within.csv         # MLM fixed effects — within-modality models
    ├── bold_asl_mlm_cross.csv          # MLM fixed effects — cross-modality models
    ├── bold_asl_mlm_marginal_bold.png  # Figure 3 equivalent — BOLD REST2-REST1
    ├── bold_asl_mlm_marginal_asl.png   # Figure 3 equivalent — ASL REST2-REST1
    └── bold_asl_mlm_marginal_cross.png # Figure 3 equivalent — BOLD-ASL difference
```

---

## Methodology

### Section 2.5 — Dual Regression

RSN time series extracted via GLM (Step 1 of FSL `dual_regression`):

$$T = (S^\top S)^{-1} S^\top Y^\top$$

- Atlas: Smith 2009 10-RSN ICA maps (`nilearn.datasets.fetch_atlas_smith_2009().maps`)
- Atlas resampled to data space via `nilearn.image.resample_to_img`

### Section 2.6 — Multiscale Entropy (MSE)

- **m = 2**, **r = 0.5 × SD(original signal)**
- **Scales 1–6** (common range for both modalities)
  - BOLD (333 tp): valid up to scale 13 by N/25 heuristic; capped at 6 for fair comparison
  - ASL (63 tp): valid up to scale 6 by N/10 heuristic (floor(63/6) = 10 = threshold)
- Coarse-graining: non-overlapping window average
- Scale validity guard: coarse-grained length ≥ max(10, m+2)

### Section 2.7 — Multilevel Modeling (MLM)

```
MSE_diff ~ Timescale + MSE_pre + (1 + Timescale | subject)
```

- Random intercept + random slope for Timescale
- Maximum likelihood estimation (`reml=False`)
- Predictors centered (z-scored) before fitting
- AR(1) approximated by random slope (statsmodels `MixedLM` limitation)
- Fallback to random-intercept-only if random slope fails to converge

**Models (within-modality, subjects with REST1+REST2):**

| Model   | DV                | Fixed predictors      |
| ------- | ----------------- | --------------------- |
| Model 1 | MSE_REST2 − REST1 | Timescale + MSE_REST1 |
| Model 2 | MSE_REST1         | Timescale             |
| Model 3 | MSE_REST2         | Timescale + MSE_REST1 |

**Models (cross-modality, BOLD vs ASL at REST1):**

| Model   | DV             | Fixed predictors    |
| ------- | -------------- | ------------------- |
| Model A | MSE_BOLD − ASL | Timescale + MSE_ASL |
| Model B | MSE_ASL        | Timescale           |
| Model C | MSE_BOLD       | Timescale + MSE_ASL |

> **Note:** The paper's primary predictors (Age, Memory Accuracy, Sex, Premorbid IQ) are not available in our dataset. The MLM structure is directly adapted; modality (BOLD vs ASL) and session (REST1 vs REST2) serve as the primary comparison variables.

---

## Installation

```bash
pip install numpy scipy pandas matplotlib nilearn nibabel statsmodels
```

---

## Results

> ⚠️ Figures below are from a **fast-mode run (2 subjects)**. Full results will be updated after running all 19 subjects.

### MSE Curves — Default Mode Network (Figure 2 equivalent)

Multiscale entropy across timescales 1–6 for BOLD and ASL at REST1 and REST2.  
ASL shows consistently higher complexity at fine scales; both modalities trend upward with timescale.

<img width="1050" height="600" alt="image" src="https://github.com/user-attachments/assets/e30d2230-780f-45c0-ba39-7f0f7f9a1c98" />

---

### Within-modality: REST1 → REST2 Change (BOLD & ASL)

REST2 − REST1 MSE difference per timescale in the DMN.  
Blue bars indicate p < 0.05 (paired t-test). Pattern reflects test-retest reliability of session differences.

<img width="1784" height="617" alt="image" src="https://github.com/user-attachments/assets/a8e96ed6-5cc3-464b-b7b7-a6c1922c03ea" />

---

### Cross-modality: BOLD − ASL at REST1

BOLD has lower DMN complexity than ASL at fine timescales (1–4), converging at coarser scales.  
Blue bar (scale 3) reached p < 0.05 in the 2-subject pilot.

<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/eb50124c-b341-41db-80ae-42cb36bee226" />

---

## Usage

```bash
# Full run — all 19 subjects (~1–2 hours)
python bold_asl_run.py

# Fast test — first 2 subjects only
python bold_asl_run.py --fast

# Skip computation, re-run analysis on existing cache
python bold_asl_run.py --analysis_only
```

### Pipeline steps

| Step | Script                    | Description                      |
| ---- | ------------------------- | -------------------------------- |
| 1    | `bold_asl_01_loader.py`   | Build scan inventory DataFrame   |
| 2–3  | `bold_asl_02_rsn_mse.py`  | Dual regression + MSE per scan   |
| 4    | `bold_asl_03_analysis.py` | Paired t-tests + MSE curve plots |
| 5    | `bold_asl_04_mlm.py`      | MLM (Table 2 equivalent)         |

---

## Implementation Coverage vs Paper

| Section             | Description                                                | Status                            |
| ------------------- | ---------------------------------------------------------- | --------------------------------- |
| 2.5 Dual Regression | $(S^\top S)^{-1}S^\top Y^\top$ via `np.linalg.lstsq`       | Complete                          |
| 2.6 MSE             | m=2, r=0.5, scales 1–6, N/10 guard                         | Complete                          |
| 2.7 MLM structure   | Random intercept + slope, ML estimation, MSE_pre covariate | Complete                          |
| 2.7 MLM predictors  | Age, Memory Accuracy, Sex, Premorbid IQ                    | Not applicable (data unavailable) |
| 2.7 AR(1)           | Approximated via random slope                              | Partial                           |

---

## Results (fast mode: 2 subjects, GroupKFold k=2)

> `python run_all.py --fast --cv group_kfold --k 2 --max_scale 3`

### Feature Ablation (Accuracy mean ± std)

| Condition         | LR                | SVM           | RF            | XGB           |
| ----------------- | ----------------- | ------------- | ------------- | ------------- |
| entropy_only      | 0.188 ± 0.000     | 0.156 ± 0.031 | 0.250 ± 0.063 | 0.250 ± 0.063 |
| stats_only        | 0.156 ± 0.094     | 0.188 ± 0.063 | 0.188 ± 0.063 | 0.188 ± 0.000 |
| **spectral_only** | **0.438 ± 0.063** | 0.344 ± 0.094 | 0.406 ± 0.031 | 0.313 ± 0.063 |
| ent+stats         | 0.188 ± 0.063     | 0.250 ± 0.000 | 0.156 ± 0.031 | 0.219 ± 0.031 |
| **ent+spec**      | **0.406 ± 0.031** | 0.219 ± 0.031 | 0.406 ± 0.094 | 0.313 ± 0.000 |
| all_features      | 0.313 ± 0.063     | 0.188 ± 0.063 | 0.219 ± 0.031 | 0.219 ± 0.094 |

> 4-class problem (chance level = 0.25). Spectral features dominate; the ent+spec combination consistently outperforms entropy alone.

### Feature Ablation Plot

![Feature Ablation](results/ablation_plot.png)

### Feature Importance (Random Forest, all data)

![Feature Importance](results/feature_importance.png)

> Beta/alpha band power and spectral entropy rank highest — RCMSE (scale 1–3) also appears within the top 10.

---

## Feature Extraction Structure

One time series (68 points) → **20-dimensional** feature vector

| Group    | Features                                                               | Dim |
| -------- | ---------------------------------------------------------------------- | --- |
| RCMSE    | scale 1, 2, 3                                                          | 3   |
| Stats    | mean, std, skew, kurt, p25, p75, iqr, min, max, range                  | 10  |
| Spectral | delta/theta/alpha/beta power, total power, spectral entropy, peak freq | 7   |

> `channel_avg=True` (default): average 129 channels → 1D → 20D  
> `channel_avg=False`: extract per channel → 20 × 129 = 2580D

---

## Classification Models

- Logistic Regression (L2, C=1)
- SVM (RBF kernel)
- Random Forest (100 estimators)
- XGBoost (100 rounds, depth 3)

Validation: subject-wise CV (`groups` = subject ID) — prevents data leakage across subjects
