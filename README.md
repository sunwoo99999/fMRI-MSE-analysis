# fMRI Network Complexity: BOLD vs ASL Multiscale Entropy Analysis

Resting-state fMRI network complexity analysis following **McDonough et al. 2019** (_Entropy_ 21, 1072).  
Applies rcMSE (Refined Composite MSE) to BOLD and ASL data from the same 19 subjects and compares network complexity across modalities and sessions.

> **Algorithm note:** ASL scans contain only ~63 TR. Standard MSE collapses to NaN at scales 4–6 because coarse-grained series shrink to ≤15 points and template-match counts reach zero. The pipeline therefore defaults to **rcMSE** (Wu et al. 2014, _Front. Neuroinform._), which pools template counts across all τ offset coarse-grained sequences per scale, recovering ~τ-fold more matches with no additional assumptions. Standard MSE remains available via `--method mse`.

> **AR(1) note:** Section 2.7 MLM now uses **R `nlme::lme` with `corAR1()`** when Rscript is available — an exact match to the paper's autocorrelation structure. Python `statsmodels` (no AR(1)) is used as a fallback only when R is not installed.

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
├── bold_asl_02_rsn_mse.py      # Dual regression (Sec 2.5) + MSE / rcMSE (Sec 2.6)
├── bold_asl_03_analysis.py     # Paired t-tests + visualization
├── bold_asl_04_mlm.py          # Multilevel modeling (Sec 2.7) — R nlme AR(1) + statsmodels fallback
├── bold_asl_mlm_ar1.R          # R script: nlme::lme + corAR1() — exact paper match
├── bold_asl_run.py             # Full pipeline entry point (auto-detects R)
│
└── results/
    ├── bold_asl_mse_raw.csv            # rcMSE values per subject/session/modality/RSN/scale
    ├── bold_asl_mse_curves.png         # Figure 2 equivalent — MSE curves
    ├── bold_asl_session_diff.png       # REST1 vs REST2 MSE difference
    ├── bold_asl_session_stats.csv      # Paired t-test results (within-modality)
    ├── bold_asl_modality_diff.png      # BOLD vs ASL MSE difference
    ├── bold_asl_modality_stats.csv     # Paired t-test results (cross-modality)
    ├── bold_asl_mlm_within.csv         # MLM fixed effects — within-modality (ar1_used column)
    ├── bold_asl_mlm_cross.csv          # MLM fixed effects — cross-modality  (ar1_used column)
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

### Section 2.6 — Multiscale Entropy (MSE / rcMSE)

- **m = 2**, **r = 0.5 × SD(original signal)**
- **Scales 1–6** (common range for both modalities)
  - BOLD (333 tp): valid up to scale 13 by N/25 heuristic; capped at 6 for fair comparison
  - ASL (63 tp): floor(63/τ) drops to 10 at scale 6 — borderline for standard MSE
- **Default algorithm: rcMSE** (Refined Composite MSE, Wu et al. 2014, _Front. Neuroinform._)
  - At scale τ, generates τ offset coarse-grained sequences $y^{(k)}_j$ for $k = 0, \ldots, \tau-1$
  - Pools template match counts across all offsets before computing entropy:

$$\text{rcMSE}(\tau) = -\log\!\ \frac{\sum_{k=0}^{\tau-1} A_k}{\sum_{k=0}^{\tau-1} B_k}$$

- Recovers ~τ-fold more template pairs from the same short signal
- At scale 1 rcMSE is identical to standard sample entropy
- Monte Carlo validation (N=500 iid signals, length=63): standard MSE yields mean 0.04 NaN/signal (max 2); rcMSE yields **0 NaN** across all trials
- Standard MSE (Costa et al. 2002) retained for reference; selectable via `--method mse`

### Section 2.7 — Multilevel Modeling (MLM)

```
MSE_diff ~ Timescale + MSE_pre + (1 + Timescale | subject),  corAR1(~ timescale | subject)
```

**AR(1) implementation (논문 exact match):**

- 우선 경로: `bold_asl_mlm_ar1.R` → R `nlme::lme(correlation = corAR1(form = ~ timescale | subject), method = "ML")`
- 대안 경로 (R 미설치 시): `statsmodels MixedLM` — AR(1) 없음, random slope으로 부분 보완
- 결과 CSV에 `ar1_used` 컬럼으로 실제 사용된 모형 기록

**그 외 설정:**

- Random intercept + random slope for Timescale
- Maximum likelihood estimation (`method = "ML"`, `reml = FALSE`)
- 예측변수 모두 z-score 표준화 후 투입
- n < 3 또는 수렴 실패 시 random intercept only로 자동 fallback

**⚠️ 실험 설계 한계 (코드로 해결 불가):**

| 논문                                              | 본 연구                                |
| ------------------------------------------------- | -------------------------------------- |
| REST_pre → 기억 인코딩 과제 → REST_post (같은 날) | REST1, REST2 = 별도 세션 (다른 날)     |
| DV = 인코딩 직후 뇌의 기억 공고화(consolidation)  | DV = 세션 간 복잡도 변화 (test-retest) |
| Age, Memory Accuracy, Sex, IQ 공변량              | 해당 변수 데이터 없음                  |

→ 본 연구의 MLM은 논문의 통계 구조를 정확히 따르되, **연구 질문은 BOLD-ASL 모달리티 등가성 검증**으로 재정의됨.

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

**R (AR(1) MLM, 논문 exact match):**

```bash
# Windows: winget으로 자동 설치
winget install --id RProject.R
# R은 nlme 패키지를 기본 내장 — 추가 설치 불필요
# bold_asl_run.py가 %LOCALAPPDATA%\Programs\R\에서 Rscript를 자동 탐지함
```

> R 미설치 시 statsmodels fallback으로 자동 전환 (경고 없이 진행됨).

---

## Results

> Figures below are from the full run (**19 subjects**, April 1 2026).

### MSE Curves — Default Mode Network (Figure 2 equivalent)

Multiscale entropy across timescales 1–6 for BOLD and ASL at REST1 and REST2.  
ASL shows consistently higher complexity at fine scales; both modalities trend upward with timescale.

![MSE curves](results/bold_asl_mse_curves.png)

---

### Within-modality: REST1 → REST2 Change (BOLD & ASL)

REST2 − REST1 MSE difference per timescale in the DMN.  
Blue bars indicate p < 0.05 (paired t-test). Pattern reflects test-retest reliability of session differences.

![Session difference](results/bold_asl_session_diff.png)

---

### Cross-modality: BOLD − ASL at REST1

BOLD has lower DMN complexity than ASL at fine timescales (1–4), converging at coarser scales.  
Blue bar (scale 3) reached p < 0.05 in the 2-subject pilot.

![Modality difference](results/bold_asl_modality_diff.png)

---

## Usage

```bash
# Full run — all 19 subjects (~1–2 hours)
python bold_asl_run.py

# Fast test — first 2 subjects only
python bold_asl_run.py --fast

# Skip computation, re-run analysis on existing cache
python bold_asl_run.py --analysis_only

# Use standard MSE instead of rcMSE (not recommended for ASL)
python bold_asl_run.py --method mse

# Limit to 4 scales (reduces NaN risk further with standard MSE)
python bold_asl_run.py --method mse --max_scale 4
```

### Pipeline steps

| Step | Script                    | Description                                                          |
| ---- | ------------------------- | -------------------------------------------------------------------- |
| 1    | `bold_asl_01_loader.py`   | Build scan inventory DataFrame                                       |
| 2–3  | `bold_asl_02_rsn_mse.py`  | Dual regression + rcMSE per scan (default) or MSE via `--method mse` |
| 4    | `bold_asl_03_analysis.py` | Paired t-tests + MSE curve plots                                     |
| 5    | `bold_asl_04_mlm.py`      | MLM (Table 2 equivalent)                                             |

---

## Implementation Coverage vs Paper

| Section                 | 논문 명세                                                         | 구현 상태                                | 비고                                                |
| ----------------------- | ----------------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------- |
| **2.4 전처리**          | SPM12 unwarping, MELODIC ICA denoising, ANTs normalization        | ✅ **완료** (데이터 수령 시 이미 적용됨) | `swrdr63real_` prefix = 전처리 완료 파일            |
| **2.5 Dual Regression** | FSL `dual_regression` Step 1: $T = (S^\top S)^{-1} S^\top Y^\top$ | ✅ **완료**                              | `np.linalg.lstsq` 로 동일하게 구현                  |
| **2.5 RSN Atlas**       | Smith 2009 rsn10 공간지도, 10개 RSN 동시 GLM                      | ✅ **완료**                              | `nilearn.datasets.fetch_atlas_smith_2009()`         |
| **2.6 MSE 파라미터**    | m=2, r=0.5×SD(원본), scales 1–7 (N/25 heuristic)                  | ✅ **완료** (스케일 6으로 적응)          | 63TR에서 N/25=2.5 → N/10=6으로 조정, rcMSE로 안정화 |
| **2.6 rcMSE 확장**      | (논문은 표준 MSE 사용) τ-offset pooling으로 NaN 제거              | ✅ **완료** (Wu et al. 2014 추가 적용)   | ASL 63TR에서 NaN 0건 달성                           |
| **2.7 MLM 구조**        | Random intercept + slope, ML estimation, MSE_pre 공변량           | ✅ **완료**                              | R `nlme` + Python fallback                          |
| **2.7 AR(1) 자기상관**  | `corAR1(form = ~ timescale \| subject)` via R `nlme`              | ✅ **완료**                              | `bold_asl_mlm_ar1.R` → R 4.5.3 + nlme               |
| **2.7 주요 예측변수**   | Age, Memory Accuracy, Sex, Premorbid IQ                           | ❌ **해당 없음**                         | 본 데이터에 없는 변수 — 연구 질문 자체가 다름       |
| **실험 설계**           | REST_pre → 기억 인코딩 과제 → REST_post (같은 날)                 | ❌ **해당 없음**                         | REST1/REST2 = 별도 세션, 인코딩 과제 없음           |

**구현률:** 기술적 방법론(알고리즘·통계 모형) **7/7 완료** · 실험 설계 차이는 연구 질문 차이로 재정의

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
