#!/usr/bin/env Rscript
# bold_asl_mlm_ar1.R
# MLM with AR(1) autocorrelation structure via nlme::lme
# Exact implementation of McDonough et al. 2019 Section 2.7:
#   MSE ~ Timescale + MSE_pre + (1 + Timescale | subject) + corAR1
#   method = ML (maximum likelihood, not REML)
#
# Usage (called from bold_asl_04_mlm.py via subprocess):
#   Rscript bold_asl_mlm_ar1.R <input.csv> <output.csv> <include_pre>
#
# Input CSV required columns:
#   dv        - dependent variable (e.g. MSE difference or single timepoint)
#   ts_z      - timescale z-scored  (used as fixed effect predictor)
#   timescale - timescale integer 1-6 (used ONLY for corAR1 position ordering)
#   subject   - subject ID (grouping factor for random effects)
#   pre_z     - MSE_pre z-scored covariate (required only when include_pre=TRUE)
#
# Output CSV columns:
#   Value, Std.Error, DF, t-value, p-value  -- from nlme summary tTable
#   term    -- fixed-effect name (mapped to Python naming in bold_asl_04_mlm.py)
#   rs      -- integer (1 = random slope converged, 0 = random intercept only)
#   ar1     -- integer (1 = AR(1) used, always 1 in this script)
#   status  -- "ok" | "ok_ri_only" | "error" | "insufficient_data"

suppressPackageStartupMessages(library(nlme))

# ── CLI args ──────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript bold_asl_mlm_ar1.R <in.csv> <out.csv> <include_pre>")
}
in_csv      <- args[1]
out_csv     <- args[2]
include_pre <- toupper(trimws(args[3])) == "TRUE"

# ── Load & validate data ──────────────────────────────────────────────────────
df <- read.csv(in_csv, stringsAsFactors = FALSE)

# Observations must be sorted by timescale within subject for corAR1 to work
df <- df[order(df$subject, df$timescale), ]

req_cols <- c("dv", "ts_z", "timescale", "subject")
if (include_pre) req_cols <- c(req_cols, "pre_z")

missing_cols <- setdiff(req_cols, names(df))
if (length(missing_cols) > 0) {
  write.csv(
    data.frame(status = "missing_cols",
               msg    = paste(missing_cols, collapse = ","),
               rs = 0L, ar1 = 1L),
    out_csv, row.names = FALSE
  )
  quit(status = 0)
}

df <- df[complete.cases(df[req_cols]), ]

if (nrow(df) < 6 || length(unique(df$subject)) < 3) {
  write.csv(
    data.frame(status = "insufficient_data", rs = 0L, ar1 = 1L),
    out_csv, row.names = FALSE
  )
  quit(status = 0)
}

# ── Model specification ───────────────────────────────────────────────────────
# Fixed formula: DV ~ ts_z [+ pre_z]
fixed_f <- if (include_pre) {
  as.formula("dv ~ ts_z + pre_z")
} else {
  as.formula("dv ~ ts_z")
}

# AR(1) correlation: consecutive timescales within subject are autocorrelated
# corAR1(form = ~ timescale | subject) uses the integer timescale as the
# "time" axis for ordering autocorrelation within each subject's 6 observations
corr_spec <- corAR1(form = ~ timescale | subject)

ctrl <- lmeControl(
  opt       = "optim",
  maxIter   = 300,
  msMaxIter = 300,
  tolerance = 1e-5,
  returnObject = TRUE   # return even if convergence marginal
)

# ── Attempt 1: random slope for ts_z + AR(1) ─────────────────────────────────
# This matches McDonough 2019 exactly:
#   random = ~ ts_z | subject  (random intercept AND slope per subject)
tab <- tryCatch({
  m <- lme(
    fixed       = fixed_f,
    random      = ~ ts_z | subject,
    data        = df,
    correlation = corr_spec,
    method      = "ML",
    control     = ctrl
  )
  t          <- as.data.frame(summary(m)$tTable)
  t$term     <- rownames(t)
  t$rs       <- 1L
  t$ar1      <- 1L
  t$status   <- "ok"
  t
}, error = function(e1) {

  # ── Attempt 2: random intercept only + AR(1) ─────────────────────────────
  # Random slope + AR(1) is over-parameterised for small n — fall back to
  # random intercept only while keeping AR(1)
  tryCatch({
    m <- lme(
      fixed       = fixed_f,
      random      = ~ 1 | subject,
      data        = df,
      correlation = corr_spec,
      method      = "ML",
      control     = ctrl
    )
    t          <- as.data.frame(summary(m)$tTable)
    t$term     <- rownames(t)
    t$rs       <- 0L
    t$ar1      <- 1L
    t$status   <- "ok_ri_only"
    t
  }, error = function(e2) {
    data.frame(
      status = "error",
      msg    = conditionMessage(e2),
      rs     = 0L,
      ar1    = 1L,
      stringsAsFactors = FALSE
    )
  })
})

write.csv(tab, out_csv, row.names = FALSE)
