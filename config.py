"""
config.py — Single source of truth for the SUPPORT2 audited pipeline
=====================================================================
All paths, hyperparameters, feature lists, and evaluation settings live here.
Edit this file rather than hunting through scripts.

IMPORTANT: Change `PROJECT_ROOT` if you move the project on your workstation.
Windows path format supported (use raw string ``r"C:\\..."`` or forward slashes).
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import platform
import os

# ══════════════════════════════════════════════════════════════════════════════
# PATHS — adjust PROJECT_ROOT if project is moved
# ══════════════════════════════════════════════════════════════════════════════
# On Windows workstation: C:\Colab-local\support2-survival-audited
# On Linux/Mac (Anthropic sandbox): /home/claude/support2-survival-audited
# PROJECT_ROOT auto-resolves from this config.py file's location, so the
# project works no matter where it's placed on the filesystem.
# If you need to pin it to a specific path, uncomment the appropriate line below.
PROJECT_ROOT = Path(__file__).resolve().parent

# Alternative: pin explicitly (uncomment if needed)
# if platform.system() == "Windows":
#     PROJECT_ROOT = Path(r"C:\Colab-local\Lechess_checkpoint\support2-survival-audited")
# else:
#     PROJECT_ROOT = Path("/home/claude/support2-survival-audited")

# Structure under PROJECT_ROOT:
#   data/                          — raw input CSV (read-only)
#   runs/YYYYMMDD_HHMMSS/          — time-stamped output folder per run
#     ├── checkpoints/             — pickled intermediate artifacts (for resume)
#     ├── figures/                 — final PNG/PDF figures
#     ├── tables/                  — CSV tables
#     ├── logs/                    — text logs with per-stage timings
#     └── manifest.json            — config + data hash + git sha for reproducibility
#   temp/                          — long-lived temporary artifacts (not deleted)
#     └── YYYYMMDD_HHMMSS_<tag>/
DATA_DIR   = PROJECT_ROOT / "data"
RUNS_DIR   = PROJECT_ROOT / "runs"
TEMP_DIR   = PROJECT_ROOT / "temp"

DATA_CSV   = DATA_DIR / "support2_full.csv"    # original SUPPORT2 from Harrell
RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR    = RUNS_DIR / RUN_ID

# Sub-directories for the current run
CKPT_DIR   = RUN_DIR / "checkpoints"
FIG_DIR    = RUN_DIR / "figures"
TBL_DIR    = RUN_DIR / "tables"
LOG_DIR    = RUN_DIR / "logs"

# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════
SEED        = 42
N_BOOTSTRAP = 500     # bootstrap samples for C-index 95% CI (paper: 500)
N_DELONG    = 1000    # bootstrap samples for null-shifted ΔC test (paper: 1000)

# ══════════════════════════════════════════════════════════════════════════════
# DATA SPLIT
# ══════════════════════════════════════════════════════════════════════════════
TEST_FRAC   = 0.15    # paper: 15% test
VAL_FRAC    = 0.15    # paper: 15% val (of total, so 15/85 of train+val pool)
STRATIFY_BY = "event" # paper: stratified by event indicator

# ══════════════════════════════════════════════════════════════════════════════
# AUDITED FEATURE SKIP LIST  (three-layer leakage audit)
# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — principled classification per Knaus 1995 + Harrell hbiostat docs
# Layer 2 — empirical marginal-C-index test (all skipped features have C > 0.60)
# Layer 3 — informative missingness test (fix via missing-indicator columns)
SKIP_FEATURES = {
    # ── OUTCOMES (never features) ────────────────────────────────────────────
    "d.time",      # follow-up time (survival outcome)
    "death",       # mortality indicator (outcome)
    # ── ALTERNATIVE OUTCOMES (leak main outcome) ─────────────────────────────
    "hospdead",    # in-hospital death (correlated outcome)
    "sfdm2",       # 2-month functional outcome (level 5 = died before 2m)
    # ── POST-EVENT / POST-DISCHARGE ──────────────────────────────────────────
    "slos",        # study length of stay (mechanically encodes death time)
    "charges",     # hospital charges (accumulate over stay)
    "totcst",      # total cost (same)
    "totmcst",     # total micro-cost (same)
    "avtisst",     # average TISS (post-event aggregate)
    # ── DERIVED FROM OUTCOME-FITTED MODELS ───────────────────────────────────
    "aps",         # APACHE III score (derived; Harrell advises exclude)
    "sps",         # SUPPORT score (derived; Harrell advises exclude)
    "surv2m",      # SUPPORT model 2-month estimate (direct leak)
    "surv6m",      # SUPPORT model 6-month estimate (direct leak)
    # ── TREATMENT DECISIONS (self-fulfilling prophecies) ─────────────────────
    "prg2m",       # physician's 2-month estimate
    "prg6m",       # physician's 6-month estimate
    "dnr",         # DNR order (less aggressive care → death)
    "dnrday",      # day DNR was ordered (timing = deterioration)
    # ── REDUNDANT FUNCTIONAL-STATUS FEATURES ─────────────────────────────────
    "adlp",        # 62% missing + informative-missingness; adlsc is pre-specified composite
    "adls",        # 32% missing + redundant with adlsc
}

# Features KEPT that paper v3 also skipped (we re-include these)
FEATURES_REINCLUDED = {
    "hday",        # days in hospital before study entry — Knaus 1995 official predictor
}

# Features for which we ADD a missing-indicator column (informative-missingness)
MISSING_INDICATOR_FEATURES = ["glucose", "bun", "urine"]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS  (identical to paper for comparability)
# ══════════════════════════════════════════════════════════════════════════════
COX_PH_PARAMS = {
    "alpha": 0.01,          # L2 regularization
    "ties":  "efron",
}
COX_EN_PARAMS = {
    "l1_ratio":             0.5,
    "alpha_min_ratio":      0.1,
    "max_iter":             1000,
    "fit_baseline_model":   True,   # needed for predict_survival_function → IBS, DCA
}
RSF_PARAMS = {
    "n_estimators":       300,
    "max_depth":          6,
    "min_samples_leaf":   15,
    "min_samples_split":  30,
    "max_features":       "sqrt",
    "n_jobs":             max(1, os.cpu_count() - 1),  # leave one core for OS
    "random_state":       SEED,
}
GBM_PARAMS = {
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        3,
    "min_samples_leaf": 15,
    "subsample":        0.8,
    "random_state":     SEED,
}

# Optional: XGBoost as additional model (GPU-capable) for comparison
# Does NOT replace paper's GBM — this is a supplementary analysis
XGB_SURVIVAL_PARAMS = {
    "objective":         "survival:cox",
    "eval_metric":       "cox-nloglik",
    "learning_rate":     0.05,
    "max_depth":         3,
    "min_child_weight":  15,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "n_estimators":      300,
    "tree_method":       "hist",     # "gpu_hist" if GPU detected at runtime
    "random_state":      SEED,
}

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
# Paper: AUC at 18d (10th pct event times) and 108d (25th pct)
AUC_EARLY_DAYS = 18
AUC_LATE_DAYS  = 108
# Paper: AUC curve from ~3d to ~460d (10th to 90th percentile)
AUC_CURVE_MIN  = 3.0
AUC_CURVE_MAX  = 460.0
AUC_CURVE_N    = 40

# Calibration time for Fig 4
CAL_TIME_DAYS  = 108

# DCA threshold range
DCA_THRESH_MIN = 0.05
DCA_THRESH_MAX = 0.95
DCA_THRESH_N   = 91    # step of 0.01
# DCA threshold of interest for net benefit comparison in text
DCA_THRESH_REPORT = 0.60

# Subgroup analysis: minimum test-set size
SUBGROUP_MIN_N = 30

# ══════════════════════════════════════════════════════════════════════════════
# SHAP SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
SHAP_N_BACKGROUND = 150        # background samples (K-means style summary of training set)
SHAP_N_EXPLAIN    = 500        # patients to explain from test set
SHAP_MAX_EVALS_MULT = 10       # max_evals = 10 × n_features  (was 2× — improve fidelity)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
DPI         = 300              # publication quality
FIG_FORMAT  = "png"            # also save PDF copy
PALETTE     = ["#185FA5", "#1D9E75", "#D85A30", "#7F77DD", "#888780",
               "#8B4513", "#4B0082", "#008B8B"]

# ══════════════════════════════════════════════════════════════════════════════
# GPU DETECTION (detected at runtime in gpu.py)
# ══════════════════════════════════════════════════════════════════════════════
USE_GPU_IF_AVAILABLE = True    # set to False to force CPU
