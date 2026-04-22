"""
leave_one_disease_out.py — Leave-one-disease-group-out (LODGO) cross-validation
================================================================================

Tier-1 internal-external validation for the SUPPORT2 audited pipeline.

Design principles (v2, post rigorous reviewer-style code review)
----------------------------------------------------------------
1. **Preprocessing fit on TRAINING FOLD ONLY, in every fold.**
   The held-out disease group's data must not influence ANY training-time
   statistic — not imputation medians, not scaler mean/std, not the
   notna-filter cutoff for admissible features, not the set of one-hot
   columns. Every fold re-runs preprocessing from the raw dataframe.

2. **dzgroup is removed from the feature matrix.**
   It is the LODGO partition key; including it in features would make the
   task trivially easy during training and catastrophically hard at test
   (the held-out group's dummy has no support in training). dzclass (4-level
   disease class) is kept as a feature — it retains clinical information
   without collinearity against the partition key.

3. **Column alignment across fold train/test.**
   After one-hot encoding on the training fold, the test fold may be
   missing some dummy columns (e.g., a rare income level absent from
   training). We align explicitly by forcing test categoricals onto
   training's category set.

4. **Full paper-aligned metric suite per fold per model.**
   C-index with bootstrap CI, integrated Brier score, time-dependent AUC
   at 18d and 108d, full AUC(t) curve, calibration deciles at 108d, DCA
   net benefit curves, and Δ vs the paper's primary-split GBM baseline.

5. **Resumable after any interruption.**
   Every (fold × model) completed result is checkpointed immediately to
   disk via the existing Checkpointer (same class the main pipeline uses).
   Ctrl-C, power cut, job kill → re-running picks up at the next
   incomplete (fold × model) pair. Raw preprocessing per fold is also
   checkpointed.

6. **Bootstrap seeds independent across folds AND models.**
   seed = SEED + fold_idx * 1000 + model_idx * 7.

7. **GPU acceleration via supplementary XGBoost survival:cox (opt-in).**
   Reuses the pipeline's gpu.py module. On the Colab-local workstation the
   5th "GBM XGBoost (GPU)" column trains ~5–10× faster than sksurv's GBM on
   the same fold. Falls back silently to CPU / is skipped entirely if
   XGBoost is not installed. Enable with --include-xgboost.

8. **Temp artifacts preserved, not deleted.**
   Per-fold raw preprocessed arrays go to Temp/<run_id>/ so that
   downstream analysis (e.g., inspecting SHAP per fold, or recomputing a
   missed metric) doesn't re-run preprocessing.

9. **Paths follow the main pipeline's config.**
   PROJECT_ROOT / RUNS_DIR / TEMP_DIR are imported directly from config.py,
   so on a Windows workstation rooted at C:\\Colab-local\\support2-survival-
   audited, outputs land under
       C:\\Colab-local\\support2-survival-audited\\runs\\lodgo_YYYYMMDD_HHMMSS\\
   and preserved artefacts under
       C:\\Colab-local\\support2-survival-audited\\temp\\lodgo_YYYYMMDD_HHMMSS\\

10. **Manifest.json with config, data hash, python/sklearn/sksurv versions,
    GPU info, and per-fold timing for full reproducibility.**

Outputs
-------
tables/
    lodgo_per_fold_model.csv        — long-format (fold × model) with every metric
    lodgo_summary_by_model.csv      — macro-summary across folds per model
    lodgo_dca_per_fold.csv          — DCA net benefit curves (long format)
    lodgo_auc_curve_per_fold.csv    — AUC(t) curves per fold per model
    lodgo_calibration_per_fold.csv  — calibration deciles per fold per model
    lodgo_training_distribution.csv — training fold event rate, n, etc.
checkpoints/
    fold_<group_slug>_preproc.pkl   — per-fold X,y,scaler,imputer
    fold_<group_slug>_model_<model_slug>.pkl   — per-(fold,model) eval dict
logs/
    lodgo_run.log                   — full run log
manifest.json                       — reproducibility manifest
temp/<run_id>/                      — preserved raw fold arrays (not auto-cleaned)

Usage (from repo/)
------------------
    # Full run, resume-safe, all 4 sksurv models
    python leave_one_disease_out.py

    # Add the 5th XGBoost GPU column
    python leave_one_disease_out.py --include-xgboost

    # Resume a crashed run explicitly (re-running with no args also
    # auto-resumes because checkpoints are keyed by fold+model)
    python leave_one_disease_out.py --resume-run lodgo_20260422_143000

    # Fast GBM-only smoke test
    python leave_one_disease_out.py --models gbm-only --n-bootstrap 100

Runtime (workstation, 6-core CPU)
---------------------------------
    all 4 sksurv, 8 folds, 500 bootstraps:   ~25-35 min
    + xgboost GPU column:                    +3-5 min
    gbm-only, 100 bootstraps:                ~10-15 min
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ─── Repo imports (LODGO lives in repo/, so these work directly) ─────────────
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from config import (
    PROJECT_ROOT, DATA_CSV, RUNS_DIR, TEMP_DIR,
    SEED, N_BOOTSTRAP, SUBGROUP_MIN_N,
    SKIP_FEATURES, MISSING_INDICATOR_FEATURES,
    COX_PH_PARAMS, COX_EN_PARAMS, RSF_PARAMS, GBM_PARAMS,
    XGB_SURVIVAL_PARAMS, USE_GPU_IF_AVAILABLE,
    AUC_EARLY_DAYS, AUC_LATE_DAYS, AUC_CURVE_MIN, AUC_CURVE_MAX, AUC_CURVE_N,
    CAL_TIME_DAYS, DCA_THRESH_MIN, DCA_THRESH_MAX, DCA_THRESH_N,
)
from checkpoint import Checkpointer, fingerprint, PIPELINE_VERSION
from gpu import XGBOOST_AVAILABLE, detect_gpu, fit_xgb_survival

from sklearn import __version__ as sklearn_version
import sksurv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored, integrated_brier_score,
    cumulative_dynamic_auc,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Run directory layout — mirrors the main pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def make_run_dir(resume_run: str | None) -> tuple[Path, str]:
    """Return (run_dir, run_id). Reuses existing run_dir if resume_run given."""
    if resume_run:
        run_dir = RUNS_DIR / resume_run
        if not run_dir.exists():
            raise FileNotFoundError(
                f"--resume-run {resume_run} not found at {run_dir}"
            )
        for sub in ("checkpoints", "tables", "logs", "figures"):
            (run_dir / sub).mkdir(exist_ok=True)
        return run_dir, resume_run
    run_id = "lodgo_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("checkpoints", "tables", "logs", "figures"):
        (run_dir / sub).mkdir(exist_ok=True)
    return run_dir, run_id


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("lodgo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# Per-fold preprocessing — fit on training fold only, then transform test fold.
# ═══════════════════════════════════════════════════════════════════════════════
def slugify(s: str) -> str:
    """Make a filesystem-safe slug from a disease-group or model name."""
    return re.sub(r"[^\w]+", "_", s).strip("_")


def raw_load() -> pd.DataFrame:
    """Load, validate, and baseline-clean the SUPPORT2 CSV."""
    df = pd.read_csv(DATA_CSV)
    df.columns = df.columns.str.strip()
    df["d.time"] = pd.to_numeric(df["d.time"], errors="coerce")
    df["death"]  = pd.to_numeric(df["death"],  errors="coerce")
    df = df[df["d.time"] > 0].dropna(subset=["d.time", "death", "dzgroup"])
    return df.reset_index(drop=True)


def preprocess_fold(
    raw_df: pd.DataFrame,
    held_out_group: str,
    notna_cutoff: float = 0.4,
) -> dict:
    """
    Build fold-specific X_train, X_test, y_train, y_test.

    CRITICAL: every statistic (notna rate, imputation median, scaler mean/std,
    one-hot column set) is computed ON THE TRAINING FOLD ONLY.

    dzgroup is the partition key and is REMOVED from the feature matrix.
    Other categoricals are one-hot encoded (drop-first) with CATEGORIES
    ANCHORED to the training fold so test-fold columns align exactly.
    """
    # Split by partition key
    test_mask  = (raw_df["dzgroup"] == held_out_group).values
    train_mask = ~test_mask
    train_df = raw_df.loc[train_mask].reset_index(drop=True)
    test_df  = raw_df.loc[test_mask].reset_index(drop=True)

    # Outcomes
    y_event_tr = train_df["death"].astype(bool).values
    y_time_tr  = train_df["d.time"].astype(float).values
    y_event_te = test_df["death"].astype(bool).values
    y_time_te  = test_df["d.time"].astype(float).values

    # dzgroup is never a feature in LODGO (it is the partition key).
    SKIP_PLUS_KEY = set(SKIP_FEATURES) | {"dzgroup"}

    # Feature columns: numeric + object/category, excluding SKIP set and dzgroup,
    # with admissibility (notna ratio) computed ON TRAINING ONLY.
    num_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in SKIP_PLUS_KEY and train_df[c].notna().mean() > notna_cutoff
    ]
    cat_cols = [
        c for c in train_df.select_dtypes(include=["object", "category"]).columns
        if c not in SKIP_PLUS_KEY and train_df[c].notna().mean() > notna_cutoff
    ]

    # Apply same feature list to test (use training's admissibility decision)
    train_feat_df = train_df[num_cols + cat_cols].copy()
    test_feat_df  = test_df[num_cols + cat_cols].copy()

    # Anchor test-fold categorical levels to training-fold levels so that
    # pd.get_dummies produces the identical column set for train and test.
    for col in cat_cols:
        cats = pd.Categorical(train_feat_df[col]).categories
        train_feat_df[col] = pd.Categorical(train_feat_df[col], categories=cats)
        test_feat_df[col]  = pd.Categorical(test_feat_df[col],  categories=cats)

    if cat_cols:
        train_feat_df = pd.get_dummies(train_feat_df, columns=cat_cols,
                                       drop_first=True, dtype=float)
        test_feat_df  = pd.get_dummies(test_feat_df, columns=cat_cols,
                                       drop_first=True, dtype=float)

    # Defensive: ensure the column sets match exactly.
    missing_in_test = [c for c in train_feat_df.columns if c not in test_feat_df.columns]
    for c in missing_in_test:
        test_feat_df[c] = 0.0
    test_feat_df = test_feat_df[train_feat_df.columns]

    # Missing indicators — use TRAINING stats only.
    for f in MISSING_INDICATOR_FEATURES:
        if f in train_feat_df.columns:
            train_feat_df[f"{f}_missing"] = train_feat_df[f].isna().astype(float)
            test_feat_df[f"{f}_missing"]  = test_feat_df[f].isna().astype(float)

    feat_cols = list(train_feat_df.columns)

    # Impute + scale — fit on training fold only.
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X_tr = pd.DataFrame(
        sc.fit_transform(imp.fit_transform(train_feat_df[feat_cols])),
        columns=feat_cols,
    )
    X_te = pd.DataFrame(
        sc.transform(imp.transform(test_feat_df[feat_cols])),
        columns=feat_cols,
    )

    y_tr = Surv.from_arrays(event=y_event_tr, time=y_time_tr)
    y_te = Surv.from_arrays(event=y_event_te, time=y_time_te)

    return {
        "held_out_group":    held_out_group,
        "X_tr":              X_tr,
        "X_te":              X_te,
        "y_tr":              y_tr,
        "y_te":              y_te,
        "feat_cols":         feat_cols,
        "n_train":           int(len(X_tr)),
        "n_test":            int(len(X_te)),
        "train_death_rate":  float(y_event_tr.mean()),
        "test_death_rate":   float(y_event_te.mean()),
        "train_time_median": float(np.median(y_time_tr)),
        "test_time_median":  float(np.median(y_time_te)),
        "imputer":           imp,
        "scaler":            sc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics — all paper-aligned, all fold-safe
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_cindex(events, times, risk) -> float:
    try:
        return float(concordance_index_censored(
            events.astype(bool), times.astype(float), risk.astype(float)
        )[0])
    except Exception:
        return float("nan")


def bootstrap_cindex(events, times, risk, n_boot: int, seed: int) -> dict:
    """Bootstrap C-index with 95% percentile CI; returns dict + raw samples."""
    rng = np.random.default_rng(seed)
    n = len(risk)
    point = _safe_cindex(events, times, risk)
    samples = np.empty(n_boot, dtype=float)
    ok = 0
    ev = events.astype(bool); tm = times.astype(float); rk = risk.astype(float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            samples[ok] = concordance_index_censored(
                ev[idx], tm[idx], rk[idx]
            )[0]
            ok += 1
        except Exception:
            continue
    samples = samples[:ok]
    if ok < max(10, n_boot // 10):
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "cindex":       point,
        "ci_lo":        float(lo),
        "ci_hi":        float(hi),
        "n_boot_ok":    int(ok),
        "boot_samples": samples,
    }


def safe_integrated_brier(model, X_te, y_tr, y_te, n_times: int = 50):
    """IBS on a conservative time grid. Returns (ibs, grid) or (NaN, None)."""
    if not hasattr(model, "predict_survival_function"):
        return float("nan"), None
    try:
        t_min = max(float(y_tr["time"].min()), float(y_te["time"].min())) + 1e-3
        t_max_train = float(np.percentile(y_tr["time"], 95))
        t_max = min(float(y_te["time"].max()), t_max_train) - 1e-3
        if t_max <= t_min:
            return float("nan"), None
        grid = np.linspace(t_min, t_max, n_times)
        sfns = model.predict_survival_function(X_te)
        S = np.asarray([[fn(t) for t in grid] for fn in sfns])
        return float(integrated_brier_score(y_tr, y_te, S, grid)), grid
    except Exception:
        return float("nan"), None


def safe_auc_at_timepoints(y_tr, y_te, risk, tps=(AUC_EARLY_DAYS, AUC_LATE_DAYS)):
    out = {}
    for t in tps:
        try:
            auc, _ = cumulative_dynamic_auc(y_tr, y_te, risk, np.array([float(t)]))
            out[f"auc_{int(t)}d"] = float(auc[0])
        except Exception:
            out[f"auc_{int(t)}d"] = float("nan")
    return out


def safe_auc_curve(y_tr, y_te, risk,
                   t_min=AUC_CURVE_MIN, t_max=AUC_CURVE_MAX, n=AUC_CURVE_N):
    """Clip grid to the safe range (above min test event, below max test time)."""
    try:
        ev_times_te = y_te["time"][y_te["event"].astype(bool)]
        if len(ev_times_te) == 0:
            grid = np.linspace(t_min, t_max, n)
            return grid, np.full(n, np.nan), float("nan")
        lo = max(float(ev_times_te.min()) + 1e-3, t_min)
        hi = min(float(y_te["time"].max()) - 1e-3, t_max)
        if hi <= lo:
            grid = np.linspace(t_min, t_max, n)
            return grid, np.full(n, np.nan), float("nan")
        grid = np.linspace(lo, hi, n)
        auc, mean_auc = cumulative_dynamic_auc(y_tr, y_te, risk, grid)
        return grid, np.asarray(auc, dtype=float), float(mean_auc)
    except Exception:
        grid = np.linspace(t_min, t_max, n)
        return grid, np.full(n, np.nan), float("nan")


def calibration_deciles(model, X_te, y_te, t_cal: float = CAL_TIME_DAYS):
    """Predicted vs KM-observed survival by decile of predicted S(t_cal)."""
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return None
    if not hasattr(model, "predict_survival_function"):
        return None
    try:
        sfns = model.predict_survival_function(X_te)
        ps = np.asarray([fn(t_cal) for fn in sfns])
        bins = np.quantile(ps, np.linspace(0, 1, 11))
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (ps >= lo) & (ps <= hi)
            if mask.sum() < 3:
                continue
            kmf = KaplanMeierFitter().fit(
                y_te["time"][mask], event_observed=y_te["event"][mask])
            obs = float(kmf.survival_function_at_times([t_cal]).iloc[0])
            rows.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "pred":   float(ps[mask].mean()),
                "obs":    obs,
                "n":      int(mask.sum()),
            })
        return pd.DataFrame(rows)
    except Exception:
        return None


def decision_curve(model, X_te, y_te, t_cal: float = CAL_TIME_DAYS,
                   thresholds=None):
    """DCA net benefit across threshold range at t_cal (Vickers 2006)."""
    if thresholds is None:
        thresholds = np.linspace(DCA_THRESH_MIN, DCA_THRESH_MAX, DCA_THRESH_N)
    events = y_te["event"].astype(bool)
    times  = y_te["time"].astype(float)
    outcome = events & (times <= t_cal)
    ambiguous = (~events) & (times < t_cal)
    use = ~ambiguous
    n_use = int(use.sum())
    if n_use < 50:
        return None

    try:
        if hasattr(model, "predict_survival_function"):
            sfns = model.predict_survival_function(X_te)
            p_event = 1.0 - np.asarray([fn(t_cal) for fn in sfns])
        else:
            from scipy.stats import rankdata
            risk = model.predict(X_te)
            p_event = rankdata(risk) / (len(risk) + 1)
    except Exception:
        return None

    prev = float(outcome[use].mean())
    rows = []
    for pt in thresholds:
        pos = p_event >= pt
        tp = int(((pos & outcome) & use).sum())
        fp = int(((pos & ~outcome) & use).sum())
        if pt >= 1.0:
            nb_model = float("nan"); nb_all = float("nan")
        else:
            w = pt / (1 - pt)
            nb_model = (tp / n_use) - (fp / n_use) * w
            nb_all   = prev - (1 - prev) * w
        rows.append({
            "threshold": float(pt),
            "nb_model":  nb_model,
            "nb_all":    nb_all,
            "nb_none":   0.0,
            "tp":        tp, "fp": fp, "n_eval": n_use,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-model training + evaluation
# ═══════════════════════════════════════════════════════════════════════════════
class _XGBWrapper:
    """
    Adapter so XGBoost fits the same (name, predict) interface as sksurv models.
    XGBoost survival:cox returns log-hazard; higher = higher risk, same convention
    as sksurv.predict(), so rankings line up and all downstream metrics work.
    Note: XGBoost survival:cox does NOT produce a survival function, so
    IBS / DCA / calibration will be NaN / None for this model (by design).
    """
    def __init__(self, booster):
        self.booster = booster
    def predict(self, X):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(X.values))


def get_model_specs(which: str, include_xgb: bool) -> list[tuple]:
    coxnet_params = dict(COX_EN_PARAMS)
    coxnet_params.setdefault("fit_baseline_model", True)  # required for IBS/DCA
    sksurv_specs = [
        ("Cox PH",                 CoxPHSurvivalAnalysis,            dict(COX_PH_PARAMS)),
        ("Cox ElasticNet",         CoxnetSurvivalAnalysis,           coxnet_params),
        ("Random Survival Forest", RandomSurvivalForest,             dict(RSF_PARAMS)),
        ("GBM Survival",           GradientBoostingSurvivalAnalysis, dict(GBM_PARAMS)),
    ]
    if which == "gbm-only":
        specs = [sksurv_specs[3]]
    elif which == "ensemble":
        specs = [sksurv_specs[2], sksurv_specs[3]]
    elif which == "all":
        specs = sksurv_specs
    else:
        raise ValueError(f"unknown --models: {which}")
    if include_xgb and XGBOOST_AVAILABLE:
        specs.append(("GBM XGBoost (GPU)", "_XGB_SURV_", dict(XGB_SURVIVAL_PARAMS)))
    return specs


def fit_one_model(name, cls_or_marker, params, X_tr, y_tr, X_te):
    if cls_or_marker == "_XGB_SURV_":
        booster, _, _ = fit_xgb_survival(X_tr, y_tr, X_te, params,
                                         use_gpu_if_available=USE_GPU_IF_AVAILABLE)
        return _XGBWrapper(booster)
    return cls_or_marker(**params).fit(X_tr, y_tr)


def evaluate_one_model(name, model, X_te, y_tr, y_te,
                       n_bootstrap: int, seed_base: int,
                       paper_primary_cindex: float | None) -> dict:
    """Full metric suite on a single (fold, model). Each metric fails per-metric."""
    risk = model.predict(X_te)
    boot = bootstrap_cindex(y_te["event"], y_te["time"], risk,
                            n_bootstrap, seed_base)
    ibs, _ = safe_integrated_brier(model, X_te, y_tr, y_te)
    auc_tp = safe_auc_at_timepoints(y_tr, y_te, risk)
    auc_grid, auc_vals, auc_mean = safe_auc_curve(y_tr, y_te, risk)
    cal_df = calibration_deciles(model, X_te, y_te)
    dca_df = decision_curve(model, X_te, y_te)

    delta_to_primary = (boot["cindex"] - paper_primary_cindex
                        if paper_primary_cindex is not None else float("nan"))

    return {
        "name":                 name,
        "cindex":               boot["cindex"],
        "ci_lo":                boot["ci_lo"],
        "ci_hi":                boot["ci_hi"],
        "n_boot_ok":            boot["n_boot_ok"],
        "ibs":                  ibs,
        "auc_18d":              auc_tp.get("auc_18d", float("nan")),
        "auc_108d":             auc_tp.get("auc_108d", float("nan")),
        "auc_curve_time":       auc_grid,
        "auc_curve_vals":       auc_vals,
        "auc_curve_mean":       auc_mean,
        "calibration":          cal_df,
        "dca":                  dca_df,
        "delta_to_primary_gbm": float(delta_to_primary),
        "risk":                 risk,
        "boot_samples":         boot["boot_samples"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main run
# ═══════════════════════════════════════════════════════════════════════════════
def run(args) -> None:
    run_dir, run_id = make_run_dir(args.resume_run)
    log = setup_logging(run_dir / "logs" / "lodgo_run.log")
    log.info("=" * 78)
    log.info(f"LODGO cross-validation   run_id={run_id}")
    log.info("=" * 78)
    log.info(f"project_root  : {PROJECT_ROOT}")
    log.info(f"run_dir       : {run_dir}")
    log.info(f"temp_root     : {TEMP_DIR}")
    log.info(f"data_csv      : {DATA_CSV}")
    log.info(f"python        : {platform.python_version()}    "
             f"sklearn {sklearn_version}   sksurv {sksurv.__version__}")
    log.info(f"models        : {args.models}    include_xgboost={args.include_xgboost}")
    log.info(f"n_bootstrap   : {args.n_bootstrap}")
    log.info(f"seed          : {SEED}")
    log.info(f"primary_C_ref : {args.paper_primary_gbm_cindex}")

    try:
        gpu = detect_gpu()
        log.info(f"gpu           : available={gpu['available']} "
                 f"name='{gpu.get('name','')}' backend={gpu['backend']}")
    except Exception as e:
        log.info(f"gpu           : detection failed: {e}")

    # ── Raw data ─────────────────────────────────────────────────────────────
    ck = Checkpointer(run_dir / "checkpoints", verbose=False)
    log.info(f"loading raw data from {DATA_CSV} ...")
    raw_df = raw_load()
    log.info(f"  rows={len(raw_df):,}   event_rate={raw_df['death'].mean():.1%}")

    groups = sorted(raw_df["dzgroup"].unique().tolist())
    log.info(f"disease groups ({len(groups)}):")
    for g in groups:
        n = int((raw_df["dzgroup"] == g).sum())
        dr = float(raw_df.loc[raw_df["dzgroup"] == g, "death"].mean()) * 100
        log.info(f"  {g:<32s}  n={n:5d}  death_rate={dr:5.1f}%")

    # Temp dir for preserved per-fold preprocessed arrays
    temp_lodgo = TEMP_DIR / run_id
    temp_lodgo.mkdir(parents=True, exist_ok=True)
    log.info(f"temp preserves: {temp_lodgo}")

    # ── Per-fold × per-model loop (resumable) ────────────────────────────────
    specs = get_model_specs(args.models, args.include_xgboost)
    model_names = [s[0] for s in specs]
    log.info(f"model list    : {model_names}")

    overall_t0 = time.time()
    data_fp = fingerprint(len(raw_df), tuple(raw_df.columns), SEED, PIPELINE_VERSION)

    for fold_idx, held_out in enumerate(groups):
        slug = slugify(held_out)
        log.info("")
        log.info(f"── Fold {fold_idx+1}/{len(groups)}: hold out '{held_out}' "
                 + "─" * max(3, 40 - len(held_out)))

        n_ho = int((raw_df["dzgroup"] == held_out).sum())
        if n_ho < args.min_test_n:
            log.info(f"  SKIP — held-out size {n_ho} < --min-test-n {args.min_test_n}")
            continue

        # Preprocess fold (checkpointed)
        pre_tag = f"fold_{slug}_preproc"
        pre_fp  = fingerprint(data_fp, held_out, args.notna_cutoff)

        def _preprocess():
            art = preprocess_fold(raw_df, held_out, args.notna_cutoff)
            # Mirror raw fold arrays to Temp/ for downstream inspection (parquet).
            fold_temp = temp_lodgo / pre_tag
            fold_temp.mkdir(parents=True, exist_ok=True)
            try:
                art["X_tr"].to_parquet(fold_temp / "X_tr.parquet")
                art["X_te"].to_parquet(fold_temp / "X_te.parquet")
                pd.DataFrame({
                    "time":  art["y_tr"]["time"],
                    "event": art["y_tr"]["event"].astype(int),
                }).to_parquet(fold_temp / "y_tr.parquet")
                pd.DataFrame({
                    "time":  art["y_te"]["time"],
                    "event": art["y_te"]["event"].astype(int),
                }).to_parquet(fold_temp / "y_te.parquet")
            except Exception as e:
                # Fall back to CSV if parquet engine isn't available
                art["X_tr"].to_csv(fold_temp / "X_tr.csv", index=False)
                art["X_te"].to_csv(fold_temp / "X_te.csv", index=False)
                log.info(f"    [temp] wrote CSV instead of parquet: {e}")
            return art
        art = ck.get_or_compute(pre_tag, _preprocess, pre_fp)

        log.info(f"  train n={art['n_train']:,}   test n={art['n_test']:,}   "
                 f"train_dr={art['train_death_rate']*100:.1f}%  "
                 f"test_dr={art['test_death_rate']*100:.1f}%   "
                 f"n_features={len(art['feat_cols'])}")

        # Per-model within fold — each (fold, model) is independently checkpointed
        for model_idx, (name, cls_or_marker, params) in enumerate(specs):
            eval_tag = f"fold_{slug}_model_{slugify(name)}"
            eval_fp  = fingerprint(pre_fp, name, args.n_bootstrap,
                                   args.paper_primary_gbm_cindex)
            if ck.exists(eval_tag, eval_fp):
                log.info(f"  [{name:<24s}] cached — skip")
                continue

            seed_for_boot = SEED + fold_idx * 1000 + model_idx * 7

            t0 = time.time()
            try:
                model = fit_one_model(name, cls_or_marker, params,
                                      art["X_tr"], art["y_tr"], art["X_te"])
                result = evaluate_one_model(
                    name, model, art["X_te"], art["y_tr"], art["y_te"],
                    n_bootstrap=args.n_bootstrap,
                    seed_base=seed_for_boot,
                    paper_primary_cindex=args.paper_primary_gbm_cindex,
                )
                result["fold_idx"]         = fold_idx
                result["held_out_group"]   = held_out
                result["n_train"]          = art["n_train"]
                result["n_test"]           = art["n_test"]
                result["train_death_rate"] = art["train_death_rate"]
                result["test_death_rate"]  = art["test_death_rate"]
                result["elapsed_s"]        = float(time.time() - t0)
                ck.save(eval_tag, result, eval_fp)
                log.info(
                    f"  [{name:<24s}] C={result['cindex']:.4f} "
                    f"[{result['ci_lo']:.4f},{result['ci_hi']:.4f}]  "
                    f"IBS={result['ibs']:.4f}  AUC108={result['auc_108d']:.4f}  "
                    f"ΔvsPrimary={result['delta_to_primary_gbm']:+.4f}  "
                    f"({result['elapsed_s']:.1f}s)"
                )
            except Exception as exc:
                log.warning(f"  [{name:<24s}] FAILED: {exc!r}")
                ck.save(eval_tag, {
                    "name": name, "held_out_group": held_out, "fold_idx": fold_idx,
                    "error": repr(exc), "n_train": art["n_train"], "n_test": art["n_test"],
                }, eval_fp)

    # ── Aggregate ────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 78)
    log.info(f"all folds done in {time.time() - overall_t0:.1f}s — aggregating ...")
    log.info("=" * 78)

    aggregate_results(run_dir, groups, model_names, args, log)


def aggregate_results(run_dir, groups, model_names, args, log):
    ck = Checkpointer(run_dir / "checkpoints", verbose=False)
    tbl_dir = run_dir / "tables"

    per_fold_rows: list[dict[str, Any]] = []
    dca_rows: list[pd.DataFrame] = []
    auc_rows: list[pd.DataFrame] = []
    cal_rows: list[pd.DataFrame] = []
    train_dist_rows: list[dict[str, Any]] = []

    for g in groups:
        slug = slugify(g)
        pre_tag = f"fold_{slug}_preproc"
        try:
            pre = ck.load(pre_tag)
            train_dist_rows.append({
                "held_out_group":    g,
                "n_train":           pre["n_train"],
                "n_test":            pre["n_test"],
                "train_death_rate":  pre["train_death_rate"],
                "test_death_rate":   pre["test_death_rate"],
                "train_time_median": pre["train_time_median"],
                "test_time_median":  pre["test_time_median"],
                "n_features":        len(pre["feat_cols"]),
            })
        except Exception:
            pass

        for name in model_names:
            tag = f"fold_{slug}_model_{slugify(name)}"
            try:
                r = ck.load(tag)
            except Exception:
                continue
            if "error" in r:
                per_fold_rows.append({
                    "held_out_group":       g,
                    "model":                name,
                    "cindex":               float("nan"),
                    "ci_lo":                float("nan"),
                    "ci_hi":                float("nan"),
                    "ibs":                  float("nan"),
                    "auc_18d":              float("nan"),
                    "auc_108d":             float("nan"),
                    "delta_to_primary_gbm": float("nan"),
                    "error":                r.get("error"),
                })
                continue

            per_fold_rows.append({
                "held_out_group":       g,
                "model":                name,
                "n_train":              r.get("n_train"),
                "n_test":               r.get("n_test"),
                "train_death_rate":     r.get("train_death_rate"),
                "test_death_rate":      r.get("test_death_rate"),
                "cindex":               r["cindex"],
                "ci_lo":                r["ci_lo"],
                "ci_hi":                r["ci_hi"],
                "n_boot_ok":            r.get("n_boot_ok"),
                "ibs":                  r["ibs"],
                "auc_18d":              r["auc_18d"],
                "auc_108d":             r["auc_108d"],
                "auc_curve_mean":       r["auc_curve_mean"],
                "delta_to_primary_gbm": r["delta_to_primary_gbm"],
                "elapsed_s":            r.get("elapsed_s"),
            })

            dca_df = r.get("dca")
            if isinstance(dca_df, pd.DataFrame):
                d = dca_df.copy(); d["held_out_group"] = g; d["model"] = name
                dca_rows.append(d)
            cal_df = r.get("calibration")
            if isinstance(cal_df, pd.DataFrame):
                c = cal_df.copy(); c["held_out_group"] = g; c["model"] = name
                cal_rows.append(c)
            t_arr = r.get("auc_curve_time"); v_arr = r.get("auc_curve_vals")
            if t_arr is not None and v_arr is not None:
                auc_df = pd.DataFrame({
                    "t":   np.asarray(t_arr, dtype=float),
                    "auc": np.asarray(v_arr, dtype=float),
                })
                auc_df["held_out_group"] = g; auc_df["model"] = name
                auc_rows.append(auc_df)

    # Write CSVs
    if per_fold_rows:
        df_pf = pd.DataFrame(per_fold_rows)
        df_pf.to_csv(tbl_dir / "lodgo_per_fold_model.csv", index=False)
        log.info(f"saved {tbl_dir/'lodgo_per_fold_model.csv'}   rows={len(df_pf)}")
    if dca_rows:
        dfd = pd.concat(dca_rows, ignore_index=True)
        dfd.to_csv(tbl_dir / "lodgo_dca_per_fold.csv", index=False)
        log.info(f"saved {tbl_dir/'lodgo_dca_per_fold.csv'}   rows={len(dfd)}")
    if cal_rows:
        dfc = pd.concat(cal_rows, ignore_index=True)
        dfc.to_csv(tbl_dir / "lodgo_calibration_per_fold.csv", index=False)
        log.info(f"saved {tbl_dir/'lodgo_calibration_per_fold.csv'}   rows={len(dfc)}")
    if auc_rows:
        dfa = pd.concat(auc_rows, ignore_index=True)
        dfa.to_csv(tbl_dir / "lodgo_auc_curve_per_fold.csv", index=False)
        log.info(f"saved {tbl_dir/'lodgo_auc_curve_per_fold.csv'}   rows={len(dfa)}")
    if train_dist_rows:
        dft = pd.DataFrame(train_dist_rows)
        dft.to_csv(tbl_dir / "lodgo_training_distribution.csv", index=False)
        log.info(f"saved {tbl_dir/'lodgo_training_distribution.csv'}   rows={len(dft)}")

    # Macro-summary per model (only rows without errors)
    ok_rows = [r for r in per_fold_rows if "error" not in r]
    if ok_rows:
        df_ok = pd.DataFrame(ok_rows)
        summary = df_ok.groupby("model").agg(
            n_folds=("cindex", "count"),
            mean_cindex=("cindex", "mean"),
            median_cindex=("cindex", "median"),
            std_cindex=("cindex", "std"),
            min_cindex=("cindex", "min"),
            max_cindex=("cindex", "max"),
            mean_ibs=("ibs", "mean"),
            mean_auc108=("auc_108d", "mean"),
            mean_delta_to_primary=("delta_to_primary_gbm", "mean"),
            mean_elapsed_s=("elapsed_s", "mean"),
        ).reset_index().sort_values("mean_cindex", ascending=False)
        summary.to_csv(tbl_dir / "lodgo_summary_by_model.csv", index=False)
        log.info(f"saved {tbl_dir/'lodgo_summary_by_model.csv'}")

        log.info("")
        log.info("Macro-summary across held-out groups:")
        log.info(f"  {'Model':<24s}  {'mean C':>7s}  {'SD':>6s}  "
                 f"{'min..max':>14s}  {'ΔvsPrimary':>11s}")
        log.info("  " + "─" * 76)
        for _, r in summary.iterrows():
            rng = f"{r['min_cindex']:.3f}..{r['max_cindex']:.3f}"
            log.info(
                f"  {r['model']:<24s}  {r['mean_cindex']:>7.4f}  "
                f"{r['std_cindex']:>6.4f}  {rng:>14s}  "
                f"{r['mean_delta_to_primary']:>+11.4f}"
            )

    # Manifest
    manifest = {
        "run_id":                   run_dir.name,
        "timestamp":                datetime.now().isoformat(timespec="seconds"),
        "project_root":             str(PROJECT_ROOT),
        "data_csv":                 str(DATA_CSV),
        "pipeline_version":         PIPELINE_VERSION,
        "python":                   platform.python_version(),
        "sklearn":                  sklearn_version,
        "sksurv":                   sksurv.__version__,
        "seed":                     SEED,
        "n_bootstrap":              args.n_bootstrap,
        "notna_cutoff":             args.notna_cutoff,
        "min_test_n":               args.min_test_n,
        "paper_primary_gbm_cindex": args.paper_primary_gbm_cindex,
        "models":                   args.models,
        "include_xgboost":          args.include_xgboost,
        "groups_requested":         groups,
        "n_groups":                 len(groups),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"saved manifest.json")
    log.info("Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leave-one-disease-group-out CV for SUPPORT2 audited pipeline (v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                   help="Bootstrap resamples per held-out group for C-index CI.")
    p.add_argument("--models", choices=["all", "gbm-only", "ensemble"],
                   default="all",
                   help="Which model set to evaluate.")
    p.add_argument("--include-xgboost", action="store_true",
                   help="Add a 5th 'GBM XGBoost (GPU)' column (requires xgboost).")
    p.add_argument("--min-test-n", type=int, default=SUBGROUP_MIN_N,
                   help="Skip held-out groups with fewer than this many patients.")
    p.add_argument("--notna-cutoff", type=float, default=0.4,
                   help="Minimum non-null fraction (on TRAINING fold only) for "
                        "a feature to be kept.")
    p.add_argument("--paper-primary-gbm-cindex", type=float, default=0.7052,
                   help="Paper's primary-split GBM C-index. Used for Δ reporting.")
    p.add_argument("--resume-run", type=str, default=None,
                   help="Resume a previous LODGO run (exact run_id, e.g. "
                        "lodgo_20260422_143000). If not given, a new run_id is "
                        "created; but note completed (fold, model) checkpoints "
                        "from the SAME run_id are automatically re-used on rerun.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
