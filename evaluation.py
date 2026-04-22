"""
evaluation.py — All evaluation metrics with parallel bootstrap
================================================================
Implements:
  - C-index with 95% bootstrap CI (parallel)
  - IBS (integrated Brier score) with safe time-grid
  - Time-dependent AUC at early (18d) and late (108d) timepoints
  - Full AUC(t) curve for Fig 3
  - Calibration data at t = 108d for Fig 4
  - Null-shifted bootstrap test for C-index differences (vs GBM)
  - Decision Curve Analysis (net benefit across threshold range)
  - Subgroup C-index per disease group with bootstrap CI
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed

from sksurv.metrics import (
    concordance_index_censored, integrated_brier_score,
    cumulative_dynamic_auc,
)

from config import (
    SEED, N_BOOTSTRAP, N_DELONG,
    AUC_EARLY_DAYS, AUC_LATE_DAYS, AUC_CURVE_MIN, AUC_CURVE_MAX, AUC_CURVE_N,
    CAL_TIME_DAYS, DCA_THRESH_MIN, DCA_THRESH_MAX, DCA_THRESH_N,
    SUBGROUP_MIN_N,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _c_from_idx(events, times, risk, idx):
    """Compute C-index on bootstrap sample. Returns NaN on failure."""
    try:
        return concordance_index_censored(events[idx], times[idx], risk[idx])[0]
    except Exception:
        return np.nan


def _parallel_bootstrap_cindex(events, times, risk, n_boot: int, n_jobs: int = -1):
    """
    Parallel bootstrap of C-index using joblib processes.
    sksurv's concordance_index_censored holds the GIL, so threads don't help;
    processes do. On Windows, processes have spawn overhead — we chunk work.
    """
    rng = np.random.default_rng(SEED)
    n = len(risk)
    # Pre-generate all bootstrap index arrays deterministically
    idxs = rng.integers(0, n, size=(n_boot, n))

    # For small-to-medium n_boot, a simple sequential loop beats process overhead
    if n_boot <= 500:
        results = [_c_from_idx(events, times, risk, idxs[i]) for i in range(n_boot)]
    else:
        # For larger counts, use processes; chunk to amortise spawn cost
        results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
            delayed(_c_from_idx)(events, times, risk, idxs[i]) for i in range(n_boot)
        )
    return np.array([r for r in results if not np.isnan(r)])


def cindex_with_ci(events, times, risk, n_boot: int = N_BOOTSTRAP):
    """
    Return dict: {"cindex": float, "ci_lo": float, "ci_hi": float,
                  "bootstrap_samples": np.ndarray}
    """
    ci = concordance_index_censored(events.astype(bool), times.astype(float),
                                     risk.astype(float))[0]
    boots = _parallel_bootstrap_cindex(events.astype(bool), times.astype(float),
                                        risk.astype(float), n_boot)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "cindex":            float(ci),
        "ci_lo":             float(lo),
        "ci_hi":             float(hi),
        "bootstrap_samples": boots,
        "n_boot_success":    len(boots),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IBS with safe time grid
# ═══════════════════════════════════════════════════════════════════════════════
def integrated_brier(model, Xte, ytr, yte, n_times: int = 50):
    """
    Compute Integrated Brier Score on a safe time grid.

    The grid must stay strictly inside (min_test_time, max_test_time) AND
    inside the support of the training KM censoring distribution.

    Returns (ibs, time_grid). Returns (NaN, None) if model lacks
    predict_survival_function.
    """
    if not hasattr(model, "predict_survival_function"):
        return np.nan, None

    # Safe range: strictly inside test time support
    t_min = max(float(ytr["time"].min()), float(yte["time"].min())) + 1e-3
    # Also bounded by training censoring support — use 95th percentile of training times
    t_max_train = float(np.percentile(ytr["time"], 95))
    t_max = min(float(yte["time"].max()), t_max_train) - 1e-3
    if t_max <= t_min:
        return np.nan, None

    times_grid = np.linspace(t_min, t_max, n_times)
    try:
        surv_fns = model.predict_survival_function(Xte)
        surv_mat = np.asarray([[fn(t) for t in times_grid] for fn in surv_fns])
        ibs = integrated_brier_score(ytr, yte, surv_mat, times_grid)
        return float(ibs), times_grid
    except Exception as e:
        print(f"    [IBS] failed: {e}")
        return np.nan, None


# ═══════════════════════════════════════════════════════════════════════════════
# Time-dependent AUC
# ═══════════════════════════════════════════════════════════════════════════════
def auc_at_timepoints(ytr, yte, risk, timepoints=(AUC_EARLY_DAYS, AUC_LATE_DAYS)):
    """Compute AUC at specific timepoints via IPCW. Returns dict."""
    out = {}
    for t in timepoints:
        try:
            auc, _ = cumulative_dynamic_auc(ytr, yte, risk, np.array([float(t)]))
            out[f"auc_{int(t)}d"] = float(auc[0])
        except Exception as e:
            print(f"    [AUC@{t}d] failed: {e}")
            out[f"auc_{int(t)}d"] = np.nan
    return out


def auc_curve(ytr, yte, risk,
              t_min=AUC_CURVE_MIN, t_max=AUC_CURVE_MAX, n=AUC_CURVE_N):
    """Full AUC(t) curve for Fig 3."""
    grid = np.linspace(t_min, t_max, n)
    try:
        auc, mean_auc = cumulative_dynamic_auc(ytr, yte, risk, grid)
        return grid, auc, float(mean_auc)
    except Exception as e:
        print(f"    [AUC curve] failed: {e}")
        return grid, np.full(n, np.nan), np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration at t = 108d
# ═══════════════════════════════════════════════════════════════════════════════
def calibration_data(model, Xte, yte, t_cal: float = CAL_TIME_DAYS):
    """
    Build calibration scatter (predicted vs KM-observed survival) at t_cal,
    binned by deciles of predicted survival.

    Returns DataFrame with columns: pred, obs, n, bin_lo, bin_hi.
    """
    from lifelines import KaplanMeierFitter
    if not hasattr(model, "predict_survival_function"):
        return None
    try:
        sfns = model.predict_survival_function(Xte)
        ps = np.array([fn(t_cal) for fn in sfns])
        bins = np.quantile(ps, np.linspace(0, 1, 11))
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (ps >= lo) & (ps <= hi)
            if mask.sum() < 3: continue
            kmf = KaplanMeierFitter().fit(
                yte["time"][mask], event_observed=yte["event"][mask])
            obs_surv = float(kmf.survival_function_at_times([t_cal]).iloc[0])
            rows.append({
                "bin_lo":  float(lo),
                "bin_hi":  float(hi),
                "pred":    float(ps[mask].mean()),
                "obs":     obs_surv,
                "n":       int(mask.sum()),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"    [calibration] failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Null-shifted bootstrap test for ΔC (paper's primary comparison test)
# ═══════════════════════════════════════════════════════════════════════════════
def null_shifted_delta_c(events, times, risk_a, risk_b, n_boot: int = N_DELONG):
    """
    Null-shifted bootstrap test for ΔC = C(A) - C(B).

    Method:
      1. observed ΔC on full data
      2. draw n_boot bootstrap resamples, compute ΔC* on each
      3. shift to null: ΔC*_null = ΔC* − ΔC_obs
      4. two-sided p-value = mean(|ΔC*_null| ≥ |ΔC_obs|), floored at 1/n_boot
      5. 95% CI is standard (un-shifted) percentile interval of ΔC*

    Returns dict with keys: C_A, C_B, delta_obs, ci_lo, ci_hi, p_value, n_boot_success
    """
    ev = events.astype(bool); tm = times.astype(float)
    ra = risk_a.astype(float); rb = risk_b.astype(float)

    c_a = concordance_index_censored(ev, tm, ra)[0]
    c_b = concordance_index_censored(ev, tm, rb)[0]
    delta_obs = c_a - c_b

    rng = np.random.default_rng(SEED)
    n = len(ra)

    def _one(idx):
        try:
            a = concordance_index_censored(ev[idx], tm[idx], ra[idx])[0]
            b = concordance_index_censored(ev[idx], tm[idx], rb[idx])[0]
            return a - b
        except Exception:
            return np.nan

    idxs = rng.integers(0, n, size=(n_boot, n))
    # sksurv holds GIL → sequential is fast enough; parallelism for very large counts
    if n_boot <= 1000:
        diffs = [_one(idxs[i]) for i in range(n_boot)]
    else:
        diffs = Parallel(n_jobs=-1, prefer="processes", batch_size="auto")(
            delayed(_one)(idxs[i]) for i in range(n_boot))
    diffs = np.array([d for d in diffs if not np.isnan(d)])

    if len(diffs) == 0:
        return None

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    # null-shift
    null_diffs = diffs - delta_obs
    p = max(float(np.mean(np.abs(null_diffs) >= np.abs(delta_obs))), 1.0 / n_boot)

    return {
        "C_A":             float(c_a),
        "C_B":             float(c_b),
        "delta_obs":       float(delta_obs),
        "ci_lo":           float(ci_lo),
        "ci_hi":           float(ci_hi),
        "p_value":         float(p),
        "n_boot_success":  len(diffs),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Decision Curve Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def decision_curve_analysis(model, Xte, yte,
                            thresholds=None, t_cal: float = CAL_TIME_DAYS):
    """
    Compute DCA net benefit across threshold range.

    Net benefit formula (Vickers 2006):
        NB = TP/n  −  FP/n × pt/(1-pt)

    We convert survival predictions → "probability of event by t_cal" = 1 − S(t_cal).

    Returns DataFrame with columns: threshold, nb_model, nb_all, nb_none
    """
    if thresholds is None:
        thresholds = np.linspace(DCA_THRESH_MIN, DCA_THRESH_MAX, DCA_THRESH_N)

    events = yte["event"].astype(bool)
    times  = yte["time"].astype(float)
    n = len(events)

    # Binary outcome: event observed by t_cal
    # (ignore censored patients with censor time < t_cal — they are ambiguous)
    outcome = events & (times <= t_cal)            # definite event by t_cal
    ambiguous = (~events) & (times < t_cal)         # censored before t_cal — exclude
    use = ~ambiguous

    n_use = use.sum()
    if n_use < 50:
        return None

    # Predicted probability of event by t_cal
    try:
        if hasattr(model, "predict_survival_function"):
            sfns = model.predict_survival_function(Xte)
            surv = np.array([fn(t_cal) for fn in sfns])
            p_event = 1 - surv
        else:
            # risk score → map to [0,1] via rank-scaling (for models without survival fn)
            risk = model.predict(Xte)
            # rank-based CDF ∈ (0,1) avoids distribution assumptions
            from scipy.stats import rankdata
            p_event = rankdata(risk) / (len(risk) + 1)
    except Exception as e:
        print(f"    [DCA] could not produce calibrated p_event: {e}")
        return None

    # event prevalence among non-ambiguous patients
    prev = outcome[use].mean()

    rows = []
    for pt in thresholds:
        # predict positive if p_event ≥ pt
        predicted_pos = p_event >= pt
        tp = ((predicted_pos & outcome) & use).sum()
        fp = ((predicted_pos & ~outcome) & use).sum()
        nb_model = (tp / n_use) - (fp / n_use) * pt / (1 - pt) if pt < 1 else np.nan
        # treat-all: everybody predicted positive
        nb_all = prev - (1 - prev) * pt / (1 - pt) if pt < 1 else np.nan
        # treat-none: always 0
        nb_none = 0.0
        rows.append({
            "threshold": float(pt),
            "nb_model":  float(nb_model) if nb_model == nb_model else np.nan,
            "nb_all":    float(nb_all) if nb_all == nb_all else np.nan,
            "nb_none":   float(nb_none),
            "tp":        int(tp),
            "fp":        int(fp),
            "n_eval":    int(n_use),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Subgroup analysis
# ═══════════════════════════════════════════════════════════════════════════════
def subgroup_cindex(risk, events, times, groups,
                    min_n: int = SUBGROUP_MIN_N, n_boot: int = 500):
    """
    C-index per disease group with bootstrap CI.
    Returns DataFrame with: group, n, cindex, ci_lo, ci_hi, death_rate.
    """
    rows = []
    for grp in np.unique(groups):
        mask = groups == grp
        if mask.sum() < min_n:
            continue
        result = cindex_with_ci(events[mask], times[mask], risk[mask], n_boot)
        rows.append({
            "group":         str(grp),
            "n":             int(mask.sum()),
            "cindex":        result["cindex"],
            "ci_lo":         result["ci_lo"],
            "ci_hi":         result["ci_hi"],
            "death_rate":    float(events[mask].mean()),
        })
    return pd.DataFrame(rows).sort_values("cindex", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation wrapper — evaluates one model fully
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model_full(name: str, model, Xte, ytr, yte) -> dict:
    """
    Run complete evaluation suite for one model. Returns dict with all metrics.
    """
    print(f"[EVAL] {name}")
    risk = model.predict(Xte) if hasattr(model, "predict") else None

    result = {"name": name}

    # Core discrimination
    t0 = time.time()
    result.update(cindex_with_ci(yte["event"], yte["time"], risk))
    print(f"  C-index = {result['cindex']:.4f} [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]"
          f"  ({time.time()-t0:.1f}s)")

    # IBS
    t0 = time.time()
    ibs, grid = integrated_brier(model, Xte, ytr, yte)
    result["ibs"] = ibs
    if not np.isnan(ibs):
        print(f"  IBS     = {ibs:.4f}  ({time.time()-t0:.1f}s)")

    # AUC at timepoints
    result.update(auc_at_timepoints(ytr, yte, risk))
    print(f"  AUC@{AUC_EARLY_DAYS}d = {result[f'auc_{AUC_EARLY_DAYS}d']:.4f}  "
          f"AUC@{AUC_LATE_DAYS}d = {result[f'auc_{AUC_LATE_DAYS}d']:.4f}")

    # Full AUC curve
    grid, auc_vals, mean_auc = auc_curve(ytr, yte, risk)
    result["auc_curve_time"]  = grid
    result["auc_curve_vals"]  = auc_vals
    result["auc_curve_mean"]  = mean_auc

    # Calibration
    result["calibration"] = calibration_data(model, Xte, yte)

    # DCA
    result["dca"] = decision_curve_analysis(model, Xte, yte)

    result["risk"] = risk
    return result
