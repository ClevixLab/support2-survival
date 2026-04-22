"""
gpu.py — Optional GPU acceleration via XGBoost
================================================
scikit-survival's GradientBoostingSurvivalAnalysis is CPU-only.
We add an OPTIONAL parallel training of XGBoost survival:cox (GPU-capable)
as a supplementary model. This does NOT replace the paper's primary GBM result
(which uses sksurv); it is a separate comparison column.

If no GPU is available or XGBoost is not installed, the module silently skips
XGBoost and the rest of the pipeline runs as before.

Rationale: reviewers may ask "why not GPU-accelerated" and having a GPU model
trained on identical data with identical splits is a simple, defensible answer.
"""
from __future__ import annotations
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


def detect_gpu() -> dict:
    """
    Return a dict with: {"available": bool, "name": str, "backend": str}

    Tries (in order): CUDA via torch, CUDA via nvidia-smi, fallback CPU.
    """
    info = {"available": False, "name": "", "backend": "cpu"}

    # Try torch first (most reliable for CUDA detection)
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"]      = torch.cuda.get_device_name(0)
            info["backend"]   = "cuda (torch)"
            info["count"]     = torch.cuda.device_count()
            return info
    except ImportError:
        pass

    # Try nvidia-smi (no torch required)
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip()
        if out:
            info["available"] = True
            info["name"]      = out.split("\n")[0]
            info["backend"]   = "cuda (nvidia-smi)"
            return info
    except Exception:
        pass

    return info


def xgb_tree_method(use_gpu_if_available: bool = True) -> str:
    """Return the correct tree_method string for the detected hardware."""
    if not XGBOOST_AVAILABLE:
        return "hist"
    gpu = detect_gpu()
    if use_gpu_if_available and gpu["available"]:
        # XGBoost ≥ 2.0 syntax
        try:
            import xgboost
            major = int(xgboost.__version__.split(".")[0])
            return "hist"  # with device="cuda" set elsewhere (XGB 2.x)
        except Exception:
            return "gpu_hist"  # XGBoost < 2.0
    return "hist"


def fit_xgb_survival(X_train, y_train, X_test, params: dict,
                     use_gpu_if_available: bool = True):
    """
    Fit an XGBoost survival:cox model.

    Returns
    -------
    (model, risk_train, risk_test)  — raw risk scores (higher = higher risk)
    """
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is not installed — pip install xgboost")

    # Survival targets are encoded in y as negative=censored duration, positive=event duration
    # for XGBoost survival:cox: y > 0 = event observed at time y; y < 0 = censored at time -y.
    def _encode(y_struct):
        # sksurv.util.Surv structured array
        times  = y_struct["time"].astype(float)
        events = y_struct["event"].astype(bool)
        # survival:cox expects positive time for event, negative for censored
        return np.where(events, times, -times)

    import numpy as np
    dtrain = xgb.DMatrix(X_train.values, label=_encode(y_train))
    dtest  = xgb.DMatrix(X_test.values)

    p = dict(params)
    p["tree_method"] = xgb_tree_method(use_gpu_if_available)
    gpu = detect_gpu()
    if use_gpu_if_available and gpu["available"]:
        try:
            major = int(xgb.__version__.split(".")[0])
            if major >= 2:
                p["device"] = "cuda"
        except Exception:
            pass

    n_estimators = p.pop("n_estimators", 300)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = xgb.train(p, dtrain, num_boost_round=n_estimators,
                           verbose_eval=False)

    risk_train = booster.predict(dtrain)
    risk_test  = booster.predict(dtest)
    return booster, risk_train, risk_test
