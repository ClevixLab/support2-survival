"""
models.py — Train all survival models with identical preprocessing
====================================================================
Trains the 4 core models (Cox PH, Cox EN, RSF, GBM) with paper's
hyperparameters. Optionally trains XGBoost survival (GPU-capable)
as a supplementary model.

All models are fit on the TRAINING set only. Validation set is reserved
for HPO (not used here to match paper).
"""
from __future__ import annotations
import time
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from config import (
    COX_PH_PARAMS, COX_EN_PARAMS, RSF_PARAMS, GBM_PARAMS,
    XGB_SURVIVAL_PARAMS, USE_GPU_IF_AVAILABLE,
)
from gpu import XGBOOST_AVAILABLE, detect_gpu, fit_xgb_survival


def train_sksurv_models(Xtr, ytr) -> dict:
    """Train the 4 scikit-survival models (CPU)."""
    models = {}
    times  = {}

    specs = [
        ("Cox PH",                  CoxPHSurvivalAnalysis,           COX_PH_PARAMS),
        ("Cox ElasticNet",          CoxnetSurvivalAnalysis,          COX_EN_PARAMS),
        ("Random Survival Forest",  RandomSurvivalForest,            RSF_PARAMS),
        ("GBM Survival",            GradientBoostingSurvivalAnalysis,GBM_PARAMS),
    ]
    for name, cls, params in specs:
        print(f"[TRAIN] {name}  params={params}")
        t0 = time.time()
        m = cls(**params).fit(Xtr, ytr)
        times[name] = time.time() - t0
        models[name] = m
        print(f"  done in {times[name]:.1f}s")
    return {"models": models, "train_times": times}


def train_xgb_supplementary(Xtr, ytr, Xte) -> dict | None:
    """
    Optional: train XGBoost survival:cox as supplementary comparison.
    Returns None if XGBoost is not installed.
    """
    if not XGBOOST_AVAILABLE:
        print("[TRAIN] XGBoost not installed — skipping supplementary GPU model")
        return None

    gpu = detect_gpu()
    print(f"[TRAIN] XGBoost survival:cox  (GPU detected: {gpu['available']}, "
          f"device: {gpu.get('name', 'cpu')})")
    t0 = time.time()
    booster, risk_train, risk_test = fit_xgb_survival(
        Xtr, ytr, Xte, XGB_SURVIVAL_PARAMS,
        use_gpu_if_available=USE_GPU_IF_AVAILABLE,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")
    return {
        "booster":    booster,
        "risk_train": risk_train,
        "risk_test":  risk_test,
        "train_time": elapsed,
        "gpu_used":   gpu["available"] and USE_GPU_IF_AVAILABLE,
    }
