"""
data.py — Leakage-audited SUPPORT2 preprocessing
==================================================
Implements the three-layer audit described in the paper:
  Layer 1: principled classification (Knaus 1995 + Harrell hbiostat)
  Layer 2: empirical marginal-C-index test
  Layer 3: informative-missingness test with missing-indicator columns

All excluded/retained features are exported to a CSV for the paper's
supplementary table.

Returned artifact: dict with keys
    raw_df, clean_df, Xtr, Xval, Xte, ytr, yval, yte,
    feat_cols, idx_train, idx_val, idx_test,
    audit_report (DataFrame), data_hash (str)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from scipy.stats import spearmanr

from config import (
    DATA_CSV, SEED, TEST_FRAC, VAL_FRAC,
    SKIP_FEATURES, MISSING_INDICATOR_FEATURES, FEATURES_REINCLUDED,
)


# Reason codes for the audit report
AUDIT_REASONS = {
    "d.time":      ("OUTCOME",    "Follow-up time (survival outcome)"),
    "death":       ("OUTCOME",    "Mortality indicator (outcome)"),
    "hospdead":    ("OUTCOME_ALT","In-hospital death — alternative outcome, leaks"),
    "sfdm2":       ("POST-EVENT", "2-month functional outcome; level 5 = died"),
    "slos":        ("POST-EVENT", "Study length of stay — mechanically encodes death time"),
    "charges":     ("POST-EVENT", "Hospital charges — accumulates over stay"),
    "totcst":      ("POST-EVENT", "Total cost — accumulates over stay"),
    "totmcst":     ("POST-EVENT", "Total micro-cost — accumulates over stay"),
    "avtisst":     ("POST-EVENT", "Average TISS score — post-event aggregate"),
    "aps":         ("DERIVED",    "APACHE III score (Harrell advises exclude)"),
    "sps":         ("DERIVED",    "SUPPORT score (Harrell advises exclude)"),
    "surv2m":      ("DERIVED",    "SUPPORT model 2-month prediction (direct target leak)"),
    "surv6m":      ("DERIVED",    "SUPPORT model 6-month prediction (direct target leak)"),
    "prg2m":       ("TREATMENT",  "Physician 2-month estimate (self-fulfilling)"),
    "prg6m":       ("TREATMENT",  "Physician 6-month estimate (self-fulfilling)"),
    "dnr":         ("TREATMENT",  "DNR order (less aggressive care → death)"),
    "dnrday":      ("TREATMENT",  "Day of DNR order (timing = deterioration)"),
    "adlp":        ("REDUNDANT",  "62% missing; replaced by adlsc composite"),
    "adls":        ("REDUNDANT",  "32% missing; replaced by adlsc composite"),
}


def compute_data_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the raw data for the manifest."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(str(df.shape).encode())
    h.update(str(tuple(df.columns)).encode())
    return h.hexdigest()


def marginal_cindex(values: np.ndarray, event: np.ndarray, time: np.ndarray) -> float:
    """Compute C-index of a single feature (for empirical leakage test)."""
    from sksurv.metrics import concordance_index_censored
    mask = ~np.isnan(values.astype(float))
    if mask.sum() < 100:
        return np.nan
    try:
        ci = concordance_index_censored(
            event[mask].astype(bool), time[mask].astype(float), values[mask].astype(float)
        )[0]
        return ci
    except Exception:
        return np.nan


def audit_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the three-layer audit report for every column in raw SUPPORT2.
    This is the content of the paper's supplementary Table S1.
    """
    rows = []
    events = df["death"].astype(bool).values
    times  = df["d.time"].astype(float).values
    for col in df.columns:
        miss_pct = df[col].isna().mean() * 100
        # Missingness → outcome difference
        miss_mask = df[col].isna().values
        if miss_mask.sum() >= 50 and (~miss_mask).sum() >= 50:
            dr_miss = df.loc[miss_mask, "death"].mean() * 100
            dr_obs  = df.loc[~miss_mask, "death"].mean() * 100
            miss_diff = dr_miss - dr_obs
        else:
            miss_diff = np.nan
        # Marginal C-index
        if pd.api.types.is_numeric_dtype(df[col]):
            ci = marginal_cindex(df[col].values, events, times)
        else:
            # factorize categoricals for marginal test
            codes = pd.factorize(df[col])[0].astype(float)
            codes[codes < 0] = np.nan
            ci = marginal_cindex(codes, events, times)
        ci_strength = max(ci, 1 - ci) if not np.isnan(ci) else np.nan

        # Decision
        layer1, reason = AUDIT_REASONS.get(col, ("ADMISSION", "Baseline / day-3 feature"))
        if col in SKIP_FEATURES:
            decision = "SKIP"
        else:
            decision = "USE"
        # Flag informative missingness
        if col in MISSING_INDICATOR_FEATURES:
            miss_handling = "+ missing indicator"
        else:
            miss_handling = ""

        rows.append({
            "Feature":       col,
            "Missing %":     round(miss_pct, 1),
            "Marginal C":    round(ci, 4) if not np.isnan(ci) else np.nan,
            "|C - 0.5|+0.5": round(ci_strength, 4) if not np.isnan(ci_strength) else np.nan,
            "Death rate diff (miss - obs, pp)": round(miss_diff, 1) if not np.isnan(miss_diff) else np.nan,
            "Layer 1 category":  layer1,
            "Layer 1 reason":    reason,
            "Decision":          decision,
            "Preprocessing note": miss_handling,
        })
    return pd.DataFrame(rows).sort_values(
        ["Decision", "|C - 0.5|+0.5"], ascending=[True, False]
    ).reset_index(drop=True)


def load_and_preprocess(data_path: Path = DATA_CSV) -> dict:
    """
    Load SUPPORT2 and apply leakage-audited preprocessing.

    Returns a dict of all artifacts needed by downstream stages.
    """
    print(f"[DATA] loading {data_path}")
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"SUPPORT2 CSV not found at {data_path}. "
            f"Download from https://hbiostat.org/data/repo/support2.csv "
            f"and place in {Path(data_path).parent}/"
        )
    raw_df = pd.read_csv(data_path)
    raw_df.columns = raw_df.columns.str.strip()
    data_hash = compute_data_hash(raw_df)
    print(f"  {len(raw_df):,} rows × {raw_df.shape[1]} columns   SHA256[:16]={data_hash[:16]}")

    # Validate we have the expected dataset (N and shape)
    if len(raw_df) not in (9105, 9104):     # allow ±1 for occasional cleaning
        print(f"  WARNING: expected ~9,105 rows, got {len(raw_df):,}")

    # ── Build audit report on raw data ───────────────────────────────────────
    print("[AUDIT] building three-layer audit report...")
    audit_report = audit_all_columns(raw_df)

    # ── Clean outcomes ───────────────────────────────────────────────────────
    df = raw_df.copy()
    df["d.time"] = pd.to_numeric(df["d.time"], errors="coerce")
    df["death"]  = pd.to_numeric(df["death"],  errors="coerce")
    df = df[df["d.time"] > 0].dropna(subset=["d.time", "death"])
    print(f"  after outcome-cleaning: {len(df):,} rows   event rate {df['death'].mean():.1%}")

    # ── Build feature list ───────────────────────────────────────────────────
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in SKIP_FEATURES and df[c].notna().mean() > 0.4]
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if c not in SKIP_FEATURES and df[c].notna().mean() > 0.4]

    # Preserve dzgroup for subgroup analysis before one-hot destroys it
    dzgroup_orig = df["dzgroup"].copy() if "dzgroup" in df.columns else None

    # One-hot encode (drop-first to avoid collinearity)
    for col in cat_cols:
        d = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
        df = pd.concat([df.drop(columns=[col]), d], axis=1)

    # Add missing indicators (informative-missingness fix)
    for f in MISSING_INDICATOR_FEATURES:
        if f in df.columns:
            df[f"{f}_missing"] = df[f].isna().astype(float)

    # Final feature list
    feat_cols = [c for c in df.columns
                 if c not in SKIP_FEATURES | {"d.time", "death"}
                 and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  final feature count: {len(feat_cols)}")
    print(f"  - includes hday (re-added): {'hday' in feat_cols}")
    print(f"  - includes missing indicators: "
          f"{sum(1 for f in MISSING_INDICATOR_FEATURES if f'{f}_missing' in feat_cols)}/{len(MISSING_INDICATOR_FEATURES)}")

    # ── Train/val/test split ──────────────────────────────────────────────────
    X       = df[feat_cols].copy()
    y_time  = df["d.time"].values.astype(float)
    y_event = df["death"].values.astype(bool)

    idx = np.arange(len(X))
    idx_tv, idx_test = train_test_split(
        idx, test_size=TEST_FRAC, random_state=SEED, stratify=y_event)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=VAL_FRAC/(1-TEST_FRAC), random_state=SEED,
        stratify=y_event[idx_tv])
    print(f"  split: train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")

    # ── Impute + scale (fit on train only — no leakage) ──────────────────────
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr  = pd.DataFrame(sc.fit_transform(imp.fit_transform(X.iloc[idx_train])), columns=feat_cols)
    Xval = pd.DataFrame(sc.transform(imp.transform(X.iloc[idx_val])),           columns=feat_cols)
    Xte  = pd.DataFrame(sc.transform(imp.transform(X.iloc[idx_test])),          columns=feat_cols)

    ytr  = Surv.from_arrays(event=y_event[idx_train], time=y_time[idx_train])
    yval = Surv.from_arrays(event=y_event[idx_val],   time=y_time[idx_val])
    yte  = Surv.from_arrays(event=y_event[idx_test],  time=y_time[idx_test])

    # Attach dzgroup for subgroup analysis
    dzgroup_test = dzgroup_orig.values[idx_test] if dzgroup_orig is not None else None

    return {
        "raw_df":        raw_df,
        "data_hash":     data_hash,
        "audit_report":  audit_report,
        "clean_df":      df,
        "Xtr": Xtr, "Xval": Xval, "Xte": Xte,
        "ytr": ytr, "yval": yval, "yte": yte,
        "feat_cols":     feat_cols,
        "idx_train":     idx_train,
        "idx_val":       idx_val,
        "idx_test":      idx_test,
        "dzgroup_test":  dzgroup_test,
        "imputer":       imp,
        "scaler":        sc,
    }
