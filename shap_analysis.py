"""
shap_analysis.py — SHAP computations for GBM Survival
========================================================
Provides:
  - compute_shap_values(): main SHAP values for the audited model
  - compute_shap_with_slos(): rerun SHAP with slos included (proof of leakage)
    → shows slos ranks #2 before audit, confirming audit impact
  - select_waterfall_cases(): pick 3 representative patients (high/mid/low risk)

All computations are seeded for reproducibility.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap

from config import SEED, SHAP_N_BACKGROUND, SHAP_N_EXPLAIN, SHAP_MAX_EVALS_MULT


def compute_shap_values(gbm_model, Xte: pd.DataFrame,
                        n_background: int = SHAP_N_BACKGROUND,
                        n_explain: int = SHAP_N_EXPLAIN,
                        seed: int = SEED):
    """
    Compute SHAP values via PermutationExplainer.

    Seeded via numpy default_rng for deterministic background/explain selection.
    """
    rng = np.random.default_rng(seed)
    n = len(Xte)
    bg_idx = rng.choice(n, size=min(n_background, n), replace=False)
    ex_idx = rng.choice(n, size=min(n_explain, n),    replace=False)

    background = Xte.iloc[bg_idx]
    explain_on = Xte.iloc[ex_idx]

    max_evals = max(2 * Xte.shape[1] + 1, SHAP_MAX_EVALS_MULT * Xte.shape[1])

    explainer = shap.PermutationExplainer(
        gbm_model.predict,
        background,
        max_evals=max_evals,
        seed=seed,
    )
    sv = explainer(explain_on)
    # Attach feature names if not already present
    if not hasattr(sv, "feature_names") or sv.feature_names is None:
        sv.feature_names = list(Xte.columns)
    return sv, ex_idx


def shap_importance_ranking(sv) -> pd.DataFrame:
    """Mean |SHAP| per feature, sorted descending."""
    mean_abs = np.abs(sv.values).mean(axis=0)
    names = sv.feature_names or [f"f{i}" for i in range(sv.values.shape[1])]
    df = (pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))
    df["rank"] = df.index + 1
    return df


def select_waterfall_cases(sv) -> dict:
    """Return indices (within sv) of 3 representative patients: high / mid / low."""
    row_sum = sv.values.sum(axis=1)
    return {
        "high_risk":   int(np.argmax(row_sum)),
        "low_risk":    int(np.argmin(row_sum)),
        "medium_risk": int(np.argsort(np.abs(row_sum - np.median(row_sum)))[0]),
    }


def compute_shap_with_slos_included(Xtr_with_slos, ytr, Xte_with_slos,
                                    gbm_params: dict,
                                    n_background: int = SHAP_N_BACKGROUND,
                                    n_explain: int = SHAP_N_EXPLAIN,
                                    seed: int = SEED):
    """
    Refit GBM WITH slos included, compute SHAP — proves slos is top feature.
    This is used for the paper's "before audit" figure demonstrating the leakage.

    Parameters
    ----------
    Xtr_with_slos : DataFrame — training matrix INCLUDING the 'slos' column
    ytr           : Surv structured array
    Xte_with_slos : DataFrame — test matrix INCLUDING 'slos'

    Returns
    -------
    dict with: model, sv (SHAP values), importance (ranking DataFrame)
    """
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    gbm_leaky = GradientBoostingSurvivalAnalysis(**gbm_params).fit(Xtr_with_slos, ytr)
    sv, _ = compute_shap_values(gbm_leaky, Xte_with_slos, n_background, n_explain, seed)
    ranking = shap_importance_ranking(sv)
    return {"model": gbm_leaky, "sv": sv, "importance": ranking}


def build_before_after_comparison(sv_audited, sv_leaky) -> pd.DataFrame:
    """
    Side-by-side ranking of top features BEFORE audit (with slos)
    vs AFTER audit (without slos).
    """
    before = shap_importance_ranking(sv_leaky).rename(
        columns={"rank": "rank_before", "mean_abs_shap": "importance_before"})
    after = shap_importance_ranking(sv_audited).rename(
        columns={"rank": "rank_after", "mean_abs_shap": "importance_after"})
    merged = before.merge(after, on="feature", how="outer")
    merged["rank_delta"] = merged["rank_before"] - merged["rank_after"]
    return merged.sort_values("rank_before").reset_index(drop=True)
