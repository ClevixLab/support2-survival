"""
figures.py — Paper figures at publication quality (DPI=300, PNG + PDF)
========================================================================
Generates all figures defined in the paper plus new ones for the audit:
  Fig 1  — Kaplan–Meier curves by disease group
  Fig 2  — Model comparison with bootstrap 95% CI error bars
  Fig 3  — Time-dependent AUC
  Fig 4  — Calibration at t = 108d
  Fig 5a — SHAP beeswarm (audited model)
  Fig 5b — SHAP bar chart (audited model)
  Fig 6  — SHAP waterfall (high/mid/low)
  Fig 8  — Subgroup C-index with bootstrap CI
  Fig 10 — Decision Curve Analysis
  Fig A1 (NEW) — SHAP ranking BEFORE vs AFTER audit (shows slos at #2 before)
  Fig A2 (NEW) — Feature audit overview (marginal C-index highlighted)

Every figure is saved at DPI=300 as both PNG and PDF.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lifelines import KaplanMeierFitter

from config import PALETTE, DPI, AUC_EARLY_DAYS, AUC_LATE_DAYS, CAL_TIME_DAYS


def _setup_style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "figure.dpi":         100,         # display DPI
        "savefig.dpi":        DPI,         # output DPI (300 = publication)
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })


def _save(fig, out_dir: Path, name: str):
    """Save figure as both PNG and PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] saved {name}.png/.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Kaplan–Meier by disease group
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_km_by_disease(clean_df, out_dir):
    _setup_style()
    groups = clean_df["dzgroup"].value_counts().head(5).index.tolist() \
             if "dzgroup" in clean_df else []
    fig, ax = plt.subplots(figsize=(10, 6))
    for grp, color in zip(groups, PALETTE):
        sub = clean_df[clean_df["dzgroup"] == grp]
        KaplanMeierFitter().fit(
            sub["d.time"], event_observed=sub["death"], label=grp,
        ).plot_survival_function(ax=ax, ci_show=True, color=color, lw=2)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan–Meier survival curves by disease group — SUPPORT2")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.05)
    _save(fig, out_dir, "fig1_kaplan_meier")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Model C-index comparison + IBS
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_model_comparison(results: dict, out_dir):
    _setup_style()
    # Sort by C-index ascending (for horizontal bar with best at top)
    names = sorted(results.keys(), key=lambda n: results[n]["cindex"])
    vals  = [results[n]["cindex"] for n in names]
    los   = [results[n]["cindex"] - results[n]["ci_lo"] for n in names]
    his   = [results[n]["ci_hi"] - results[n]["cindex"] for n in names]
    ibs   = [results[n].get("ibs", np.nan) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = PALETTE[:len(names)]
    axes[0].barh(names, vals, xerr=[los, his], color=colors, height=0.5,
                 error_kw=dict(ecolor="#444441", elinewidth=1.5, capsize=4))
    axes[0].axvline(0.5, color="gray", ls="--", lw=1)
    axes[0].set_xlim(0.4, 0.78)
    axes[0].set_xlabel("C-index (↑ better, error bars = 95% bootstrap CI)")
    axes[0].set_title("Concordance index")
    for i, v in enumerate(vals):
        axes[0].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

    # IBS panel
    ibs_mask = [not np.isnan(x) for x in ibs]
    if any(ibs_mask):
        ibs_names = [n for n, m in zip(names, ibs_mask) if m]
        ibs_vals  = [x for x, m in zip(ibs, ibs_mask) if m]
        axes[1].barh(ibs_names, ibs_vals, color=PALETTE[:len(ibs_names)], height=0.5)
        axes[1].set_xlabel("Integrated Brier Score (↓ better)")
        axes[1].set_title("Integrated Brier Score")
        for i, v in enumerate(ibs_vals):
            axes[1].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
    fig.suptitle("Model performance — SUPPORT2 test set")
    fig.tight_layout()
    _save(fig, out_dir, "fig2_model_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Time-dependent AUC
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_auc_curve(results: dict, out_dir):
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, r), color in zip(results.items(), PALETTE):
        grid, vals, mean_auc = r.get("auc_curve_time"), r.get("auc_curve_vals"), r.get("auc_curve_mean")
        if grid is None or vals is None: continue
        ax.plot(grid, vals, label=f"{name}  (mean AUC={mean_auc:.3f})", color=color, lw=2)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random (0.5)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("AUC(t)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Time-dependent AUC — SUPPORT2")
    ax.legend(fontsize=9)
    _save(fig, out_dir, "fig3_auc_over_time")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Calibration at t = 108d
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_calibration(results: dict, out_dir):
    _setup_style()
    capable = {n: r for n, r in results.items() if r.get("calibration") is not None}
    if not capable:
        return
    n_col = len(capable)
    fig, axes = plt.subplots(1, n_col, figsize=(4.5 * n_col, 4.5), sharey=True)
    if n_col == 1: axes = [axes]

    from scipy.stats import pearsonr
    for ax, (name, r) in zip(axes, capable.items()):
        cal = r["calibration"]
        if cal is None or len(cal) < 3:
            ax.text(0.5, 0.5, "N/A", ha="center", transform=ax.transAxes)
            ax.set_title(name); continue
        ax.scatter(cal["pred"], cal["obs"], s=55, color=PALETTE[0], zorder=5)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        corr, _ = pearsonr(cal["pred"], cal["obs"])
        ax.text(0.05, 0.9, f"r={corr:.2f}", transform=ax.transAxes, fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted survival")
        ax.set_ylabel("Observed (Kaplan–Meier)")
        ax.set_title(name)
    fig.suptitle(f"Calibration at t={CAL_TIME_DAYS}d — SUPPORT2")
    fig.tight_layout()
    _save(fig, out_dir, "fig4_calibration")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — SHAP beeswarm + bar
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_shap(sv, feat_names, out_dir):
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 7)); plt.sca(ax)
    shap.plots.beeswarm(sv, max_display=15, show=False)
    ax.set_title("SHAP feature importance — GBM Survival (audited)")
    _save(fig, out_dir, "fig5a_shap_beeswarm")

    mean_abs = np.abs(sv.values).mean(axis=0)
    fi = (pd.Series(mean_abs, index=feat_names)
          .sort_values(ascending=True).tail(15))
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors = [PALETTE[0] if v >= fi.quantile(0.7) else "#B4B2A9" for v in fi.values]
    fi.plot.barh(ax=ax2, color=colors, edgecolor="none")
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_title("Global feature importance (top 15) — GBM Survival (audited)")
    _save(fig2, out_dir, "fig5b_shap_bar")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — SHAP waterfall (3 patients)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_shap_waterfall(sv, cases: dict, out_dir):
    _setup_style()
    for label, idx in cases.items():
        fig, ax = plt.subplots(figsize=(9, 6)); plt.sca(ax)
        shap.plots.waterfall(sv[idx], max_display=12, show=False)
        ax.set_title(f"SHAP waterfall — {label.replace('_', ' ')}")
        _save(fig, out_dir, f"fig6_shap_waterfall_{label}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Subgroup C-index with bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_subgroup(sub_df: pd.DataFrame, out_dir):
    _setup_style()
    if len(sub_df) == 0: return
    df = sub_df.sort_values("cindex")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [PALETTE[0] if c >= 0.65 else PALETTE[2] for c in df["cindex"]]
    los = df["cindex"] - df["ci_lo"]
    his = df["ci_hi"] - df["cindex"]
    ax.barh(df["group"], df["cindex"], xerr=[los, his],
            color=colors, height=0.55,
            error_kw=dict(ecolor="#444441", elinewidth=1.3, capsize=3))
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("C-index (bars = 95% bootstrap CI)")
    ax.set_title("Subgroup C-index by disease group — GBM Survival (audited)")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["cindex"] + 0.004, i,
                f"{r['cindex']:.3f}  (n={int(r['n'])}, {r['death_rate']*100:.1f}% dead)",
                va="center", fontsize=8)
    ax.set_xlim(0.4, 0.85)
    _save(fig, out_dir, "fig8_subgroup_cindex")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Decision Curve Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_dca(dca_df: pd.DataFrame, out_dir, model_label: str = "GBM Survival"):
    _setup_style()
    if dca_df is None or len(dca_df) == 0: return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dca_df["threshold"], dca_df["nb_model"],
            color=PALETTE[0], lw=2.2, label=model_label)
    ax.plot(dca_df["threshold"], dca_df["nb_all"],
            color=PALETTE[2], ls="--", lw=1.6, label="Treat all")
    ax.plot(dca_df["threshold"], dca_df["nb_none"],
            color="gray", ls=":", lw=1.4, label="Treat none")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision Curve Analysis — {model_label}")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(dca_df["threshold"].min(), dca_df["threshold"].max())
    _save(fig, out_dir, "fig10_dca")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig A1 — SHAP BEFORE vs AFTER audit (signature audit figure)
# ═══════════════════════════════════════════════════════════════════════════════
def figA1_shap_before_after(comparison_df: pd.DataFrame, out_dir, top_k: int = 12):
    """
    Two-panel figure:
      Left: top 12 features BEFORE audit (with slos) — slos should be #2
      Right: top 12 features AFTER audit — slos absent, distribution of others
    """
    _setup_style()
    before = comparison_df.sort_values("rank_before").head(top_k).copy()
    after  = comparison_df.sort_values("rank_after").head(top_k).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: BEFORE (highlight slos in red)
    colors_b = ["#D85A30" if f == "slos" else "#185FA5" for f in before["feature"]]
    axes[0].barh(before["feature"][::-1], before["importance_before"][::-1],
                 color=colors_b[::-1], height=0.6)
    axes[0].set_title("(A) BEFORE audit: SHAP top-12 features\n"
                      "with slos (post-discharge leakage) included", fontsize=11)
    axes[0].set_xlabel("Mean |SHAP value|")

    # Right: AFTER
    axes[1].barh(after["feature"][::-1], after["importance_after"][::-1],
                 color="#1D9E75", height=0.6)
    axes[1].set_title("(B) AFTER audit: SHAP top-12 features\n"
                      "post-discharge variables excluded, hday included", fontsize=11)
    axes[1].set_xlabel("Mean |SHAP value|")

    fig.suptitle("Figure A1 — SHAP feature importance BEFORE vs AFTER leakage audit",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "figA1_shap_before_after")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig A2 — Feature audit overview (marginal C-index of excluded vs included)
# ═══════════════════════════════════════════════════════════════════════════════
def figA2_audit_overview(audit_df: pd.DataFrame, out_dir):
    """Scatter/box plot showing excluded features have higher marginal C."""
    _setup_style()
    df = audit_df.dropna(subset=["|C - 0.5|+0.5"]).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    # Group by Decision + Layer 1 category for colour
    categories = df["Layer 1 category"].unique()
    y_offset = {c: i for i, c in enumerate(sorted(categories))}

    for cat in sorted(categories):
        sub = df[df["Layer 1 category"] == cat]
        is_skipped = sub["Decision"] == "SKIP"
        color = "#D85A30" if is_skipped.any() else "#1D9E75"
        y = [y_offset[cat]] * len(sub)
        ax.scatter(sub["|C - 0.5|+0.5"], y,
                   color=color, s=60, alpha=0.7, edgecolor="white", linewidth=0.5)
        # Label outliers
        for _, row in sub.iterrows():
            if row["|C - 0.5|+0.5"] > 0.6:
                ax.annotate(row["Feature"],
                            (row["|C - 0.5|+0.5"], y_offset[cat]),
                            fontsize=7, alpha=0.8,
                            xytext=(3, 3), textcoords="offset points")

    ax.set_yticks(list(y_offset.values()))
    ax.set_yticklabels(list(y_offset.keys()))
    ax.axvline(0.6, color="red", ls="--", lw=1, alpha=0.6,
               label="Empirical leakage threshold |C|=0.6")
    ax.axvline(0.5, color="gray", ls=":", lw=1, label="Random")
    ax.set_xlabel("Marginal C-index (directionality removed)")
    ax.set_title("Figure A2 — Three-layer feature audit: marginal prognostic strength\n"
                 "by variable category (higher = more predictive of outcome alone)")
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, out_dir, "figA2_audit_overview")
