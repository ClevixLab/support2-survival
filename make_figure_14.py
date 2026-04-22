"""
make_figure_14.py — Generate Figure 14 for the SUPPORT2 audited paper
======================================================================

Figure 14: Leave-one-disease-group-out (LODGO) cross-validation bar chart.

Reads the LODGO output CSV from the most recent LODGO run and produces
a publication-ready figure matching the style of the paper's existing
figures (300 DPI TIFF, sans-serif font, neutral palette with orange
accent for transportability-concerning groups).

Usage
-----
    # Auto-detect most recent lodgo_* run under repo/runs/
    python make_figure_14.py

    # Specify a particular run
    python make_figure_14.py --run-id lodgo_20260422_221500

    # Specify a particular model (default: GBM Survival)
    python make_figure_14.py --model "GBM Survival"

    # Custom output path
    python make_figure_14.py --output figures/Fig14.tif

Output
------
Two files side-by-side in the run's figures/ directory (or --output path):
    Fig14.tif   — 300 DPI TIFF, PLOS-ONE-ready
    Fig14.pdf   — vector version for editing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ─── Import from repo (assumes script lives in repo/) ─────────────────────────
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from config import RUNS_DIR, DPI, PALETTE


# ══════════════════════════════════════════════════════════════════════════════
# Find most recent LODGO run
# ══════════════════════════════════════════════════════════════════════════════
def find_latest_lodgo_run(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("lodgo_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No lodgo_* run found under {runs_dir}. "
            f"Run `python leave_one_disease_out.py` first."
        )
    return candidates[-1]


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
def make_figure(df: pd.DataFrame, model_name: str, primary_baseline: float,
                output_path: Path, dpi: int = 300) -> None:
    """
    Horizontal bar chart: one row per disease group, sorted by C-index descending.
    Error bars = 95% bootstrap CI.
    Colour coding: blue (ΔC ≥ −0.05) vs orange (ΔC < −0.05).
    Horizontal dashed line = primary-split baseline.
    """
    # Filter to requested model and sort
    sub = df[df["model"] == model_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for model '{model_name}' in LODGO results")
    sub = sub.sort_values("cindex", ascending=True).reset_index(drop=True)
    # ^ ascending for horizontal layout: worst at bottom, best at top

    n = len(sub)
    # Error bar endpoints (not lengths) relative to point estimate
    err_lo = sub["cindex"].values - sub["ci_lo"].values
    err_hi = sub["ci_hi"].values - sub["cindex"].values

    # Colour by transportability gap
    thresh = 0.05
    colours = np.where(
        (primary_baseline - sub["cindex"].values) >= thresh,
        "#D85A30",   # orange — concerning drop
        "#185FA5",   # blue — transports acceptably
    )

    # Figure size: double column ≈ 7.5 inches wide for PLOS; height scales with n
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * n + 1.6))

    y = np.arange(n)
    ax.barh(
        y, sub["cindex"].values,
        xerr=[err_lo, err_hi],
        color=colours, edgecolor="black", linewidth=0.7,
        error_kw={"ecolor": "black", "capsize": 4, "elinewidth": 1.2, "capthick": 1.2},
        zorder=3,
    )

    # Reference lines
    ax.axvline(
        primary_baseline, color="#222222", linestyle="--", linewidth=1.3,
        zorder=2, label=f"Primary-split baseline (C = {primary_baseline:.4f})",
    )
    ax.axvline(
        0.5, color="#888888", linestyle="-", linewidth=1.0,
        zorder=1, label="Random (C = 0.5)",
    )

    # Annotations: "C [lo, hi], ΔC" at bar end
    for i, (_, r) in enumerate(sub.iterrows()):
        annot = (
            f"  {r['cindex']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]   "
            f"ΔC = {r['cindex'] - primary_baseline:+.3f}"
        )
        # Place at right edge of bar, slightly offset
        ax.text(
            r["cindex"] + err_hi[i] + 0.005, i, annot,
            va="center", ha="left", fontsize=9, color="#222222",
        )

    # Axis formatting
    ax.set_yticks(y)
    # Show held-out group name and (n = test size)
    group_labels = [
        f"{r['held_out_group']}  (n = {int(r['n_test'])})"
        for _, r in sub.iterrows()
    ]
    ax.set_yticklabels(group_labels, fontsize=10)
    ax.set_xlabel("C-index on held-out disease group (95% CI)", fontsize=11)
    ax.set_xlim(0.48, 0.92)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Title
    mean_c = sub["cindex"].mean()
    mean_delta = mean_c - primary_baseline
    ax.set_title(
        f"Leave-one-disease-group-out cross-validation — {model_name}\n"
        f"Mean C = {mean_c:.3f}   (mean ΔC = {mean_delta:+.3f} vs primary)",
        fontsize=12, pad=12,
    )

    # Legend
    # Create proxy handles for colour meaning
    blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#185FA5", edgecolor="black")
    orange_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#D85A30", edgecolor="black")
    dashed_line = plt.Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.3)
    solid_line = plt.Line2D([0], [0], color="#888888", linestyle="-", linewidth=1.0)
    ax.legend(
        [blue_patch, orange_patch, dashed_line, solid_line],
        [
            "ΔC ≥ −0.05 (transports acceptably)",
            "ΔC < −0.05 (transportability concern)",
            f"Primary-split baseline (C = {primary_baseline:.4f})",
            "Random (C = 0.5)",
        ],
        loc="upper center", fontsize=9, framealpha=0.95, edgecolor="#444444",
        bbox_to_anchor=(0.5, -0.15), ncol=2,
    )

    # Tight layout, save — reserve space at bottom for legend
    plt.subplots_adjust(bottom=0.22, top=0.90, left=0.22, right=0.96)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, format="tiff",
                pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")
    print(f"saved {pdf_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Generate Figure 14 (LODGO bar chart)")
    p.add_argument("--run-id", type=str, default=None,
                   help="LODGO run id (default: most recent lodgo_* under runs/)")
    p.add_argument("--model", type=str, default="GBM Survival",
                   help="Model name to plot (must match a row in lodgo_per_fold_model.csv)")
    p.add_argument("--primary-baseline", type=float, default=0.7052,
                   help="Primary-split C-index reference")
    p.add_argument("--output", type=str, default=None,
                   help="Output TIFF path (default: <run>/figures/Fig14.tif)")
    p.add_argument("--dpi", type=int, default=DPI,
                   help="Output DPI (default from config.py)")
    args = p.parse_args()

    if args.run_id:
        run_dir = RUNS_DIR / args.run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run {run_dir} does not exist")
    else:
        run_dir = find_latest_lodgo_run(RUNS_DIR)
        print(f"using latest LODGO run: {run_dir.name}")

    csv = run_dir / "tables" / "lodgo_per_fold_model.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Did the LODGO run complete?"
        )
    df = pd.read_csv(csv)

    # Drop rows with errors or missing C-index
    df = df[df["cindex"].notna()].copy()
    if df.empty:
        raise ValueError("All rows in lodgo_per_fold_model.csv have NaN C-index")

    output_path = Path(args.output) if args.output else (run_dir / "figures" / "Fig14.tif")
    make_figure(df, args.model, args.primary_baseline, output_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
