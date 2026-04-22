"""
convert_figures_to_tiff.py — Convert PNG figures to PLOS ONE TIFF submission format
======================================================================================
PLOS ONE submission requires:
  - TIFF or EPS format (not PNG)
  - 300-600 DPI for line art + combination figures (we use 300)
  - Files named Fig1.tif, Fig2.tif, ..., FigA1.tif, FigA2.tif
  - LZW compression recommended

Usage:
    python convert_figures_to_tiff.py

Reads PNG figures from the latest run's figures/ folder and writes TIFF files
to plos_submission/figures_tiff/ in the required naming convention.

Mapping PNG name -> PLOS ONE file name:
    fig1_kaplan_meier.png              -> Fig1.tif
    fig2_model_comparison.png          -> Fig2.tif
    fig3_auc_over_time.png             -> Fig3.tif
    fig4_calibration.png               -> Fig4.tif
    fig5a_shap_beeswarm.png            -> Fig5.tif
    fig5b_shap_bar.png                 -> Fig6.tif
    fig6_shap_waterfall_high_risk.png  -> Fig7.tif
    fig6_shap_waterfall_medium_risk.png-> Fig8.tif
    fig6_shap_waterfall_low_risk.png   -> Fig9.tif
    fig8_subgroup_cindex.png           -> Fig10.tif
    fig10_dca.png                      -> Fig11.tif
    figA1_shap_before_after.png        -> Fig12.tif  (Figure A1 in manuscript)
    figA2_audit_overview.png           -> Fig13.tif  (Figure A2 in manuscript)
"""
from __future__ import annotations
from pathlib import Path
import sys
from PIL import Image

# Mapping: PNG name -> (PLOS ONE submission file name, manuscript figure label)
MAPPING = [
    ("fig1_kaplan_meier.png",               "Fig1.tif",  "Figure 1"),
    ("fig2_model_comparison.png",           "Fig2.tif",  "Figure 2"),
    ("fig3_auc_over_time.png",              "Fig3.tif",  "Figure 3"),
    ("fig4_calibration.png",                "Fig4.tif",  "Figure 4"),
    ("fig5a_shap_beeswarm.png",             "Fig5.tif",  "Figure 5"),
    ("fig5b_shap_bar.png",                  "Fig6.tif",  "Figure 6"),
    ("fig6_shap_waterfall_high_risk.png",   "Fig7.tif",  "Figure 7 (high risk)"),
    ("fig6_shap_waterfall_medium_risk.png", "Fig8.tif",  "Figure 8 (medium risk)"),
    ("fig6_shap_waterfall_low_risk.png",    "Fig9.tif",  "Figure 9 (low risk)"),
    ("fig8_subgroup_cindex.png",            "Fig10.tif", "Figure 10"),
    ("fig10_dca.png",                       "Fig11.tif", "Figure 11"),
    ("figA1_shap_before_after.png",         "Fig12.tif", "Figure 12 (A1)"),
    ("figA2_audit_overview.png",            "Fig13.tif", "Figure 13 (A2)"),
]


def find_latest_run() -> Path:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        sys.exit("[ERROR] no runs/ folder found. Run the pipeline first.")
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    if not runs:
        sys.exit("[ERROR] no completed runs found in runs/")
    return runs[-1]


def main():
    run_dir = find_latest_run()
    fig_dir = run_dir / "figures"
    out_dir = Path("plos_submission/figures_tiff")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONVERT] source: {fig_dir}")
    print(f"          target: {out_dir}")
    print()

    for src_name, dst_name, label in MAPPING:
        src = fig_dir / src_name
        dst = out_dir / dst_name
        if not src.exists():
            print(f"  [SKIP] {src_name} not found")
            continue
        img = Image.open(src)
        # Convert mode if needed — TIFF requires RGB or grayscale, not RGBA
        if img.mode in ("RGBA", "P"):
            background = Image.new("RGB", img.size, "white")
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        # Save as TIFF with LZW compression at 300 DPI
        img.save(dst, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
        size_kb = dst.stat().st_size / 1024
        print(f"  [OK] {src_name:40s} -> {dst_name:12s} ({label}, {size_kb:.0f} KB)")

    print()
    print(f"[DONE] TIFF files ready for PLOS ONE submission at: {out_dir}")
    print(f"       Upload each as a separate figure file when submitting.")


if __name__ == "__main__":
    main()
