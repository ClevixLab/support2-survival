# Reproduction Guide

This document specifies the **exact expected outputs** from `reproduce.sh` /
`reproduce.bat`, with numerical tolerance notes for reviewers and users who
want to verify the pipeline independently.

## Tier A — full reproduction from raw data (~45-60 min)

```bash
bash reproduce.sh           # Linux / macOS
reproduce.bat               # Windows
```

Or, in three manual steps:

```bash
pip install -r requirements.txt
python run_pipeline.py              # ~20-30 min (main pipeline)
python leave_one_disease_out.py     # ~25-35 min (LODGO)
python make_figure_14.py            # ~10 seconds (Figure 14)
```

## Tier B — regenerate figures only (~2 min)

Requires `runs/v1.0_reference/checkpoints/` from the Zenodo archive for this
release. This skips all training and re-runs only the plotting and tabulation
stages — useful for verifying plotting code without waiting for training.

```bash
bash reproduce.sh --from-checkpoints
```

## Expected outputs by paper location

### Main manuscript Table 2 — model comparison

Source CSV: `runs/<RUN_ID>/tables/table2_main_results.csv`

| Model | C-index | CI lo | CI hi | IBS | AUC@18d | AUC@108d |
| :-- | --: | --: | --: | --: | --: | --: |
| Cox PH | 0.6965 | 0.6802 | 0.7147 | 0.1799 | 0.7648 | 0.7397 |
| Cox Elastic Net | 0.6937 | 0.6760 | 0.7119 | 0.1805 | 0.7536 | 0.7333 |
| Random Survival Forest | 0.6795 | 0.6626 | 0.6966 | 0.1872 | 0.7233 | 0.7191 |
| **GBM Survival** | **0.7052** | **0.6861** | **0.7230** | **0.1778** | **0.7785** | **0.7508** |

Tolerance: C-index ±0.0002 across runs with the same seed; ±0.002 across
different sklearn patch versions. IBS ±0.001.

### Main manuscript Table 3 — subgroup C-index

Source CSV: `runs/<RUN_ID>/tables/table3_subgroup.csv`

| Disease Group | N (test) | C-index | CI lo | CI hi | Death rate |
| :-- | --: | --: | --: | --: | --: |
| Coma | 84 | 0.712 | 0.638 | 0.782 | 89.3% |
| Cirrhosis | 81 | 0.696 | 0.650 | 0.773 | 61.7% |
| ARF/MOSF w/Sepsis | 547 | 0.685 | 0.659 | 0.714 | 60.3% |
| Lung Cancer | 140 | 0.649 | 0.593 | 0.690 | 92.1% |
| CHF | 189 | 0.645 | 0.596 | 0.699 | 57.1% |
| MOSF w/Malig | 106 | 0.638 | 0.570 | 0.696 | 92.5% |
| COPD | 148 | 0.609 | 0.557 | 0.657 | 56.8% |
| Colon Cancer | 71 | 0.589 | 0.513 | 0.652 | 78.9% |

Tolerance: C-index ±0.005 per group; CI bounds ±0.01.

### Main manuscript Table 4 — LODGO cross-validation

Source CSV: `runs/lodgo_<RUN_ID>/tables/lodgo_per_fold_model.csv`

| Held-out Group | N (train/test) | Death rate | C-index | CI lo | CI hi | ΔC vs primary | IBS | AUC@108d |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| Coma | 8509 / 596 | 81.2% | 0.679 | 0.653 | 0.703 | −0.026 | 0.119 | 0.770 |
| ARF/MOSF w/Sepsis | 5590 / 3515 | 59.1% | 0.669 | 0.658 | 0.681 | −0.036 | 0.254 | 0.715 |
| COPD | 8138 / 967 | 58.0% | 0.656 | 0.636 | 0.683 | −0.049 | 0.212 | 0.722 |
| Cirrhosis | 8597 / 508 | 64.8% | 0.655 | 0.621 | 0.688 | −0.051 | 0.243 | 0.710 |
| MOSF w/Malig | 8393 / 712 | 90.9% | 0.631 | 0.606 | 0.656 | −0.074 | 0.141 | 0.667 |
| Lung Cancer | 8197 / 908 | 91.7% | 0.614 | 0.592 | 0.638 | −0.091 | 0.129 | 0.659 |
| CHF | 7718 / 1387 | 60.9% | 0.606 | 0.586 | 0.625 | −0.099 | 0.226 | 0.647 |
| Colon Cancer | 8593 / 512 | 83.4% | 0.587 | 0.564 | 0.614 | −0.118 | 0.167 | 0.683 |

Macro-summary: mean C = 0.637, SD = 0.033, mean ΔC = −0.068. All 8 folds have
95% CIs entirely below the primary-split baseline.

Tolerance: C-index ±0.005 per fold. Bootstrap seeds are independent across
folds (SEED + fold_idx × 1000 + model_idx × 7), so cross-fold comparisons
should be numerically stable across runs.

### Figure outputs

All figures are saved at 300 DPI as both PNG (default) and PDF. A separate
TIFF conversion for PLOS ONE submission is available via
`python convert_figures_to_tiff.py`.

| Figure # | Source artefact | Script location |
| :-- | :-- | :-- |
| 1 | `fig1_kaplan_meier.png` | `figures.py` → `fig1_km_by_disease` |
| 2 | `fig2_model_comparison.png` | `figures.py` → `fig2_model_comparison` |
| 3 | `fig3_auc_over_time.png` | `figures.py` → `fig3_auc_curve` |
| 4 | `fig4_calibration.png` | `figures.py` → `fig4_calibration` |
| 5 | `fig5_shap_summary.png` | `figures.py` → `fig5_shap` |
| 6 | `fig6_shap_waterfall.png` | `figures.py` → `fig6_shap_waterfall` |
| 7 | `fig7_features_by_category.png` | generated as part of `figA2_audit_overview` |
| 8 | `fig8_subgroup_cindex.png` | `figures.py` → `fig8_subgroup` |
| 9 | `fig9_missingness_pattern.png` | `figures.py` |
| 10 | `fig10_dca.png` | `figures.py` → `fig10_dca` |
| 11 | `fig11_dca_full_range.png` | (variant of Fig 10) |
| 12 | `figA1_shap_before_after.png` | `figures.py` → `figA1_shap_before_after` |
| 13 | `figA2_audit_overview.png` | `figures.py` → `figA2_audit_overview` |
| **14** | **`Fig14.tif`** (from LODGO run) | **`make_figure_14.py`** |

## Cross-platform notes

- **sklearn version**: 1.3.x is required (pinned in `requirements.txt`).
  sklearn 1.4+ changes default behaviour of `SimpleImputer` for edge cases and
  produces C-indices that differ by ±0.001 from the reported values.
- **sksurv version**: 0.22.2 is the paper's reference version.
- **BLAS backend**: on AMD CPUs using AOCL-BLAS, bootstrap samples may
  differ in the 4th decimal compared to Intel MKL. Point estimates are
  unaffected.
- **GPU XGBoost** (supplementary): produces different risk scores than
  sksurv's GBM by design (different boosting implementation). Its C-index
  is reported in the paper's supplementary table.

## If your numbers differ

1. **Check the manifest.json** in your run folder — compare `data_hash_sha256`,
   `python`, `platform`, and library versions against those reported in
   `docs/REFERENCE_ENV.txt` (or the Zenodo archive README).
2. **Run tests**: `pytest tests/ -v` — all 18 tests should pass. A failure
   indicates a specific invariant was violated.
3. **Check `n_bootstrap`**: fast mode (`--fast`) uses n=100 instead of n=500,
   which widens CIs by ~2×. Do not compare fast-mode CIs to paper values.
4. **Bootstrap differences within tolerance**: point C-indices should match
   to the 4th decimal. If they differ by more than 0.001, the data, split
   seed, or sklearn/sksurv version has changed.

## Troubleshooting

**`ModuleNotFoundError: config`** — you must run from the repo root, not
from `repo/notebooks/` or a parent directory.

**`sksurv import error: numpy dtype size changed`** — binary incompatibility
between installed numpy and sksurv. Fix: `pip install -r requirements.txt`
will install matching versions.

**Pipeline killed halfway through** — rerun with `--resume LATEST`; the
pipeline resumes from the last completed checkpoint.

**Figure 14 script fails with `No lodgo_* run found`** — you must run
`python leave_one_disease_out.py` before `python make_figure_14.py`.
