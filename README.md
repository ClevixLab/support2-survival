# SUPPORT2 Audited Survival Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](tests/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)

**A Principled Audit, Corrected Baselines, LODGO Transportability, and Open Reference Pipeline**

Reproducible pipeline accompanying:

> Truong Quynh Hoa, Hoang Dinh Cuong, Luu Duc Trung.
> *Hidden Leakage in SUPPORT2 Survival Benchmarks: A Principled Audit,
> Corrected Baselines, and Open Reference Pipeline.*
> PLOS ONE, 2026.

**Code version:** v1.0.0
**Data version:** SUPPORT2, 9,105 patients × 47 variables
(SHA-256 `7748a3cb0afd8520d7344da5f2deaeed7bfb299d1e3258448e691a4ea72b9b8e`)

---

## TL;DR

```bash
git clone https://github.com/ClevixLab/support2-survival.git
cd support2-survival
pip install -r requirements.txt
bash reproduce.sh                  # or reproduce.bat on Windows
```

One command reproduces **every table, figure, and reported number in the paper**
(Tables 2, 3, 4 and Figures 1–14). Runtime ~45–60 minutes on a 6-core CPU.

For a **5-minute quickstart** that reproduces Table 2 and Figure 12 only:

```bash
jupyter notebook notebooks/00_quickstart.ipynb
```

---

## What this repository provides

### Four contributions of the paper

1. **Three-layer leakage audit** — for all 47 SUPPORT2 variables.
   - Layer 1: principled classification against Knaus 1995 + Harrell hbiostat
   - Layer 2: empirical marginal-C-index test (|C| > 0.60 threshold)
   - Layer 3: informative-missingness test + binary indicator columns
   - Result: 17 principled exclusions + 2 outcome targets + 28 retained
     admission-time features + 3 missing indicators = 45 final features
2. **Corrected baselines** — four bootstrap-validated survival models
   (Cox PH, Cox Elastic Net, Random Survival Forest, Gradient Boosting Survival)
   with null-shifted ΔC pairwise tests.
   GBM Survival achieves **C = 0.7052 [0.6861, 0.7230]**, IBS 0.178.
3. **Open reference pipeline** — stage-based checkpointing, resumable after
   any interruption, per-run manifests (data SHA-256, config snapshot,
   completed stages) for reproducibility audits.
4. **LODGO transportability benchmark** — leave-one-disease-group-out
   cross-validation across all 8 pre-specified diagnostic groups.
   Mean C = 0.637, range 0.587–0.679, mean ΔC = −0.068 vs the primary-split
   baseline. To our knowledge the first such benchmark on SUPPORT2 with
   bootstrap CIs across all 8 groups.

### For downstream users

- **Audited feature set** (Table S1) — drop-in replacement for any survival
  ML study on SUPPORT2. Start from this list rather than raw 47 variables.
- **Quickstart notebook** (`notebooks/00_quickstart.ipynb`) — 5 minutes,
  no setup beyond `pip install`, reproduces key paper claims.
- **Three-layer audit protocol** — applicable to MIMIC-IV, eICU,
  AmsterdamUMCdb, HiRID, or any publicly released ICU dataset where
  post-admission and outcome-fitted variables are mixed with admission-time
  features (see `docs/REPRODUCE.md` for the 4-step adoption protocol).

---

## Reproduction

### Tier A — full reproduction from raw data (~45-60 min)

```bash
bash reproduce.sh            # Linux / macOS
reproduce.bat                # Windows
```

This runs:
1. `python run_pipeline.py` — main pipeline (Tables 2, 3; Figures 1–13; Table S1, S2)
2. `python leave_one_disease_out.py` — LODGO cross-validation (Table 4)
3. `python make_figure_14.py` — Figure 14 bar chart

Outputs are time-stamped in `runs/YYYYMMDD_HHMMSS/` (main) and
`runs/lodgo_YYYYMMDD_HHMMSS/` (LODGO), each with `figures/`, `tables/`,
`logs/`, `checkpoints/`, and `manifest.json`.

### Tier B — regenerate figures from precomputed checkpoints (~2 min)

```bash
bash reproduce.sh --from-checkpoints
```

Requires `runs/v1.0_reference/checkpoints/` which is bundled in the Zenodo
archive for this release. Use this if you want to verify the plotting and
tabulation code without re-training.

### Interrupted? Resume with one command

```bash
python run_pipeline.py --resume LATEST
python leave_one_disease_out.py                   # also auto-resumes
```

---

## Quickstart notebook

```bash
jupyter notebook notebooks/00_quickstart.ipynb
```

The notebook is 19 cells, ~3 minutes runtime, and covers:

1. Load SUPPORT2 raw data (9,105 patients).
2. Apply the three-layer audit → produce `FEATURES_AUDITED` list (Table S1).
3. Train Gradient Boosting Survival on the canonical 70/15/15 split.
4. Evaluate on held-out test set → reproduce Table 2 numbers.
5. Compute SHAP before vs after audit → reproduce Figure 12.
6. How-to-adopt guidance for other datasets.

---

## Tests

```bash
pytest tests/ -v
```

18 smoke tests verify:
- Layer 1 excludes Harrell's 8 variables (aps, sps, surv2m, surv6m, prg2m, prg6m, dnr, dnrday)
- Layer 1 retains `hday` (Knaus 1995)
- Layer 2 empirical claims hold (OUTCOME/DERIVED-category exclusions have |C| > 0.60)
- Layer 2 admissible features have |C| < 0.70
- Layer 3 missing indicators are correctly binary
- Preprocessing fits on training only (train means ≈ 0, test means drift from 0)
- No outcome leakage into feature matrix
- Train/test column alignment
- Cohort size = 9,105 with ~68% event rate across 8 disease groups

Tests complete in ~45 seconds and do not require GPU or training.

---

## Project layout

```
support2-survival/
├── data/
│   └── support2_full.csv        # bundled; 9,105 × 47; SHA-256 verified
├── notebooks/
│   └── 00_quickstart.ipynb      # 5-minute adoption notebook
├── tests/
│   └── test_audit.py            # 18 pytest smoke tests
├── docs/
│   ├── REPRODUCE.md             # step-by-step reproduction with expected values
│   ├── FIGURES.md               # figure → source map (Figures 1–14)
│   └── TABLES.md                # table → CSV source map
├── config.py                    # single source of truth: paths, hyperparameters, features
├── checkpoint.py                # atomic checkpointing with input-hash invalidation
├── data.py                      # three-layer audit + preprocessing
├── models.py                    # Cox PH / Cox EN / RSF / GBM (+ optional XGBoost)
├── evaluation.py                # C-index + CI, IBS, AUC, calibration, DCA, ΔC, subgroup
├── shap_analysis.py             # SHAP with reproducible seed + before/after audit
├── figures.py                   # Figures 1–13 at 300 DPI (PNG + PDF)
├── gpu.py                       # optional GPU detection for XGBoost
├── run_pipeline.py              # main orchestrator — 10 resumable stages
├── leave_one_disease_out.py     # LODGO cross-validation (Table 4, Figure 14)
├── make_figure_14.py            # Figure 14 bar chart generator
├── download_data.py             # fallback data-download with SHA-256 verification
├── convert_figures_to_tiff.py   # PNG → TIFF 300 DPI for PLOS submission
├── reproduce.sh                 # one-command reproduction (Linux / macOS)
├── reproduce.bat                # one-command reproduction (Windows)
├── requirements.txt             # pinned dependencies
├── .zenodo.json                 # Zenodo metadata for auto-mint DOI
├── CITATION.cff                 # citation metadata
├── CHANGELOG.md                 # v1.0.0 release notes
├── LICENSE                      # MIT
└── README.md                    # this file
```

---

## Hardware requirements

- **Minimum**: Python 3.11+, 8 GB RAM, 2 GB free disk. CPU-only.
- **Recommended**: 6+ CPU cores (for parallelised bootstrap). A GPU is NOT
  required — sksurv models are CPU-only. Optional supplementary XGBoost
  survival column (`--include-xgboost`) uses GPU if available.

Runtimes observed on a 6-core laptop CPU:

| Task | Runtime |
| :-- | --: |
| `run_pipeline.py` (all stages, with SHAP) | 20-30 min |
| `run_pipeline.py --no-shap` | 8-12 min |
| `leave_one_disease_out.py` (4 models × 8 folds × 500 bootstrap) | 25-35 min |
| `leave_one_disease_out.py --include-xgboost` | +3-5 min |
| `leave_one_disease_out.py --models gbm-only --n-bootstrap 100` (smoke test) | 10-15 min |
| `make_figure_14.py` | 10 seconds |
| Quickstart notebook | 3 min |
| Smoke tests (`pytest tests/`) | 45 seconds |

---

## Citation

If you use this code, the audited feature set, or the LODGO benchmark in
your research, please cite both the paper and this software release:

**Paper:**
```
Truong QH, Hoang DC, Luu DT. Hidden Leakage in SUPPORT2 Survival Benchmarks:
A Principled Audit, Corrected Baselines, and Open Reference Pipeline.
PLOS ONE. 2026;XX(X):e0XXXXXX. doi:10.1371/journal.pone.XXXXXXX
```

**Software (Zenodo archive):**
```
Truong QH, Hoang DC, Luu DT. SUPPORT2 Audited Survival Benchmark (v1.0.0).
Zenodo. 2026. doi:10.5281/zenodo.XXXXXXX
```

BibTeX entries are available in `CITATION.cff`.

---

## License

Code: MIT License (see `LICENSE`).
Data: SUPPORT2 is distributed under the terms of Harrell's hbiostat data use
policy; see [https://hbiostat.org/data/](https://hbiostat.org/data/).

---

## Acknowledgements

The SUPPORT2 dataset is curated by Frank E. Harrell Jr. at Vanderbilt
University Medical Center (hbiostat.org) and was originally collected by
the SUPPORT investigators (Knaus et al., 1995).

---

## Questions / issues

Please open an issue at
[https://github.com/ClevixLab/support2-survival/issues](https://github.com/ClevixLab/support2-survival/issues)
or email the corresponding author (hoa@clevix.vn).
