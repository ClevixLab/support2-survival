# CHANGELOG

## v1.0.0 — 2026-04-22

First public release accompanying the PLOS ONE submission. Complete
re-architecture from earlier development versions with a three-layer
leakage audit as the primary methodological contribution, plus leave-one-
disease-group-out transportability benchmarking and a reproducibility
infrastructure suitable for independent verification and downstream
adoption.

### Added (core)
- **Three-layer leakage audit**
  - Layer 1: principled classification against Knaus 1995 + Harrell hbiostat
  - Layer 2: empirical marginal C-index test with 0.60 threshold
  - Layer 3: informative-missingness test with binary indicator columns
- **Missing-indicator columns** for `glucose`, `bun`, `urine` (Layer 3 fix)
- **`hday` restored** as a legitimate admission-time predictor per Knaus 1995
- **Bootstrap 95% CI** for all per-disease-group subgroup C-indices
- **Stage-based resumable pipeline** with atomic checkpointing and
  input-hash invalidation (`checkpoint.py`)
- **Per-run manifest** with SHA-256 data hash and config snapshot
- **`download_data.py`** script that tries multiple official mirrors and
  verifies SHA-256 against reference
- **XGBoost survival** (optional, GPU-capable) as supplementary model
- **Figure 12** visualising SHAP feature ranking BEFORE vs AFTER audit
- **Figure 13** visualising marginal C-index by audit category
- **Table S1** full 47-variable audit inventory with decisions and rationale
- **Table S2** SHAP rank comparison (before vs after audit)

### Added (LODGO transportability, new in v1.0 final)
- **`leave_one_disease_out.py`**: leave-one-disease-group-out cross-validation
  across all 8 pre-specified disease groups
  - Independent preprocessing (imputation, scaling, one-hot) fit on training
    fold only in every iteration — no leakage from held-out group
  - Bootstrap 95% CIs with independent seeds per (fold, model)
  - Full paper-aligned metric suite per fold: C-index, IBS, AUC(t),
    calibration deciles, DCA, Δ vs primary-split baseline
  - Resumable at (fold × model) granularity
  - Optional 5th column: XGBoost survival:cox (GPU) via `--include-xgboost`
  - Preserved fold-level preprocessed arrays in `temp/<run_id>/` for
    downstream analysis
- **Table 4** LODGO per-fold results with bootstrap CIs
- **Figure 14** (`make_figure_14.py`): per-group LODGO C-index bar chart
  with colour-coded transportability threshold

### Added (reproducibility infrastructure)
- **`reproduce.sh` and `reproduce.bat`**: one-command reproduction with two
  tiers — Tier A (full from raw data, ~45-60 min) and Tier B (figures only
  from precomputed checkpoints, ~2 min)
- **`tests/test_audit.py`**: 18 smoke tests for audit logic, preprocessing
  invariants, and dataset integrity. Run with `pytest tests/ -v`
- **`notebooks/00_quickstart.ipynb`**: 5-minute quickstart reproducing
  Table 2 and Figure 12 with adoption-friendly guidance
- **`docs/REPRODUCE.md`**: step-by-step reproduction guide with expected
  values and numerical tolerance notes
- **`docs/FIGURES.md`**: figure → source map (14 figures)
- **`docs/TABLES.md`**: table → CSV source map (4 main + 2 supplementary)
- **`.zenodo.json`**: metadata for automatic DOI minting on GitHub release

### Changed
- GBM Survival feature count: 43 → 45 (`hday` + 3 missing-indicators;
  2 ADL redundant features removed)
- GBM IBS improved from 0.187 to 0.178 (+5% calibration after audit)
- All bootstrap computations vectorised; ~10× speedup vs threaded version
- All file writes use explicit UTF-8 encoding for Windows compatibility
- Figure output at publication DPI=300, both PNG and PDF
- `requirements.txt` tightened with upper bounds for cross-version
  reproducibility (sklearn < 1.4, sksurv == 0.22.2, numpy < 2.0)

### Removed
- Prognostic Entropy (PE) sections, figures, and analyses. Exploratory audit
  showed that PE computed as the variance of cumulative boosting-stage risk
  correlates substantially with |risk| (Spearman ~0.49 on SUPPORT2), making
  the independence claim in earlier drafts untenable. A bootstrap-ensemble
  alternative is deferred to future work.
- `slos` retained as predictor in earlier versions; now documented as Layer-1
  leakage variable.
- `adlp`, `adls` retained as predictors in earlier versions; now excluded as
  redundant components of the `adlsc` composite.
- `build_manuscript.js`, `build_cover_letter.js`: manuscript-specific
  scripts removed from the public repo release.
