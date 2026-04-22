#!/usr/bin/env bash
# reproduce.sh — one-command reproduction of all paper results
# =============================================================
# Tier A (default): reproduce every number, table, and figure from raw data.
#                    Runtime: ~45-60 min on a 6-core CPU laptop.
#                    Use this if you want to verify the pipeline end-to-end.
#
# Tier B (--from-checkpoints): skip training, re-generate figures/tables
#                    from precomputed checkpoints in runs/v1.0_reference/.
#                    Runtime: ~2 min.
#                    Use this if you want to verify plotting/table code only.
#
# Usage:
#   bash reproduce.sh                       # Tier A — full reproduction
#   bash reproduce.sh --from-checkpoints    # Tier B — figures/tables only
#   bash reproduce.sh --fast                # Tier A with reduced bootstrap (testing only)
#
set -e

MODE="full"
FAST_FLAG=""

for arg in "$@"; do
  case "$arg" in
    --from-checkpoints) MODE="from-checkpoints" ;;
    --fast)             FAST_FLAG="--fast" ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'
      exit 0 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

echo "========================================================================"
echo "SUPPORT2 Audited Survival Pipeline — Reproduction Script"
echo "========================================================================"

# Check Python + dependencies
echo "[1/4] Checking Python environment..."
python --version
python -c "import sksurv, sklearn, pandas, numpy, shap, lifelines, matplotlib" 2>/dev/null \
    || { echo "Missing deps. Run: pip install -r requirements.txt"; exit 1; }
echo "    OK"

# Check data file
echo "[2/4] Checking dataset..."
if [ ! -f "data/support2_full.csv" ]; then
  echo "    Dataset not found. Downloading from hbiostat.org..."
  python download_data.py
fi
python -c "
import pandas as pd, hashlib
d = open('data/support2_full.csv','rb').read()
h = hashlib.sha256(d).hexdigest()
print(f'    data SHA-256: {h[:16]}... ({len(d)/1024:.0f} KB)')
"
echo "    OK"

if [ "$MODE" = "from-checkpoints" ]; then
  # ─── Tier B: regenerate from precomputed checkpoints ──────────────────────
  echo "[3/4] Tier B — regenerating figures and tables from runs/v1.0_reference/..."
  if [ ! -d "runs/v1.0_reference/checkpoints" ]; then
    echo "    ERROR: runs/v1.0_reference/checkpoints/ not found."
    echo "    Either switch to Tier A (remove --from-checkpoints) or download the"
    echo "    reference checkpoint bundle from the Zenodo archive for this release."
    exit 1
  fi
  python run_pipeline.py --resume v1.0_reference --stages figures,tables,manifest
  RUN_DIR="runs/v1.0_reference"
else
  # ─── Tier A: full reproduction ────────────────────────────────────────────
  echo "[3/4] Tier A — full pipeline from raw data (~45-60 min)..."
  python run_pipeline.py $FAST_FLAG
  RUN_DIR=$(ls -d runs/20* 2>/dev/null | sort | tail -1)

  echo ""
  echo "[3b/4] Running LODGO cross-validation (~25-35 min)..."
  python leave_one_disease_out.py

  echo ""
  echo "[3c/4] Generating Figure 14 from LODGO results..."
  python make_figure_14.py
fi

# ─── Final summary ────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Reproduction complete."
echo "========================================================================"
echo "Outputs in: $RUN_DIR"
echo ""
echo "Tables:"
ls "$RUN_DIR/tables/" 2>/dev/null | sed 's/^/    /'
echo ""
echo "Figures:"
ls "$RUN_DIR/figures/" 2>/dev/null | sed 's/^/    /'
echo ""
if [ -f "$RUN_DIR/manifest.json" ]; then
  echo "Manifest: $RUN_DIR/manifest.json"
fi
LODGO_DIR=$(ls -d runs/lodgo_* 2>/dev/null | sort | tail -1)
if [ -n "$LODGO_DIR" ]; then
  echo ""
  echo "LODGO results: $LODGO_DIR"
  ls "$LODGO_DIR/tables/" 2>/dev/null | sed 's/^/    /'
fi
echo "========================================================================"
echo "Compare outputs to Tables 2, 3, 4 and Figures 1–14 in the paper."
echo "See docs/REPRODUCE.md for expected values and tolerance notes."
