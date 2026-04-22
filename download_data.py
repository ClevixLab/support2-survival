"""
download_data.py — Automatically download the SUPPORT2 dataset
================================================================
Tries multiple official mirrors. Verifies with SHA-256 against the known
reference hash. Run once before the main pipeline:

    python download_data.py

If the file already exists in data/, this script does nothing.
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET   = DATA_DIR / "support2_full.csv"

# Expected SHA-256 of the canonical support2.csv (9,105 × 47).
# Computed from the reference run used in our paper.
REFERENCE_SHA256 = "7748a3cb0afd8520d7344da5f2deaeed7bfb299d1e3258448e691a4ea72b9b8e"

# Official mirrors, tried in order.
SOURCES = [
    # Harrell Vanderbilt hbiostat repository — the canonical source
    "https://hbiostat.org/data/repo/support2.csv",
    # UCI machine learning repository mirror
    "https://archive.ics.uci.edu/static/public/880/support2.csv",
    # Gensheimer/Narasimhan GitHub mirror (used by the paper)
    "https://raw.githubusercontent.com/MGensheimer/nnet-survival/master/data/support2.csv",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET.exists():
        file_sha = sha256_of(TARGET)
        if file_sha == REFERENCE_SHA256:
            print(f"[OK] {TARGET} already present and SHA-256 matches reference.")
            return 0
        print(f"[WARN] {TARGET} exists but SHA-256 differs from reference:")
        print(f"       expected {REFERENCE_SHA256}")
        print(f"       got      {file_sha}")
        print( "       Delete the file and rerun this script to re-download.")
        return 1

    # Try each mirror
    last_err = None
    for url in SOURCES:
        print(f"[DOWNLOAD] trying {url} ...")
        try:
            df = pd.read_csv(url)
            df.to_csv(TARGET, index=False)
            print(f"           saved to {TARGET}")
            file_sha = sha256_of(TARGET)
            if file_sha == REFERENCE_SHA256:
                print(f"[OK] SHA-256 matches reference ({REFERENCE_SHA256[:16]}...).")
                print(f"     rows x cols: {len(df):,} x {df.shape[1]}")
                return 0
            else:
                print(f"[WARN] SHA-256 differs from reference:")
                print(f"       expected {REFERENCE_SHA256}")
                print(f"       got      {file_sha}")
                print(f"       File saved, but reviewers may see different numbers.")
                return 0
        except Exception as e:
            print(f"           FAILED: {e}")
            last_err = e
            continue

    print()
    print("[ERROR] All mirrors failed. Last error:", last_err)
    print()
    print("Manual fallback:")
    print("  1. Open https://hbiostat.org/data/repo/support2.csv in your browser")
    print("  2. Save the file as data/support2_full.csv in this project folder")
    print("  3. Re-run: python download_data.py   (to verify SHA-256)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
