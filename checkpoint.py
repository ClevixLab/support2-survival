"""
checkpoint.py — Stage-based checkpointing for resumable pipeline
==================================================================
Every stage of the pipeline writes a checkpoint. If the pipeline crashes
or is killed (Ctrl+C), rerun with the same RUN_ID to resume from the last
completed stage — no stage is rerun unnecessarily.

Usage inside a stage function:
    @checkpoint("preproc", inputs=["raw_df"], outputs=["Xtr","Xte","ytr","yte"])
    def preprocess(raw_df):
        ...
        return {"Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte}

Or imperatively:
    ck = Checkpointer(run_dir)
    if ck.exists("stage3_models"):
        models = ck.load("stage3_models")
    else:
        models = train_models(Xtr, ytr)
        ck.save("stage3_models", models)

Every checkpoint stores:
  - data (pickled)
  - a hash of its inputs (so stale checkpoints are detected automatically)
  - timestamp
  - pipeline version tag

CRITICAL: Checkpointed pandas/numpy objects preserve dtypes exactly.
Survival arrays (sksurv.util.Surv structured arrays) pickle correctly.
"""
from __future__ import annotations
from pathlib import Path
import pickle
import hashlib
import json
import time
from typing import Any
import pandas as pd
import numpy as np


# Version tag — bump to force checkpoint invalidation across code changes
PIPELINE_VERSION = "audited_v1.0"


def _hash_bytes(obj: Any) -> str:
    """Stable hash of pickled object — used to detect stale checkpoints."""
    try:
        h = hashlib.sha256()
        if isinstance(obj, pd.DataFrame):
            # hash-by-row is fast and order-sensitive
            h.update(pd.util.hash_pandas_object(obj, index=True).values.tobytes())
            h.update(str(tuple(obj.columns)).encode())
            h.update(str(obj.shape).encode())
        elif isinstance(obj, np.ndarray):
            h.update(obj.tobytes() if obj.flags.c_contiguous else obj.copy().tobytes())
            h.update(str(obj.shape).encode())
            h.update(str(obj.dtype).encode())
        elif isinstance(obj, (str, int, float, bool, type(None))):
            h.update(str(obj).encode())
        else:
            # fall back to pickle hash
            h.update(pickle.dumps(obj, protocol=4))
        return h.hexdigest()[:16]
    except Exception:
        return "unhashable"


class Checkpointer:
    """Manages per-stage checkpoints with input-hash validation."""

    def __init__(self, ckpt_dir: Path, verbose: bool = True):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.index_file = self.dir / "_index.json"
        self._load_index()

    # ─────────────────────────────────────────────────────────────────────────
    def _load_index(self) -> None:
        if self.index_file.exists():
            with open(self.index_file) as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def _save_index(self) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    def exists(self, tag: str, input_fingerprint: str | None = None) -> bool:
        """Return True iff checkpoint exists AND its input hash matches."""
        f = self.dir / f"{tag}.pkl"
        if not f.exists():
            return False
        if tag not in self.index:
            return False
        entry = self.index[tag]
        if entry.get("version") != PIPELINE_VERSION:
            if self.verbose:
                print(f"  [ckpt] {tag}: stale (version mismatch) — will rebuild")
            return False
        if input_fingerprint is not None and entry.get("input_fp") != input_fingerprint:
            if self.verbose:
                print(f"  [ckpt] {tag}: stale (input changed) — will rebuild")
            return False
        return True

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, tag: str, obj: Any, input_fingerprint: str | None = None) -> None:
        f = self.dir / f"{tag}.pkl"
        tmp = f.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as h:
            pickle.dump(obj, h, protocol=4)
        tmp.replace(f)    # atomic on POSIX and Windows
        self.index[tag] = {
            "version":  PIPELINE_VERSION,
            "input_fp": input_fingerprint,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "size_kb":  round(f.stat().st_size / 1024, 1),
        }
        self._save_index()
        if self.verbose:
            print(f"  [ckpt] saved  {tag}.pkl  ({self.index[tag]['size_kb']} KB)")

    # ─────────────────────────────────────────────────────────────────────────
    def load(self, tag: str) -> Any:
        f = self.dir / f"{tag}.pkl"
        with open(f, "rb") as h:
            obj = pickle.load(h)
        if self.verbose:
            print(f"  [ckpt] loaded {tag}.pkl  (resumed)")
        return obj

    # ─────────────────────────────────────────────────────────────────────────
    def get_or_compute(self, tag: str, compute_fn, input_fingerprint: str | None = None):
        """
        Convenience wrapper: load checkpoint if valid, else run compute_fn and save.
        """
        if self.exists(tag, input_fingerprint):
            return self.load(tag)
        t0 = time.time()
        result = compute_fn()
        elapsed = time.time() - t0
        self.save(tag, result, input_fingerprint)
        if self.verbose:
            print(f"  [ckpt] computed {tag} in {elapsed:.1f}s")
        return result


def fingerprint(*objs) -> str:
    """Combined fingerprint of multiple objects (for checkpoint invalidation)."""
    h = hashlib.sha256()
    for o in objs:
        h.update(_hash_bytes(o).encode())
    return h.hexdigest()[:16]
