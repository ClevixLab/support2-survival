"""
run_pipeline.py — Main orchestrator for the SUPPORT2 audited pipeline
=======================================================================
Stage-based, resumable, fully logged. Each stage checkpointed so that if the
pipeline is interrupted it resumes from the last completed stage on rerun.

STAGES:
  0  setup              — verify data + env, create run folder
  1  preprocess         — load + audit + split + scale/impute
  2  train_models       — Cox PH / Cox EN / RSF / GBM  (+ XGB-GPU if available)
  3  evaluate           — per-model: C + CI, IBS, AUC, calibration, DCA
  4  delong_pairwise    — null-shifted ΔC tests vs GBM
  5  subgroup           — per-disease-group C-index with bootstrap CI
  6  shap_audited       — SHAP on the audited GBM
  7  shap_before_audit  — refit GBM with slos, compute SHAP  (proves audit impact)
  8  figures            — all paper + audit figures (PNG + PDF, DPI=300)
  9  tables             — all paper + audit tables (CSV)
 10  manifest           — write manifest.json with config + data hash

USAGE:
  python run_pipeline.py                 # start fresh in new time-stamped folder
  python run_pipeline.py --resume LATEST # resume the most recent unfinished run
  python run_pipeline.py --resume 20260422_103045  # resume specific run
  python run_pipeline.py --stages 1,2,3  # only run these stages
  python run_pipeline.py --no-shap       # skip SHAP (saves ~10 min)
  python run_pipeline.py --no-xgb        # skip XGBoost even if GPU available

Ctrl+C is safe: the pipeline will resume from the last completed stage.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Local imports ─────────────────────────────────────────────────────────────
import config as CFG
from checkpoint import Checkpointer, fingerprint, PIPELINE_VERSION
from data import load_and_preprocess
from models import train_sksurv_models, train_xgb_supplementary
from evaluation import (
    evaluate_model_full, null_shifted_delta_c,
    subgroup_cindex, cindex_with_ci,
)
from shap_analysis import (
    compute_shap_values, shap_importance_ranking,
    select_waterfall_cases, compute_shap_with_slos_included,
    build_before_after_comparison,
)
import figures as FIG
from gpu import detect_gpu


# ══════════════════════════════════════════════════════════════════════════════
# Stage definitions — declared up top so --stages can reference by name/number
# ══════════════════════════════════════════════════════════════════════════════
STAGES = [
    "setup",
    "preprocess",
    "train_models",
    "evaluate",
    "delong_pairwise",
    "subgroup",
    "shap_audited",
    "shap_before_audit",
    "figures",
    "tables",
    "manifest",
]


# ══════════════════════════════════════════════════════════════════════════════
# Logger
# ══════════════════════════════════════════════════════════════════════════════
class TeeLogger:
    """Write stdout to both terminal and log file."""
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8", buffering=1)
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg)
    def flush(self):
        self.terminal.flush(); self.log.flush()


def log_stage(stage_name: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'═'*78}\n[{ts}] STAGE: {stage_name}\n{'═'*78}")


# ══════════════════════════════════════════════════════════════════════════════
# Resume logic
# ══════════════════════════════════════════════════════════════════════════════
def resolve_run_dir(resume: str | None) -> Path:
    if resume is None:
        run_dir = CFG.RUN_DIR
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    if resume.upper() == "LATEST":
        runs = sorted([d for d in CFG.RUNS_DIR.iterdir() if d.is_dir()])
        if not runs:
            print(f"[resume] no runs found in {CFG.RUNS_DIR} — starting new run")
            run_dir = CFG.RUN_DIR
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        run_dir = runs[-1]
        print(f"[resume] latest run: {run_dir.name}")
        return run_dir
    run_dir = CFG.RUNS_DIR / resume
    if not run_dir.exists():
        raise FileNotFoundError(f"run folder not found: {run_dir}")
    print(f"[resume] resuming: {run_dir.name}")
    return run_dir


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="SUPPORT2 audited survival pipeline (resumable)")
    ap.add_argument("--resume", default=None,
                    help="Resume an existing run (RUN_ID or 'LATEST')")
    ap.add_argument("--stages", default=None,
                    help="Comma-separated stage names or 1-based numbers")
    ap.add_argument("--no-shap",  action="store_true",  help="Skip SHAP stages")
    ap.add_argument("--no-xgb",   action="store_true",  help="Skip XGBoost")
    ap.add_argument("--fast",     action="store_true",
                    help="Reduce bootstrap counts (for testing only)")
    args = ap.parse_args()

    # Resolve run folder and create sub-dirs
    run_dir  = resolve_run_dir(args.resume)
    ckpt_dir = run_dir / "checkpoints";  ckpt_dir.mkdir(exist_ok=True)
    fig_dir  = run_dir / "figures";      fig_dir.mkdir(exist_ok=True)
    tbl_dir  = run_dir / "tables";       tbl_dir.mkdir(exist_ok=True)
    log_dir  = run_dir / "logs";         log_dir.mkdir(exist_ok=True)
    CFG.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    # Tee stdout to log file
    sys.stdout = TeeLogger(log_file)

    print(f"\n{'═'*78}\nSUPPORT2 Audited Survival Pipeline  — v{PIPELINE_VERSION}")
    print(f"Run folder     : {run_dir}")
    print(f"Log file       : {log_file}")
    print(f"Data CSV       : {CFG.DATA_CSV}")
    print(f"Resume mode    : {args.resume or 'NEW RUN'}")
    print(f"Python         : {sys.version.split()[0]}")
    gpu = detect_gpu()
    print(f"GPU            : {gpu['name'] if gpu['available'] else 'none (CPU only)'}")
    print(f"Current time   : {datetime.now().isoformat(timespec='seconds')}")
    print(f"{'═'*78}")

    ck = Checkpointer(ckpt_dir, verbose=True)

    # Which stages to run
    selected = STAGES.copy()
    if args.stages:
        tokens = [t.strip() for t in args.stages.split(",")]
        selected = []
        for t in tokens:
            if t.isdigit():
                i = int(t) - 1
                if 0 <= i < len(STAGES): selected.append(STAGES[i])
            elif t in STAGES:
                selected.append(t)
    print(f"Stages selected: {selected}\n")

    # For --fast mode
    if args.fast:
        CFG.N_BOOTSTRAP = 100
        CFG.N_DELONG = 200

    # Helper: load checkpoint if present, else return None (graceful skip)
    def _load_if_exists(tag):
        return ck.load(tag) if ck.exists(tag) else None

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 1 — preprocess
    # ═════════════════════════════════════════════════════════════════════════
    if "preprocess" in selected:
        log_stage("1 / preprocess (audit + split + scale/impute)")
        data = ck.get_or_compute("s1_preprocess", load_and_preprocess)
    else:
        data = _load_if_exists("s1_preprocess")

    if data is None:
        print("[SKIP] no preprocess checkpoint — downstream stages will be skipped")
        print("       to bootstrap, run: python run_pipeline.py --stages preprocess")
        sys.exit(0)

    # Save audit report as CSV (idempotent — write every run)
    (tbl_dir / "table_S1_feature_audit.csv").write_text(
        data["audit_report"].to_csv(index=False), encoding="utf-8")
    print(f"  wrote Table S1: tables/table_S1_feature_audit.csv")

    data_fp = data["data_hash"][:16]

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2 — train models
    # ═════════════════════════════════════════════════════════════════════════
    if "train_models" in selected:
        log_stage("2 / train_models")
        def _train():
            out = train_sksurv_models(data["Xtr"], data["ytr"])
            if not args.no_xgb:
                xgb_result = train_xgb_supplementary(data["Xtr"], data["ytr"], data["Xte"])
                if xgb_result is not None:
                    out["xgb"] = xgb_result
            return out
        trained = ck.get_or_compute("s2_trained_models", _train,
                                    input_fingerprint=data_fp)
    else:
        trained = _load_if_exists("s2_trained_models")

    models = trained["models"] if trained is not None else None

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 3 — evaluate each model
    # ═════════════════════════════════════════════════════════════════════════
    if "evaluate" in selected and models is not None:
        log_stage("3 / evaluate (per-model C + CI, IBS, AUC, calibration, DCA)")
        def _eval_all():
            out = {}
            for name, m in models.items():
                out[name] = evaluate_model_full(name, m, data["Xte"], data["ytr"], data["yte"])
            if "xgb" in trained:
                class _RiskWrapper:
                    def __init__(self, risk): self._risk = risk
                    def predict(self, X=None): return self._risk
                x_wrapped = _RiskWrapper(trained["xgb"]["risk_test"])
                out["XGBoost (supplementary)"] = evaluate_model_full(
                    "XGBoost (supplementary)", x_wrapped,
                    data["Xte"], data["ytr"], data["yte"])
            return out
        results = ck.get_or_compute("s3_evaluations", _eval_all,
                                    input_fingerprint=data_fp)
    else:
        results = _load_if_exists("s3_evaluations")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 4 — null-shifted ΔC vs GBM
    # ═════════════════════════════════════════════════════════════════════════
    if "delong_pairwise" in selected and results is not None:
        log_stage("4 / delong_pairwise (null-shifted ΔC tests vs GBM)")
        def _delong():
            ref = "GBM Survival"
            rows = []
            risk_ref = results[ref]["risk"]
            for name, r in results.items():
                if name == ref: continue
                t = null_shifted_delta_c(
                    data["yte"]["event"], data["yte"]["time"],
                    risk_ref, r["risk"], n_boot=CFG.N_DELONG)
                if t is None: continue
                t["comparison"] = f"GBM vs {name}"
                rows.append(t)
            return pd.DataFrame(rows)
        delong_df = ck.get_or_compute("s4_delong", _delong, input_fingerprint=data_fp)
        print(delong_df.to_string(index=False))
    else:
        delong_df = _load_if_exists("s4_delong")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 5 — subgroup
    # ═════════════════════════════════════════════════════════════════════════
    if "subgroup" in selected and results is not None:
        log_stage("5 / subgroup (C-index per disease group with CI)")
        def _sub():
            risk = results["GBM Survival"]["risk"]
            return subgroup_cindex(
                risk, data["yte"]["event"], data["yte"]["time"],
                data["dzgroup_test"], n_boot=CFG.N_BOOTSTRAP)
        sub_df = ck.get_or_compute("s5_subgroup", _sub, input_fingerprint=data_fp)
        print(sub_df.to_string(index=False))
    else:
        sub_df = _load_if_exists("s5_subgroup")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 6 — SHAP audited
    # ═════════════════════════════════════════════════════════════════════════
    shap_audited = None
    if not args.no_shap and "shap_audited" in selected and models is not None:
        log_stage("6 / shap_audited (SHAP on audited GBM)")
        def _shap_audited():
            sv, ex_idx = compute_shap_values(models["GBM Survival"], data["Xte"])
            ranking = shap_importance_ranking(sv)
            cases = select_waterfall_cases(sv)
            return {"sv": sv, "ex_idx": ex_idx, "ranking": ranking, "cases": cases}
        shap_audited = ck.get_or_compute("s6_shap_audited", _shap_audited,
                                         input_fingerprint=data_fp)
        print("Top 10 SHAP features (audited):")
        print(shap_audited["ranking"].head(10).to_string(index=False))
    else:
        shap_audited = _load_if_exists("s6_shap_audited")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 7 — SHAP BEFORE audit (proof of leakage)
    # ═════════════════════════════════════════════════════════════════════════
    shap_leaky = None
    if (not args.no_shap and "shap_before_audit" in selected
            and data is not None and shap_audited is not None):
        log_stage("7 / shap_before_audit (refit with slos → prove leakage)")
        def _shap_leaky():
            # Re-build feature matrix WITH slos included
            raw = data["raw_df"]
            df_leaky = raw.copy()
            df_leaky["d.time"] = pd.to_numeric(df_leaky["d.time"], errors="coerce")
            df_leaky["death"]  = pd.to_numeric(df_leaky["death"],  errors="coerce")
            df_leaky = df_leaky[df_leaky["d.time"] > 0].dropna(subset=["d.time", "death"])
            # Same skip list EXCEPT keep slos
            skip_leaky = set(CFG.SKIP_FEATURES) - {"slos"}
            num_cols = [c for c in df_leaky.select_dtypes(include=[np.number]).columns
                        if c not in skip_leaky and df_leaky[c].notna().mean() > 0.4]
            cat_cols = [c for c in df_leaky.select_dtypes(include=["object","category"]).columns
                        if c not in skip_leaky and df_leaky[c].notna().mean() > 0.4]
            for col in cat_cols:
                d = pd.get_dummies(df_leaky[col], prefix=col, drop_first=True, dtype=float)
                df_leaky = pd.concat([df_leaky.drop(columns=[col]), d], axis=1)
            for f in CFG.MISSING_INDICATOR_FEATURES:
                if f in df_leaky.columns:
                    df_leaky[f"{f}_missing"] = df_leaky[f].isna().astype(float)
            feat_leaky = [c for c in df_leaky.columns
                          if c not in skip_leaky | {"d.time","death"}
                          and pd.api.types.is_numeric_dtype(df_leaky[c])]

            X = df_leaky[feat_leaky]
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            imp, sc = SimpleImputer(strategy="median"), StandardScaler()
            Xtr_l = pd.DataFrame(sc.fit_transform(imp.fit_transform(X.iloc[data["idx_train"]])),
                                 columns=feat_leaky)
            Xte_l = pd.DataFrame(sc.transform(imp.transform(X.iloc[data["idx_test"]])),
                                 columns=feat_leaky)

            out = compute_shap_with_slos_included(
                Xtr_l, data["ytr"], Xte_l, CFG.GBM_PARAMS)
            out["feat_names"] = feat_leaky
            return out
        shap_leaky = ck.get_or_compute("s7_shap_leaky", _shap_leaky,
                                       input_fingerprint=data_fp)
        print("Top 10 SHAP features BEFORE audit (with slos):")
        print(shap_leaky["importance"].head(10).to_string(index=False))

        # Compute before/after ranking comparison
        if shap_audited is not None:
            compare = build_before_after_comparison(shap_audited["sv"], shap_leaky["sv"])
            compare.to_csv(tbl_dir / "table_S2_shap_before_after.csv", index=False)
            print(f"\n  wrote Table S2: tables/table_S2_shap_before_after.csv")
    elif ck.exists("s7_shap_leaky"):
        shap_leaky = _load_if_exists("s7_shap_leaky")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 8 — figures
    # ═════════════════════════════════════════════════════════════════════════
    if "figures" in selected:
        log_stage("8 / figures (PNG + PDF at DPI=300)")
        if data is not None and "clean_df" in data:
            FIG.fig1_km_by_disease(data["clean_df"], fig_dir)
        if results is not None:
            FIG.fig2_model_comparison(results, fig_dir)
            FIG.fig3_auc_curve(results, fig_dir)
            FIG.fig4_calibration(results, fig_dir)
        if shap_audited is not None and data is not None:
            FIG.fig5_shap(shap_audited["sv"], data["feat_cols"], fig_dir)
            FIG.fig6_shap_waterfall(shap_audited["sv"], shap_audited["cases"], fig_dir)
        if sub_df is not None:
            FIG.fig8_subgroup(sub_df, fig_dir)
        if (results is not None and "GBM Survival" in results
                and results["GBM Survival"].get("dca") is not None):
            FIG.fig10_dca(results["GBM Survival"]["dca"], fig_dir, "GBM Survival (audited)")
        if shap_audited is not None and shap_leaky is not None:
            compare = build_before_after_comparison(shap_audited["sv"], shap_leaky["sv"])
            FIG.figA1_shap_before_after(compare, fig_dir)
        if data is not None:
            FIG.figA2_audit_overview(data["audit_report"], fig_dir)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 9 — tables
    # ═════════════════════════════════════════════════════════════════════════
    if "tables" in selected:
        log_stage("9 / tables")
        if results is not None:
            rows = []
            for name, r in results.items():
                rows.append({
                    "Model":        name,
                    "C-index":      round(r["cindex"], 4),
                    "CI lo":        round(r["ci_lo"], 4),
                    "CI hi":        round(r["ci_hi"], 4),
                    "IBS":          round(r["ibs"], 4) if not np.isnan(r["ibs"]) else None,
                    "AUC@18d":      round(r[f"auc_{CFG.AUC_EARLY_DAYS}d"], 4),
                    "AUC@108d":     round(r[f"auc_{CFG.AUC_LATE_DAYS}d"], 4),
                    "AUC(t) mean":  round(r["auc_curve_mean"], 4) if not np.isnan(r["auc_curve_mean"]) else None,
                })
            pd.DataFrame(rows).to_csv(tbl_dir / "table2_main_results.csv", index=False)
            print(f"  wrote tables/table2_main_results.csv")

        if sub_df is not None:
            sub_df.to_csv(tbl_dir / "table3_subgroup.csv", index=False)
            print(f"  wrote tables/table3_subgroup.csv")

        if delong_df is not None:
            delong_df.to_csv(tbl_dir / "table_delong.csv", index=False)
            print(f"  wrote tables/table_delong.csv")

        if (results is not None and "GBM Survival" in results
                and results["GBM Survival"].get("dca") is not None):
            results["GBM Survival"]["dca"].to_csv(tbl_dir / "table_dca.csv", index=False)
            print(f"  wrote tables/table_dca.csv")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 10 — manifest
    # ═════════════════════════════════════════════════════════════════════════
    if "manifest" in selected:
        log_stage("10 / manifest (config snapshot + data hash)")
        manifest = {
            "run_id":            run_dir.name,
            "pipeline_version":  PIPELINE_VERSION,
            "timestamp":         datetime.now().isoformat(timespec="seconds"),
            "python":            sys.version,
            "platform":          sys.platform,
            "gpu":               gpu,
            "data_hash_sha256":  data["data_hash"],
            "data_rows":         len(data["raw_df"]),
            "n_features":        len(data["feat_cols"]),
            "seed":              CFG.SEED,
            "n_bootstrap":       CFG.N_BOOTSTRAP,
            "n_delong":          CFG.N_DELONG,
            "skip_features":     sorted(list(CFG.SKIP_FEATURES)),
            "missing_indicators":list(CFG.MISSING_INDICATOR_FEATURES),
            "config_snapshot": {
                "COX_PH_PARAMS":  CFG.COX_PH_PARAMS,
                "COX_EN_PARAMS":  CFG.COX_EN_PARAMS,
                "RSF_PARAMS":     {k: v for k, v in CFG.RSF_PARAMS.items() if k != "n_jobs"},
                "GBM_PARAMS":     CFG.GBM_PARAMS,
                "AUC_EARLY_DAYS": CFG.AUC_EARLY_DAYS,
                "AUC_LATE_DAYS":  CFG.AUC_LATE_DAYS,
            },
            "stages_completed":  sorted(list(ck.index.keys())),
        }
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  wrote manifest.json")

    # ═════════════════════════════════════════════════════════════════════════
    # Done
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*78}\nPIPELINE COMPLETE  — {datetime.now().strftime('%H:%M:%S')}")
    print(f"Outputs in: {run_dir}")
    print(f"  figures/    — all PNG + PDF figures at DPI={CFG.DPI}")
    print(f"  tables/     — all CSV tables")
    print(f"  checkpoints/— pickled intermediate artifacts (for resume)")
    print(f"  logs/       — pipeline.log")
    print(f"  manifest.json — run metadata + data hash for reproducibility")
    print("═" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Received Ctrl+C — checkpoints saved, safe to resume with --resume LATEST")
        sys.exit(130)
    except Exception as e:
        print("\n\n[ERROR] unhandled exception:")
        traceback.print_exc()
        print("\n[HINT] rerun with --resume LATEST to continue from the last completed stage")
        sys.exit(1)
