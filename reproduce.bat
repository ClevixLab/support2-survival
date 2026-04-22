@echo off
REM reproduce.bat — one-command reproduction for Windows
REM ======================================================
REM Tier A (default): reproduce every number, table, and figure from raw data.
REM                    Runtime: ~45-60 min on a 6-core CPU laptop.
REM
REM Tier B (--from-checkpoints): skip training, re-generate figures/tables
REM                    from precomputed checkpoints in runs\v1.0_reference\.
REM                    Runtime: ~2 min.
REM
REM Usage:
REM   reproduce.bat                       (Tier A, full reproduction)
REM   reproduce.bat --from-checkpoints    (Tier B, figures only)
REM   reproduce.bat --fast                (Tier A with reduced bootstrap)

setlocal enabledelayedexpansion

set MODE=full
set FAST_FLAG=

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--from-checkpoints" set MODE=from-checkpoints
if /i "%~1"=="--fast"             set FAST_FLAG=--fast
if /i "%~1"=="-h"                 goto help
if /i "%~1"=="--help"             goto help
shift
goto parse_args
:args_done

echo ========================================================================
echo SUPPORT2 Audited Survival Pipeline — Reproduction Script (Windows)
echo ========================================================================

echo [1/4] Checking Python environment...
python --version
python -c "import sksurv, sklearn, pandas, numpy, shap, lifelines, matplotlib" 2>NUL
if errorlevel 1 (
    echo    Missing deps. Run: pip install -r requirements.txt
    exit /b 1
)
echo    OK

echo [2/4] Checking dataset...
if not exist "data\support2_full.csv" (
    echo    Dataset not found. Downloading...
    python download_data.py
)

if "%MODE%"=="from-checkpoints" (
    echo [3/4] Tier B — regenerating figures and tables from checkpoints...
    if not exist "runs\v1.0_reference\checkpoints" (
        echo    ERROR: runs\v1.0_reference\checkpoints\ not found.
        echo    Either use Tier A, or download reference checkpoints from Zenodo.
        exit /b 1
    )
    python run_pipeline.py --resume v1.0_reference --stages figures,tables,manifest
) else (
    echo [3/4] Tier A — full pipeline from raw data ^(~45-60 min^)...
    python run_pipeline.py %FAST_FLAG%
    if errorlevel 1 exit /b 1

    echo.
    echo [3b/4] Running LODGO cross-validation ^(~25-35 min^)...
    python leave_one_disease_out.py
    if errorlevel 1 exit /b 1

    echo.
    echo [3c/4] Generating Figure 14...
    python make_figure_14.py
)

echo.
echo [4/4] Reproduction complete.
echo ========================================================================
echo See docs\REPRODUCE.md for expected values and tolerance notes.
echo ========================================================================
exit /b 0

:help
echo Usage: reproduce.bat [--from-checkpoints] [--fast]
echo.
echo Tier A (default):      full pipeline from raw data, ~45-60 min
echo Tier B:                regenerate figures from precomputed checkpoints
echo --fast:                reduced bootstrap count for quick testing
exit /b 0
