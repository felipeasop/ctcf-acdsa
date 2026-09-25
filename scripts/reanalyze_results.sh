#!/usr/bin/env bash
# Recompute all reported statistics from the compact repository result table.
# This path does not download data and does not execute MEME or TOMTOM.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

bash "$SCRIPT_DIR/00_preflight.sh" --analysis-only
"$PYTHON" "$SCRIPT_DIR/11_final_analysis.py" \
    --pairs-input "$ROOT/results/final_analysis_pairs.tsv" \
    --gabpa-diagnostics "$ROOT/results/gabpa_diagnostics.json" \
    --output "$ROOT/results/final_analysis.json"
"$PYTHON" "$SCRIPT_DIR/12_record_provenance.py"
printf 'Repository statistical results reproduced without discovery reruns.\n'
