#!/usr/bin/env bash
# Run exploratory extensions, final analysis, and provenance recording.
# Existing complete plans, discoveries, and evaluations are reused.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORK=${WORK:-"$ROOT/work/main"}
JOBS=${JOBS:-1}
MEME_TIMEOUT=${MEME_TIMEOUT:-3600}
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { printf 'JOBS must be a positive integer.\n' >&2; exit 2; }

run_extension() {
    local cohort=$1
    local plan="$WORK/extensions/fine-$cohort/plan.json"
    local out="$WORK/results/extensions/fine-$cohort"
    if [[ ! -s "$plan" ]]; then
        "$PYTHON" "$SCRIPT_DIR/09_extension_plan.py" --mode fine-ctcf \
            --cohort "$cohort" --work "$WORK" --levels 55 60 65 70 --samples 10 \
            --output "$WORK/extensions/fine-$cohort"
    fi
    mkdir -p "$out/discoveries"
    "$PYTHON" "$SCRIPT_DIR/06_discover.py" --plan "$plan" \
        --background "$WORK/data/processed/$cohort/background/train_negatives_markov2.txt" \
        --jobs "$JOBS" --timeout "$MEME_TIMEOUT" --output "$out/discoveries" --execute
    if [[ ! -s "$out/evaluation/metrics.tsv" ]]; then
        "$PYTHON" "$SCRIPT_DIR/07_evaluate.py" --discoveries "$out/discoveries" \
            --positives "$WORK/data/processed/$cohort/datasets/test_positives.fa" \
            --negatives "$WORK/data/processed/$cohort/datasets/test_negatives.fa" \
            --background "$WORK/data/processed/$cohort/background/train_negatives_markov2.txt" \
            --config "$ROOT/config/design.json" --order 2 --pseudocount 0.5 \
            --output "$out/evaluation"
    fi
    if [[ ! -s "$out/summary.json" ]]; then
        "$PYTHON" "$SCRIPT_DIR/10_summarize_extension.py" \
            --evaluation "$out/evaluation" --output "$out/summary.json"
    fi
}

bash "$SCRIPT_DIR/00_preflight.sh"
for cohort in calu3-development a549-encsr035oxa mcf7-encsr000ahd; do
    run_extension "$cohort"
done

GABPA_PLAN="$WORK/extensions/gabpa-width/plan.json"
GABPA_OUT="$WORK/results/extensions/gabpa-width"
if [[ ! -s "$GABPA_PLAN" ]]; then
    "$PYTHON" "$SCRIPT_DIR/09_extension_plan.py" --mode gabpa-width \
        --cohort gabpa-a549-encsr000bpy --work "$WORK" --levels 50 75 100 \
        --samples 10 --widths 10 11 12 13 14 \
        --source-plan "$WORK/data/processed/gabpa-a549-encsr000bpy/trajectories/plan.json" \
        --output "$WORK/extensions/gabpa-width"
fi
mkdir -p "$GABPA_OUT/discoveries"
"$PYTHON" "$SCRIPT_DIR/06_discover.py" --plan "$GABPA_PLAN" \
    --background "$WORK/data/processed/gabpa-a549-encsr000bpy/background/train_negatives_markov2.txt" \
    --jobs "$JOBS" --timeout "$MEME_TIMEOUT" --output "$GABPA_OUT/discoveries" --execute
if [[ ! -s "$GABPA_OUT/evaluation/metrics.tsv" ]]; then
    "$PYTHON" "$SCRIPT_DIR/07_evaluate.py" --discoveries "$GABPA_OUT/discoveries" \
        --positives "$WORK/data/processed/gabpa-a549-encsr000bpy/datasets/test_positives.fa" \
        --negatives "$WORK/data/processed/gabpa-a549-encsr000bpy/datasets/test_negatives.fa" \
        --background "$WORK/data/processed/gabpa-a549-encsr000bpy/background/train_negatives_markov2.txt" \
        --config "$ROOT/config/design.json" --order 2 --pseudocount 0.5 \
        --output "$GABPA_OUT/evaluation"
fi
if [[ ! -s "$GABPA_OUT/summary.json" ]]; then
    "$PYTHON" "$SCRIPT_DIR/10_summarize_extension.py" \
        --evaluation "$GABPA_OUT/evaluation" --output "$GABPA_OUT/summary.json"
fi

"$PYTHON" "$SCRIPT_DIR/11_final_analysis.py" --work "$WORK" \
    --output "$ROOT/results/final_analysis.json" \
    --pairs-output "$ROOT/results/final_analysis_pairs.tsv" \
    --gabpa-diagnostics "$ROOT/results/gabpa_diagnostics.json"
"$PYTHON" "$SCRIPT_DIR/12_record_provenance.py" \
    --analysis "$ROOT/results/final_analysis.json" \
    --output "$ROOT/results/reproducibility_manifest.json"
printf 'Extensions, final analysis, and provenance completed.\n'
