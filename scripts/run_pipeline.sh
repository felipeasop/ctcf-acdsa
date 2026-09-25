#!/usr/bin/env bash
# Run stages 00–08 in order. Incomplete prepared stages fail rather than being
# mixed with a new run; discovery and evaluation resume complete cells.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$PIPELINE_DIR
source "$SCRIPT_DIR/lib/common.sh"

WORK=${CTCF_WORK:-$REPO_DIR/work/main}
JOBS=${JOBS:-3}
TIMEOUT=${MEME_TIMEOUT:-3600}
CHECK_ONLY=0

usage() {
    printf 'Usage: bash %s [--work DIR] [--jobs N] [--timeout SECONDS] [--check-only]\n' "$0"
    printf 'The single main pipeline downloads/reuses GRCh38 and frozen ENCODE peak sets, then runs stages 02–08.\n'
}
while (($#)); do
    case $1 in
        --work) WORK=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --timeout) TIMEOUT=$2; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "Unknown argument: $1" ;;
    esac
done
[[ $JOBS =~ ^[1-9][0-9]*$ && $TIMEOUT =~ ^[1-9][0-9]*$ ]] || die "JOBS and timeout must be positive integers"
GENOME_DIR=${CTCF_GENOME_DIR:-$WORK/reference}
GENOME_FA=$GENOME_DIR/Homo_sapiens.GRCh38.dna.primary_assembly.fa
CHROM_SIZES=$GENOME_DIR/GRCh38.chrom.sizes
BLACKLIST=$PIPELINE_DIR/references/hg38-blacklist.v2.bed
DESIGN=$PIPELINE_DIR/config/design.json
ROBUSTNESS=$PIPELINE_DIR/config/robustness.json
RAW_PEAKS=$WORK/data/raw/peaks

for input in "$BLACKLIST" "$DESIGN" "$ROBUSTNESS"; do
    require_nonempty "$input"
done

all_present() {
    local path
    for path in "$@"; do [[ -s $path ]] || return 1; done
}
reuse_or_empty() {
    local label=$1 directory=$2
    shift 2
    if all_present "$@"; then
        log "REUSE $label"
        return 0
    fi
    [[ ! -e $directory ]] || die "$label is incomplete at $directory; inspect it instead of mixing outputs"
    return 1
}

prepare_and_run() {
    local cohort=$1 peaks=$2
    local processed=$WORK/data/processed/$cohort
    local manifests=$WORK/manifests/$cohort
    local results=$WORK/results/$cohort
    local split=$manifests/splits
    local pairs=$manifests/negative_pairs
    local background=$processed/background
    local trajectories=$processed/trajectories
    local datasets=$processed/datasets

    log "$cohort: stage 02 — windows and split"
    if ! reuse_or_empty "$cohort/02" "$split" \
        "$split/ctcf_train_test.tsv" "$split/ctcf_train_test.json" "$split/ctcf_windows_${WINDOW_LENGTH}nt.json"; then
        python3 "$SCRIPT_DIR/02_prepare_splits.py" --narrowpeak "$peaks" \
            --chrom-sizes "$CHROM_SIZES" --blacklist "$BLACKLIST" \
            --config "$ROBUSTNESS" --output-dir "$split"
    fi

    log "$cohort: stage 03 — negatives and FASTA files"
    if ! reuse_or_empty "$cohort/03" "$pairs" \
        "$pairs/ctcf_negative_pairs.tsv" "$pairs/ctcf_negative_pairs.json" \
        "$datasets/train_positives.fa" "$datasets/train_negatives.fa" \
        "$datasets/test_positives.fa" "$datasets/test_negatives.fa"; then
        [[ ! -e $datasets ]] || die "$cohort/03 datasets are incomplete at $datasets"
        python3 "$SCRIPT_DIR/03_generate_negatives.py" \
            --cohort "$cohort" \
            --split "$split/ctcf_train_test.tsv" --genome "$GENOME_FA" \
            --chrom-sizes "$CHROM_SIZES" --peaks "$peaks" --blacklist "$BLACKLIST" \
            --config "$ROBUSTNESS" --output-dir "$pairs" --fasta-dir "$datasets"
    fi

    log "$cohort: stage 04 — background"
    if ! reuse_or_empty "$cohort/04" "$background" \
        "$background/train_negatives_markov2.txt" "$background/uniform_order0.txt" "$background/summary.json"; then
        python3 "$SCRIPT_DIR/04_prepare_background.py" \
            --train-negatives "$datasets/train_negatives.fa" --config "$ROBUSTNESS" \
            --test-negatives "$datasets/test_negatives.fa" \
            --output "$background/train_negatives_markov2.txt" \
            --uniform-output "$background/uniform_order0.txt" --summary "$background/summary.json"
    fi

    log "$cohort: stage 05 — trajectories and plan"
    if ! reuse_or_empty "$cohort/05" "$trajectories" "$trajectories/plan.json"; then
        python3 "$SCRIPT_DIR/05_trajectories.py" --cohort "$cohort" \
            --datasets "$datasets" --pairs "$pairs/ctcf_negative_pairs.tsv" \
            --config "$DESIGN" --output "$trajectories"
    fi

    # Validate provenance before reusing prepared artifacts.
    python3 - "$split/ctcf_windows_${WINDOW_LENGTH}nt.json" "$pairs/ctcf_negative_pairs.json" \
        "$background/summary.json" "$trajectories/plan.json" "$peaks" "$GENOME_FA" \
        "$CHROM_SIZES" "$BLACKLIST" "$ROBUSTNESS" "$DESIGN" "$cohort" <<'PY'
import json
import pathlib
import sys

(split_path, pairs_path, background_path, plan_path, peaks, genome, sizes,
 blacklist, robustness, design_path, cohort) = sys.argv[1:]
resolve = lambda value: str(pathlib.Path(value).resolve())
split = json.loads(pathlib.Path(split_path).read_text())
pairs = json.loads(pathlib.Path(pairs_path).read_text())
background = json.loads(pathlib.Path(background_path).read_text())
plan = json.loads(pathlib.Path(plan_path).read_text())
design = json.loads(pathlib.Path(design_path).read_text())
robustness_config = json.loads(pathlib.Path(robustness).read_text())
negative_defaults = robustness_config["negatives"]
negative_expected = dict(negative_defaults, **negative_defaults["cohorts"][cohort])
expected_split_name = f"ctcf_windows_{robustness_config['window_length_nt']}nt.json"
if pathlib.Path(split_path).name != expected_split_name:
    raise SystemExit(f"unexpected split provenance file: {split_path}")
split_table = str(pathlib.Path(split_path).with_name("ctcf_train_test.tsv"))
if split["inputs"] != list(map(resolve, (peaks, sizes, blacklist, robustness))):
    raise SystemExit(f"split provenance differs from current inputs: {split_path}")
if pairs["inputs"] != list(map(resolve, (split_table,
                                           genome, sizes, peaks, blacklist, robustness))):
    raise SystemExit(f"negative-pair provenance differs from current inputs: {pairs_path}")
if pairs.get("gc_tolerance") != negative_expected["gc_tolerance"]:
    raise SystemExit(f"negative-pair GC tolerance differs from declared cohort protocol: {pairs_path}")
if pairs.get("maximum_rounds") != negative_expected["max_rounds"]:
    raise SystemExit(f"negative-pair search budget differs from declared cohort protocol: {pairs_path}")
if background["source"] != resolve(pathlib.Path(background_path).parent.parent / "datasets/train_negatives.fa"):
    raise SystemExit(f"background source differs from current datasets: {background_path}")
if plan.get("config") != design:
    raise SystemExit(f"trajectory plan differs from current design: {plan_path}")
tasks = plan.get("tasks", [])
expected = (design["trajectories"]["samples"] * len(design["trajectories"]["levels_percent"])
            * design["trajectories"]["input_repeats"] * 2)
if len(tasks) != expected or {task["cohort"] for task in tasks} != {cohort}:
    raise SystemExit(f"trajectory plan is incomplete or belongs to another cohort: {plan_path}")
for task in tasks:
    fasta = pathlib.Path(task["input"])
    if not fasta.is_file() or not pathlib.Path(str(fasta) + ".tsv").is_file():
        raise SystemExit(f"trajectory input is missing: {fasta}")
PY

    log "$cohort: stage 06 — discovery (resumable)"
    python3 "$SCRIPT_DIR/06_discover.py" --plan "$trajectories/plan.json" \
        --background "$background/train_negatives_markov2.txt" \
        --jobs "$JOBS" --timeout "$TIMEOUT" --output "$results/discoveries" --execute

    log "$cohort: stage 07 — evaluation (resumable)"
    python3 "$SCRIPT_DIR/07_evaluate.py" --discoveries "$results/discoveries" \
        --positives "$datasets/test_positives.fa" --negatives "$datasets/test_negatives.fa" \
        --background "$background/train_negatives_markov2.txt" --config "$DESIGN" \
        --order 2 --pseudocount 0.5 --output "$results/evaluation"
}

log "stage 00 — environment and syntax"
bash "$SCRIPT_DIR/00_preflight.sh"
if ((CHECK_ONLY)); then
    log "Inputs, dependencies, and syntax verified; no download or experiment was started."
    exit 0
fi
WINDOW_LENGTH=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["window_length_nt"])' "$ROBUSTNESS")

log "stage 01 — fixed inputs (download only when absent)"
bash "$SCRIPT_DIR/01_download_inputs.sh" "$PIPELINE_DIR/manifests/inputs/external_peak_sets.tsv" "$RAW_PEAKS" "$GENOME_DIR"

prepare_and_run calu3-development "$RAW_PEAKS/calu3-development.narrowPeak"

DEV_RESULTS=$WORK/results/calu3-development
    log "calu3-development: stage 08 — fitting and freezing"
if ! reuse_or_empty "calu3-development/08" "$DEV_RESULTS/reliability" \
    "$DEV_RESULTS/reliability/pairs.tsv" "$DEV_RESULTS/reliability/protocol.json" "$DEV_RESULTS/reliability/summary.json"; then
        python3 "$SCRIPT_DIR/08_reliability.py" --evaluation "$DEV_RESULTS/evaluation" \
        --config "$DESIGN" --fit --output "$DEV_RESULTS/reliability"
fi
[[ -s $DEV_RESULTS/reliability/frozen_diagnostic.json ]] || \
    die "Development cohort did not produce a frozen diagnostic; external evaluation was not run"

for cohort in a549-encsr035oxa mcf7-encsr000ahd gabpa-a549-encsr000bpy; do
    prepare_and_run "$cohort" "$RAW_PEAKS/$cohort.narrowPeak"
    results=$WORK/results/$cohort
    log "$cohort: stage 08 — external evaluation"
    if ! reuse_or_empty "$cohort/08" "$results/reliability" \
        "$results/reliability/pairs.tsv" "$results/reliability/protocol.json" \
        "$results/reliability/predictions.tsv" "$results/reliability/summary.json"; then
        python3 "$SCRIPT_DIR/08_reliability.py" --evaluation "$results/evaluation" \
            --config "$DESIGN" --frozen "$DEV_RESULTS/reliability/frozen_diagnostic.json" \
            --output "$results/reliability"
    fi
done

log "Main pipeline completed. Results: $WORK/results"
