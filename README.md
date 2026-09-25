# OOPS–ZOOPS robustness pipeline

This repository contains the executable, reproducible analysis for the
controlled ChIP-seq motif-discovery benchmark. It compares MEME under the OOPS
and ZOOPS occurrence models for four fixed GRCh38 ENCODE peak sets:

- CTCF in Calu-3 (development cohort);
- CTCF in A549 and MCF-7 (external CTCF cohorts); and
- GABPA in A549 (external factor/cohort).

## Quick start

Run commands from this directory with the locked environment activated and
the required command-line tools available on `PATH`.

```bash
# Check dependencies, syntax, and deterministic tests only
bash scripts/run_pipeline.sh --check-only

# Run the complete pipeline, using three concurrent MEME jobs (can be adjusted as needed)
bash scripts/run_pipeline.sh --jobs 3
```

Increase `--jobs` according to available memory. `--timeout` controls the
maximum time allowed for one MEME task. The default runtime workspace is
`work/main/`, which is ignored by Git; use `--work DIR` to place it elsewhere.

The pipeline is resumable. Re-run the same command after an interruption:
complete artifacts are validated and reused, while incomplete directories stop
with an error instead of being silently mixed with a new run. Downloads are
also reused after checksum validation.

To recompute the published statistical summaries from the compact repository
table, without downloading data or running MEME/TOMTOM:

```bash
bash scripts/reanalyze_results.sh
```

This command regenerates `results/final_analysis.json` and updates the
provenance manifest. It uses the lightweight analysis-only preflight and does
not require the large runtime workspace or MEME-related executables.

For a direct dependency check, use the full preflight for the complete
pipeline, or the lightweight form for compact statistical reanalysis:

```bash
bash scripts/00_preflight.sh
bash scripts/00_preflight.sh --analysis-only
```

## Pipeline stages

| Stage | Script | Purpose |
|---:|---|---|
| 00 | `scripts/00_preflight.sh` | Check executables, Python/Bash syntax, and tests. |
| 01 | `scripts/01_download_inputs.sh` | Download or validate GRCh38 and the four frozen peak sets. |
| 02 | `scripts/02_prepare_splits.py` | Create fixed-length windows and grouped train/test splits. |
| 03 | `scripts/03_generate_negatives.py` | Generate GC-matched, paired negative sequences and FASTAs. |
| 04 | `scripts/04_prepare_background.py` | Build the training-negative Markov background. |
| 05 | `scripts/05_trajectories.py` | Create nested signal trajectories, seeds, and MEME plans. |
| 06 | `scripts/06_discover.py` | Run the planned MEME OOPS and ZOOPS discoveries. |
| 07 | `scripts/07_evaluate.py` | Score held-out sequences and compare PWMs with TOMTOM. |
| 08 | `scripts/08_reliability.py` | Fit the development diagnostic and apply it to external cohorts. |

Every stage has one responsibility. `scripts/run_pipeline.sh` invokes stages
00–08 in this order; it does not provide a second scientific processing mode.

## Inputs and runtime data

The frozen input manifest is
`manifests/inputs/external_peak_sets.tsv`. It records each ENCODE accession,
URL, peak type, and MD5 checksum. Runtime data are generated under the
ignored `work/` directory and are never required in Git.

The initial experiment uses operational signal levels 100, 75, 50, 25, 10,
and 0 percent. The exploratory extension orchestrator adds CTCF levels 55,
60, 65, and 70 percent and GABPA motif widths 10–14:

```bash
JOBS=10 MEME_TIMEOUT=3600 bash scripts/run_extensions.sh
# The number of jobs and meme timeout can be adjusted as needed
```

This command runs only computational extensions and final analysis. It does
not require the manuscript and reuses complete plans and discoveries from
`work/main/`.

The extension command writes the compact final results and provenance manifest
to `results/`. Manuscript formatting is maintained outside this repository.

## Post-08 analysis scripts

These scripts operate after the main 00–08 pipeline:

- `09_extension_plan.py` creates exploratory CTCF fine-grid and GABPA width
  plans without changing the original design;
- `10_summarize_extension.py` consolidates extension evaluations;
- `11_final_analysis.py` computes the statistical summaries, trajectory-cluster
  bootstrap intervals, and compact result table;
- `12_record_provenance.py` records software versions and SHA-256 hashes for
  the repository inputs and scripts.

The wrappers `run_extensions.sh` and `reanalyze_results.sh` call these steps
in the appropriate order. The latter uses only compact results and does not
download data or execute MEME/TOMTOM.

## Repository contents

- `scripts/` and `scripts/lib/`: pipeline, analysis, and I/O code;
- `config/`: frozen experimental and robustness settings;
- `manifests/`: frozen external-input metadata;
- `references/`: motif references and the GRCh38 blacklist;
- `tests/`: deterministic regression tests;
- `results/final_analysis_pairs.tsv`: compact table containing the 450 analyzed pairs;
- `results/final_analysis.json`: bootstrapped final statistical summaries;
- `results/gabpa_diagnostics.json`: compact GABPA diagnostic summaries;
- `results/reproducibility_manifest.json`: tool versions and SHA-256 hashes;
- `environment.yml` and `conda-linux-64.lock`: human-readable and exact Linux
  environment specifications.

Large downloads, intermediate FASTAs, MEME/TOMTOM outputs, logs, and temporary
files are intentionally excluded by `.gitignore`. They are reproducible with
the complete pipeline and should not be copied into the Git repository.

## Interpretation and scope

The benchmark measures recovery of a frozen operational motif reference and
held-out sequence discrimination under controlled sequence contamination. A
negative sequence is a matched computational control.

The compact results are sufficient to reproduce the reported statistics, but
not to recreate raw downloads or MEME output directories. Those artifacts are
generated only by the full pipeline and remain outside version control.
