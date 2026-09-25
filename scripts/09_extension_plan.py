#!/usr/bin/env python3
"""Create a separate plan for the fine-grid or motif-width extension.

This script never runs MEME. Fine-grid plans reuse the existing datasets and
negative pairs, while width plans reuse an existing trajectory plan. Outputs
are kept outside the primary protocol so the original results remain frozen.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from robustness_io import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("fine-ctcf", "gabpa-width"), required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--levels", nargs="+", type=float, required=True)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--widths", nargs="+", type=int)
    p.add_argument("--source-plan", type=Path)
    args = p.parse_args()
    if args.samples < 1 or len(set(args.levels)) != len(args.levels):
        p.error("samples must be positive and levels must be unique")
    if any(level < 0 or level > 100 for level in args.levels):
        p.error("levels must be in [0, 100]")
    if args.output.exists():
        p.error(f"output already exists: {args.output}")
    if args.mode == "fine-ctcf":
        make_fine_ctcf(args)
    else:
        make_width_plan(args)


def make_fine_ctcf(args):
    if args.cohort not in {"calu3-development", "a549-encsr035oxa", "mcf7-encsr000ahd"}:
        raise SystemExit("fine-ctcf accepts only the three CTCF cohorts")
    if args.widths:
        raise SystemExit("--widths is only valid for gabpa-width")
    config = json.loads((ROOT / "config/design.json").read_text())
    config["trajectories"] = dict(config["trajectories"],
                                  levels_percent=args.levels, samples=args.samples)
    processed = args.work / "data/processed" / args.cohort
    manifests = args.work / "manifests" / args.cohort
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=True) as handle:
        json.dump(config, handle)
        handle.flush()
        command = [sys.executable, str(ROOT / "scripts/05_trajectories.py"),
                   "--cohort", args.cohort,
                   "--datasets", str(processed / "datasets"),
                   "--pairs", str(manifests / "negative_pairs/ctcf_negative_pairs.tsv"),
                   "--config", handle.name, "--output", str(args.output),
                   "--allow-partial-levels"]
        subprocess.run(command, check=True)
    plan = json.loads((args.output / "plan.json").read_text())
    plan["role"] = "fine_signal_grid"
    plan["primary_protocol"] = str((ROOT / "config/design.json").resolve())
    atomic_json(args.output / "plan.json", plan, overwrite=True)


def make_width_plan(args):
    if args.cohort != "gabpa-a549-encsr000bpy" or not args.widths or not args.source_plan:
        raise SystemExit("gabpa-width requires the GABPA cohort, --widths and --source-plan")
    if any(width < 4 for width in args.widths):
        raise SystemExit("motif widths must be at least four")
    source = json.loads(args.source_plan.read_text())
    tasks = [{**task, "width": width, "role": "width_sensitivity"}
             for width in args.widths for task in source["tasks"]
             if task["signal_percent"] in args.levels and task["sample"] <= args.samples]
    if not tasks:
        raise SystemExit("source plan has no matching tasks")
    args.output.mkdir(parents=True)
    atomic_json(args.output / "plan.json", {"schema_version": 1, "role": "gabpa_width_sensitivity",
        "source_plan": str(args.source_plan.resolve()), "widths": args.widths, "levels": args.levels,
        "tasks": tasks})
    print(json.dumps({"tasks": len(tasks), "output": str(args.output)}))


if __name__ == "__main__":
    main()
