#!/usr/bin/env python3
"""Score frozen motifs and save all TOMTOM fields, independently of discovery."""
import argparse
import csv
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from experiments import cohort_settings, discovery_runs, evaluation_protocol, legacy_runs, tomtom_rows
from motif_scoring import apply_pwm_pseudocount, parse_markov, parse_meme_xml, score_fasta_numpy
from robustness_io import atomic_json, atomic_tsv
from robustness_metrics import average_precision, auroc


def evaluate_motif(run, args, background, shared, protocol):
    """Evaluate one fixed PWM; discovery is never rerun here."""
    start = time.monotonic()
    xml = run["directory"] / "meme/meme.xml"
    identity = {**shared, "motif": str(xml.resolve()), "discovery_id": run["run_id"]}
    evaluation_id = run["run_id"]
    directory = args.output / evaluation_id
    temporary = Path(tempfile.mkdtemp(prefix=f"{evaluation_id}.", dir=args.output / ".partial"))
    motif = parse_meme_xml(xml)
    matrix = apply_pwm_pseudocount(motif.matrix, motif.sites, args.pseudocount)
    pos = score_fasta_numpy(args.positives, matrix, background, args.order)
    neg = score_fasta_numpy(args.negatives, matrix, background, args.order)
    if {x for x, _ in pos} & {x for x, _ in neg}:
        raise ValueError("positive and negative IDs overlap")
    labels, values = [1] * len(pos) + [0] * len(neg), [s for _, s in pos + neg]
    with gzip.open(temporary / "scores.tsv.gz", "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sequence_id", "label", "max_log_odds"])
        writer.writerows((identifier, label, score) for (identifier, score), label in zip(pos + neg, labels))
    command = [args.tomtom, "-oc", str(temporary / "tomtom"), "-no-ssc", "-verbosity", "1",
        "-min-overlap", str(args.minimum_overlap), "-dist", "pearson", "-thresh", "1", str(xml.parent / "meme.txt"), str(args.reference)]
    recorded_command = command[:2] + [str(directory / "tomtom")] + command[3:]
    atomic_json(temporary / "config.json", {**identity, "tomtom_command": recorded_command})
    try:
        with (temporary / "tomtom.log").open("x") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=120)
        matches = tomtom_rows(temporary / "tomtom/tomtom.tsv")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    row = {k: run[k] for k in ("cohort", "sample", "trajectory", "order_repeat", "meme_repeat", "signal_percent", "model", "background", "meme_seed")}
    row.update(run_id=run["run_id"], discovery_id=run["run_id"], evaluation_id=evaluation_id,
        ap=average_precision(labels, values),
        input_repeat=run.get("input_repeat", 0), motif=str(xml.resolve()),
        contamination_percent=100 - float(run["signal_percent"]),
        auroc=auroc(labels, values), positives=len(pos), negatives=len(neg), scoring_order=args.order,
        pseudocount=args.pseudocount, sites=motif.sites, site_fraction=motif.sites / motif.primary_count,
        motif_evalue=motif.e_value_text, motif_neg_log10_evalue=motif.neg_log10_evalue,
        motif_width=motif.width, sample_size=motif.primary_count,
        protocol=protocol, input=str(Path(run["input"]).resolve()),
        discovery_background=run["discovery_background"],
        runtime_seconds=time.monotonic() - start)
    match_rows = []
    for match in matches:
        match_rows.append({"evaluation_id": evaluation_id, "run_id": run["run_id"],
            "sample": run["sample"], "signal_percent": run["signal_percent"], "model": run["model"], **match})
    atomic_json(temporary / "result.json", {**row, "tomtom": matches})
    os.replace(temporary, directory)
    return row, match_rows


def completed_evaluation(run, output, protocol):
    """Load a prior cell only after checking its discovery and scoring contract."""
    path = output / run["run_id"] / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if (result.get("run_id") != run["run_id"] or result.get("protocol") != protocol
            or result.get("discovery_id") != run["run_id"]):
        raise ValueError(f"existing evaluation conflicts with current inputs: {path.parent}")
    matches = [{"evaluation_id": run["run_id"], "run_id": run["run_id"],
                "sample": run["sample"], "signal_percent": run["signal_percent"],
                "model": run["model"], **match} for match in result["tomtom"]]
    return result, matches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--discoveries", type=Path)
    sources.add_argument("--legacy-root", type=Path)
    p.add_argument("--positives", type=Path, required=True)
    p.add_argument("--negatives", type=Path, required=True)
    p.add_argument("--background", type=Path, required=True)
    p.add_argument("--order", type=int, choices=range(6), default=2)
    p.add_argument("--pseudocount", type=float, default=0.5)
    p.add_argument("--reference", type=Path, help="Override the cohort reference for an explicitly separate sensitivity analysis")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/design.json")
    p.add_argument("--tomtom", default="tomtom")
    p.add_argument("--run-ids", nargs="+", help="Explicit subset for a bounded check; omitted evaluates the full supplied cohort")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not math.isfinite(args.pseudocount) or args.pseudocount < 0:
        p.error("pseudocount must be finite and nonnegative")
    background = parse_markov(args.background, args.order)
    runs = list(legacy_runs(args.legacy_root) if args.legacy_root else discovery_runs(args.discoveries))
    if args.run_ids:
        runs = [r for r in runs if r["run_id"] in args.run_ids]
        if {r["run_id"] for r in runs} != set(args.run_ids):
            p.error("one or more requested discoveries were not found")
    if not runs:
        p.error("no completed discoveries found")
    cohorts = {r["cohort"] for r in runs}
    if len(cohorts) != 1:
        p.error("evaluate exactly one cohort at a time")
    cohort = next(iter(cohorts))
    config = json.loads(args.config.read_text())
    if cohort == "calu3-original":
        target = dict(config["reliability"])
        target.update(config["factors"]["CTCF"])
        target["factor"] = "CTCF"
    else:
        _, target = cohort_settings(config, cohort)
    args.minimum_overlap = int(target["reference_minimum_overlap"])
    if args.reference is None:
        filename = "CTCF_full.meme" if cohort == "calu3-original" else target["reference_file"]
        args.reference = Path(__file__).resolve().parents[1] / "references" / filename
    shared = {"order": args.order, "pseudocount": args.pseudocount, "minimum_overlap": args.minimum_overlap,
        "inputs": {name: str(path.resolve()) for name, path in
                    (("positives", args.positives), ("negatives", args.negatives),
                     ("background", args.background), ("reference", args.reference))}}
    protocol = evaluation_protocol(shared["inputs"], args.order, args.pseudocount, args.minimum_overlap)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".partial").mkdir(exist_ok=True)
    plan = {"run_ids": [r["run_id"] for r in runs], "shared": shared, "protocol": protocol}
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing evaluation directory belongs to another plan or input")
    else:
        atomic_json(plan_path, plan)
    rows, match_rows = [], []
    reused = 0
    for run in runs:
        existing = completed_evaluation(run, args.output, protocol)
        if existing:
            row, matches = existing
            reused += 1
        else:
            row, matches = evaluate_motif(run, args, background, shared, protocol)
        rows.append(row)
        match_rows.extend(matches)
    metadata = {**shared, "protocol": protocol, "completed": len(rows), "tomtom_matches": len(match_rows)}
    evaluation_path = args.output / "evaluation.json"
    if evaluation_path.exists():
        if json.loads(evaluation_path.read_text()) != metadata:
            raise ValueError("existing evaluation summary conflicts with completed cells")
        print(json.dumps({"completed": len(rows), "reused": reused, "output": str(args.output)}))
        return
    atomic_tsv(args.output / "metrics.tsv", rows, list(rows[0]), overwrite=True)
    if match_rows:
        atomic_tsv(args.output / "tomtom.tsv", match_rows, list(match_rows[0]), overwrite=True)
    atomic_json(evaluation_path, metadata)
    print(json.dumps({"completed": len(rows), "reused": reused, "output": str(args.output)}))


if __name__ == "__main__":
    main()
