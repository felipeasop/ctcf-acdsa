#!/usr/bin/env python3
"""Summarize extension evaluations without fitting a reliability classifier."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = list(csv.DictReader((args.evaluation / "metrics.tsv").open(), delimiter="\t"))
    if not rows:
        raise SystemExit("evaluation has no rows")
    groups = defaultdict(list)
    for row in rows:
        key = (row["model"], row["signal_percent"], row.get("motif_width", ""))
        groups[key].append(row)
    summary = []
    for (model, level, width), values in sorted(groups.items(), key=lambda item: str(item[0])):
        ap = [float(row["ap"]) for row in values]
        auroc = [float(row["auroc"]) for row in values]
        summary.append({"model": model, "signal_percent": float(level),
            "motif_width": int(width) if width else None, "n": len(values),
            "mean_ap": sum(ap) / len(ap), "mean_auroc": sum(auroc) / len(auroc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source": str(args.evaluation.resolve()), "groups": summary}, indent=2) + "\n")
    print(json.dumps({"groups": len(summary), "output": str(args.output)}))


if __name__ == "__main__":
    main()
