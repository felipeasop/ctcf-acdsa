#!/usr/bin/env python3
"""Estimate a declared Markov background from training negatives only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from fasta_utils import read_fasta
from motif_scoring import parse_markov
from robustness_io import atomic_json, atomic_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-negatives", type=Path, default=ROOT / "results/robustness/datasets/train_negatives.fa")
    p.add_argument("--test-negatives", type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "config/robustness.json")
    p.add_argument("--order", type=int, choices=range(6))
    p.add_argument("--output", type=Path, default=ROOT / "results/robustness/background/train_negatives_markov2.txt")
    p.add_argument("--uniform-output", type=Path, default=ROOT / "results/robustness/background/uniform_order0.txt")
    p.add_argument("--summary", type=Path, default=ROOT / "manifests/inputs/train_negatives_markov2.json")
    p.add_argument("--executable", default="fasta-get-markov")
    p.add_argument("--meme", default="meme")
    args = p.parse_args()
    order = args.order if args.order is not None else json.loads(args.config.read_text())["meme"]["markov_order"]
    targets = [args.output, args.uniform_output, args.summary]
    if len({x.resolve() for x in targets}) != len(targets) or any(x.exists() for x in targets):
        p.error("output paths must be distinct and new")
    train_records = list(read_fasta(args.train_negatives))
    train_ids = {header[1:].split()[0] for header, _ in train_records}
    if not train_ids or any(not identifier.startswith("neg_train_") for identifier in train_ids):
        p.error("--train-negatives must contain only IDs generated for the training split")
    test_ids = set()
    if args.test_negatives:
        test_ids = {header[1:].split()[0] for header, _ in read_fasta(args.test_negatives)}
        if not test_ids or any(not identifier.startswith("neg_test_") for identifier in test_ids):
            p.error("--test-negatives must contain only IDs generated for the test split")
        if train_ids & test_ids:
            p.error("training and test negative IDs overlap")
    count = len(train_records)
    command = [args.executable, "-m", str(order), "-pseudo", "0.1", str(args.train_negatives)]
    atomic_write(args.output, lambda f: subprocess.run(command, stdout=f, check=True))
    parse_markov(args.output, order)
    atomic_write(args.uniform_output, lambda f: f.write("# fixed uniform order-0 DNA background\nA 0.25\nC 0.25\nG 0.25\nT 0.25\n"))
    version = subprocess.run([args.meme, "-version"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    test_data_used = bool(train_ids & test_ids)
    summary = {"source_role": "train_negatives_only", "test_data_used": test_data_used,
        "test_source": str(args.test_negatives.resolve()) if args.test_negatives else None, "source_sequence_count": count,
        "source": str(args.train_negatives.resolve()), "markov_order": order, "background_pseudocount": 0.1,
        "command": command, "meme_suite_version": version, "output": str(args.output.resolve()),
        "uniform_output": str(args.uniform_output.resolve())}
    atomic_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
