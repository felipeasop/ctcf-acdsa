#!/usr/bin/env python3
"""Record software versions and SHA-256 hashes for the repository analysis."""

import argparse
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from robustness_io import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command):
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[:6]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--analysis", type=Path,
                        default=ROOT / "results/final_analysis.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/reproducibility_manifest.json")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = args.repo.resolve()
    required = [
        repo / "environment.yml",
        repo / "conda-linux-64.lock",
        repo / "config/design.json",
        repo / "config/robustness.json",
        repo / "manifests/inputs/external_peak_sets.tsv",
        repo / "references/CTCF_full.meme",
        repo / "references/GABPA_MA0062.4.meme",
        repo / "references/hg38-blacklist.v2.bed",
        args.analysis.resolve(),
        repo / "results/final_analysis_pairs.tsv",
        repo / "results/gabpa_diagnostics.json",
    ]
    required.extend(sorted((repo / "scripts").glob("[0-9][0-9]_*.py")))
    required.extend(sorted((repo / "scripts").glob("*.sh")))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing provenance inputs: " + ", ".join(missing))

    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                text=True, capture_output=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None

    payload = {
        "schema_version": 1,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "git_commit": commit,
        "python": {
            "version": platform.python_version(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "tools": {
            "meme": command_output(["meme", "-version"]),
            "tomtom": command_output(["tomtom", "-version"]),
            "bedtools": command_output(["bedtools", "--version"]),
            "samtools": command_output(["samtools", "--version"]),
        },
        "sha256": {
            str(path.resolve().relative_to(repo)): sha256(path)
            for path in sorted(set(required))
        },
    }
    atomic_json(args.output.resolve(), payload, overwrite=True)
    print(f"Provenance written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
