#!/usr/bin/env python3
"""Summit windows, blacklist filtering and deterministic overlap-group splitting."""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from genomic_intervals import IntervalIndex, Region, chromosome_alias, normalize_regions_to_chrom_sizes, read_bed_regions, read_chrom_sizes
from robustness_io import atomic_json, atomic_write
from split_intervals import write_split


def narrowpeak_windows(path, chrom_sizes, extension, eligible):
    rows, excluded, invalid_summits, identifiers, noneligible = [], 0, 0, set(), Counter()
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                raise ValueError(f"{path}:{line_number}: expected ten narrowPeak columns")
            source_chrom, peak_start, peak_end, offset = fields[0], int(fields[1]), int(fields[2]), int(fields[9])
            chrom = chromosome_alias(source_chrom, chrom_sizes)
            if chrom is None or chrom not in eligible:
                noneligible[source_chrom] += 1
                continue
            if peak_start < 0 or peak_end <= peak_start or not 0 <= offset < peak_end - peak_start:
                invalid_summits += 1
                continue
            if peak_end > chrom_sizes[chrom]:
                raise ValueError(f"{path}:{line_number}: peak exceeds chromosome length")
            start, end = peak_start + offset - extension, peak_start + offset + extension + 1
            if start < 0 or end > chrom_sizes[chrom]:
                excluded += 1
                continue
            score = float(fields[8])
            if not math.isfinite(score) or score < 0:
                raise ValueError(f"{path}:{line_number}: invalid q-value statistic")
            identifier = fields[3] if fields[3] not in ("", ".") else f"peak_{line_number}"
            if identifier in identifiers:
                raise ValueError("duplicate narrowPeak identifier")
            identifiers.add(identifier)
            rows.append({"chrom": chrom, "start": start, "end": end, "peak_id": identifier, "score": score,
                         "source_chrom": source_chrom, "source_line": line_number})
    if len(rows) < 2:
        raise ValueError("fewer than two eligible peak windows")
    return rows, excluded, invalid_summits, dict(noneligible)


def filter_blacklisted_windows(rows, blacklist, chrom_sizes):
    regions, renamed, skipped = normalize_regions_to_chrom_sizes(read_bed_regions(blacklist), chrom_sizes)
    if not regions:
        raise ValueError("blacklist has no reference-compatible regions")
    index, kept, excluded = IntervalIndex(regions), [], Counter()
    for row in rows:
        if index.overlaps(Region(row["chrom"], row["start"], row["end"])):
            excluded[row["chrom"]] += 1
        else:
            kept.append(row)
    return kept, dict(excluded), renamed, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--narrowpeak", type=Path, default=ROOT / "results/preprocessing/peaks/ctcf_peaks.qvalue_sorted.narrowPeak")
    p.add_argument("--chrom-sizes", type=Path, default=ROOT / "genome/GRCh38.chrom.sizes")
    p.add_argument("--blacklist", type=Path, default=ROOT / "references/hg38-blacklist.v2.bed")
    p.add_argument("--config", type=Path, default=ROOT / "config/robustness.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "manifests/splits")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    extension, length = int(config["window_extension_bp"]), int(config["window_length_nt"])
    if extension < 0 or length != 2 * extension + 1 or not config["eligible_chromosomes"]:
        p.error("invalid window length or eligible chromosome list")
    sizes = read_chrom_sizes(args.chrom_sizes)
    eligible = {chromosome_alias(chrom, sizes) for chrom in config["eligible_chromosomes"]}
    if None in eligible:
        raise ValueError("reference is missing a configured eligible chromosome")
    rows, boundary, invalid_summits, chromosome_exclusions = narrowpeak_windows(args.narrowpeak, sizes, extension, eligible)
    canonical = len(rows)
    rows, blacklist_exclusions, renamed, skipped = filter_blacklisted_windows(rows, args.blacklist, sizes)
    if len(rows) < 2:
        raise ValueError("fewer than two windows after filters")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    windows = args.output_dir / f"ctcf_windows_{length}nt.bed"
    atomic_write(windows, lambda f: f.writelines(f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['peak_id']}\t{r['score']:.12g}\n" for r in rows))
    split = config["split"]
    report = write_split(windows, args.output_dir / "ctcf_train_test.tsv", args.output_dir / "ctcf_train_test.json",
                         split["test_fraction"], split["seed"], split["score_bins"])
    summary = {"inputs": [str(path.resolve()) for path in (args.narrowpeak, args.chrom_sizes, args.blacklist, args.config)],
        "window_length_nt": length, "canonical_windows_before_blacklist": canonical, "eligible_windows": len(rows),
        "chromosome_aliases_normalized": sum(r["source_chrom"] != r["chrom"] for r in rows),
        "boundary_windows_excluded": boundary, "noneligible_chromosome_exclusions": chromosome_exclusions,
        "invalid_summit_records": invalid_summits,
        "blacklist_chromosome_exclusions": blacklist_exclusions, "blacklist_chromosomes_renamed": renamed,
        "blacklist_regions_skipped": skipped, "score_field": "narrowPeak column 9 (-log10 q-value)",
        "windows": str(windows.resolve()), "split": report}
    atomic_json(args.output_dir / f"ctcf_windows_{length}nt.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
