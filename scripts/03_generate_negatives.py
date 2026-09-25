#!/usr/bin/env python3
"""Disjoint chromosome/length/GC-matched negatives, freezing train before test."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from genomic_intervals import IntervalIndex, MutableIntervalIndex, Region, normalize_regions_to_chrom_sizes, read_bed_regions, read_chrom_sizes
from robustness_io import atomic_json, atomic_tsv, atomic_write, read_tsv


def gc_fraction(sequence):
    sequence = sequence.upper()
    valid = sum(base in "ACGT" for base in sequence)
    return (sequence.count("G") + sequence.count("C")) / valid if valid else 0.0


def n_fraction(sequence):
    return sum(base not in "ACGT" for base in sequence.upper()) / len(sequence) if sequence else 1.0


def deterministic_start(seed, target_id, attempt, maximum_start):
    payload = f"{seed}\0{target_id}\0{attempt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (maximum_start + 1)


def extract_sequences(bedtools, genome, records, temporary_root):
    bed, output = temporary_root / "extract.bed", temporary_root / "extract.tsv"
    with bed.open("w") as f:
        f.writelines(f"{r.chrom}\t{r.start}\t{r.end}\t{i}\n" for i, r in records)
    with output.open("w") as f:
        subprocess.run([bedtools, "getfasta", "-fi", str(genome), "-bed", str(bed), "-nameOnly", "-tab"], stdout=f, check=True)
    sequences = {}
    with output.open() as f:
        for line in f:
            identifier, sequence = line.rstrip("\n").split("\t")
            if identifier in sequences:
                raise ValueError("duplicate BEDTools output identifier")
            sequences[identifier] = sequence.upper()
    if sequences.keys() != {i for i, _ in records}:
        raise ValueError("BEDTools did not return exactly one sequence per interval")
    return sequences


def sample_negatives(args, config, positives, positive_gc, sizes, forbidden, allowed, temporary):
    design = config["negatives"]
    length, max_n = config["window_length_nt"], config["ambiguous_base_fraction_max"]
    rounds = args.max_rounds if args.max_rounds is not None else design["max_rounds"]
    per_round = design["candidates_per_round"]
    if rounds < 1 or per_round < 1:
        raise ValueError("candidate search budgets must be positive")
    accepted, selected, selection_order = {}, MutableIntervalIndex(), []
    for phase in ("train", "test"):
        ratio = design[f"{phase}_per_positive"]
        unresolved = {(r["record_id"], i): r for r in sorted(positives, key=lambda r: r["record_id"])
                      if r["split"] == phase for i in range(1, ratio + 1)}
        for round_number in range(rounds):
            if not unresolved:
                break
            candidates, metadata = [], {}
            for key, row in sorted(unresolved.items()):
                maximum = sizes[row["chrom"]] - length
                if maximum < 0:
                    raise ValueError("chromosome shorter than window")
                for offset in range(per_round):
                    attempt = round_number * per_round + offset
                    start = deterministic_start(design["seed"], f"{key[0]}|negative={key[1]}", attempt, maximum)
                    region = Region(row["chrom"], start, start + length)
                    if forbidden.overlaps(region) or selected.overlaps(region) or (allowed is not None and not allowed.contains(region)):
                        continue
                    identifier = f"{phase}_candidate_{len(candidates)}"
                    candidates.append((identifier, region))
                    metadata[identifier] = key, attempt
            if not candidates:
                continue
            regions = dict(candidates)
            for identifier, sequence in extract_sequences(args.bedtools, args.genome, candidates, temporary).items():
                key, attempt = metadata[identifier]
                if key not in unresolved or len(sequence) != length or n_fraction(sequence) > max_n:
                    continue
                if abs(gc_fraction(sequence) - positive_gc[key[0]]) > design["gc_tolerance"] + 1e-12:
                    continue
                region = regions[identifier]
                if selected.overlaps(region):
                    continue
                accepted[key] = region, sequence, attempt
                selection_order.append((phase, key))
                selected.add(region)
                del unresolved[key]
        if unresolved:
            examples = ", ".join(
                f"{key[0]} (GC={positive_gc[key[0]]:.3f})"
                for key in list(sorted(unresolved))[:5]
            )
            raise RuntimeError(
                f"failed to match {len(unresolved)} {phase} negatives after {rounds} rounds "
                f"with gc_tolerance={design['gc_tolerance']:.3f}; examples: {examples}. "
                "Increase negatives.gc_tolerance explicitly and rerun stage 03."
            )
    return accepted, selection_order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, default=ROOT / "manifests/splits/ctcf_train_test.tsv")
    p.add_argument("--genome", type=Path, default=ROOT / "genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    p.add_argument("--chrom-sizes", type=Path, default=ROOT / "genome/GRCh38.chrom.sizes")
    p.add_argument("--peaks", type=Path, default=ROOT / "results/preprocessing/peaks/ctcf_peaks.narrowPeak")
    p.add_argument("--blacklist", type=Path, required=True)
    p.add_argument("--mappability-bed", type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "config/robustness.json")
    p.add_argument("--cohort", help="Select the declared cohort-specific negative-matching limits")
    p.add_argument("--max-rounds", type=int, help="Extend candidate budget without changing its deterministic stream")
    p.add_argument("--output-dir", type=Path, default=ROOT / "manifests/negative_pairs")
    p.add_argument("--fasta-dir", type=Path, default=ROOT / "results/robustness/datasets")
    p.add_argument("--bedtools", default="bedtools")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    design = config["negatives"]
    if args.cohort:
        declared = design.get("cohorts", {}).get(args.cohort)
        if declared is None:
            p.error(f"missing negative-matching settings for cohort: {args.cohort}")
        design = dict(design, **declared)
        config = dict(config, negatives=design)
    length, max_n = config["window_length_nt"], config["ambiguous_base_fraction_max"]
    if length < 1 or not 0 <= max_n < 1 or not 0 <= design["gc_tolerance"] <= 1:
        p.error("invalid window or composition limits")
    if (design["train_per_positive"], design["test_per_positive"]) != (1, 10):
        p.error("this evaluation design requires train 1:1 and test 1:10; changing prevalence needs a new protocol")
    positives = read_tsv(args.split)
    if {r["split"] for r in positives} != {"train", "test"} or len({r["record_id"] for r in positives}) != len(positives):
        raise ValueError("split must contain unique IDs and nonempty train/test partitions")
    regions = [Region(r["chrom"], int(r["start"]), int(r["end"])) for r in positives]
    if any(r.end - r.start != length for r in regions):
        raise ValueError("positive interval length mismatch")
    train_regions = IntervalIndex([r for row, r in zip(positives, regions) if row["split"] == "train"])
    if any(train_regions.overlaps(r) for row, r in zip(positives, regions) if row["split"] == "test"):
        raise ValueError("positive train/test intervals overlap; rebuild the grouped split")
    sizes = read_chrom_sizes(args.chrom_sizes)
    normalization = {}
    def normalized(path):
        values, renamed, skipped = normalize_regions_to_chrom_sizes(read_bed_regions(path), sizes)
        normalization[path.name] = {"renamed": renamed, "skipped": skipped}
        if not values:
            raise ValueError(f"no reference-compatible intervals: {path}")
        return values
    forbidden = IntervalIndex(regions + normalized(args.peaks) + normalized(args.blacklist))
    allowed = IntervalIndex(normalized(args.mappability_bed)) if args.mappability_bed else None
    if allowed is not None and any(not allowed.contains(r) for r in regions):
        raise ValueError("positive windows are outside supplied mappability regions; rebuild a declared cohort")
    if args.output_dir.exists() or args.fasta_dir.exists():
        raise FileExistsError("use new manifest and FASTA output directories")
    with tempfile.TemporaryDirectory(prefix="ctcf-negatives-") as tmp:
        temporary = Path(tmp)
        sequences = extract_sequences(args.bedtools, args.genome, [(r["record_id"], region) for r, region in zip(positives, regions)], temporary)
        if any(len(s) != length or n_fraction(s) > max_n for s in sequences.values()):
            raise ValueError("positive sequences fail length/ambiguous-base filter")
        positive_gc = {i: gc_fraction(s) for i, s in sequences.items()}
        accepted, selection_order = sample_negatives(args, config, positives, positive_gc, sizes, forbidden, allowed, temporary)
    pairs, fastas = [], defaultdict(list)
    for row in sorted(positives, key=lambda r: (r["split"], r["record_id"])):
        phase, pos_id = row["split"], row["record_id"]
        fastas[phase, "positives"].append((pos_id, sequences[pos_id]))
        for index in range(1, design[f"{phase}_per_positive"] + 1):
            region, sequence, attempt = accepted[pos_id, index]
            neg_id = f"neg_{phase}_{hashlib.sha256(f'{pos_id}|{index}'.encode()).hexdigest()[:16]}"
            fastas[phase, "negatives"].append((neg_id, sequence))
            pairs.append({"split": phase, "positive_id": pos_id, "positive_chrom": row["chrom"], "positive_start": row["start"],
                "positive_end": row["end"], "positive_gc": f"{positive_gc[pos_id]:.8f}", "negative_index": index, "negative_id": neg_id,
                "negative_chrom": region.chrom, "negative_start": region.start, "negative_end": region.end, "negative_gc": f"{gc_fraction(sequence):.8f}",
                "gc_delta": f"{abs(gc_fraction(sequence)-positive_gc[pos_id]):.8f}", "candidate_attempt": attempt})
    manifest = args.output_dir / "ctcf_negative_pairs.tsv"
    atomic_tsv(manifest, pairs, list(pairs[0]))
    for (phase, kind), records in sorted(fastas.items()):
        path = args.fasta_dir / f"{phase}_{kind}.fa"
        atomic_write(path, lambda f, records=records: f.writelines(f">{i}\n{s}\n" for i, s in records))
    sources = [args.split, args.genome, args.chrom_sizes, args.peaks, args.blacklist, args.config]
    if args.mappability_bed:
        sources.append(args.mappability_bed)
    train_order = [index for index, (phase, _) in enumerate(selection_order) if phase == "train"]
    test_order = [index for index, (phase, _) in enumerate(selection_order) if phase == "test"]
    train_negative_regions = [accepted[row["record_id"], i][0] for row in positives if row["split"] == "train"
                              for i in range(1, design["train_per_positive"] + 1)]
    test_negative_regions = [accepted[row["record_id"], i][0] for row in positives if row["split"] == "test"
                             for i in range(1, design["test_per_positive"] + 1)]
    train_negative_index = IntervalIndex(train_negative_regions)
    cross_overlaps = sum(train_negative_index.overlaps(region) for region in test_negative_regions)
    summary = {"inputs": [str(path.resolve()) for path in sources], "normalization": normalization,
        "cohort": args.cohort,
        "seed": design["seed"], "window_length_nt": length, "gc_tolerance": design["gc_tolerance"], "max_ambiguous_fraction": max_n,
        "mappability_limitation": args.mappability_bed is None, "negative_selection_order": ["train", "test"],
        "test_can_influence_training_negative_selection": (not train_order or not test_order or min(test_order) < max(train_order)),
        "cross_split_negative_overlaps": cross_overlaps,
        "maximum_rounds": args.max_rounds if args.max_rounds is not None else design["max_rounds"], "candidates_per_round": design["candidates_per_round"],
        "counts": {phase: {kind: len(fastas[phase, kind]) for kind in ("positives", "negatives")} for phase in ("train", "test")},
        "manifest": str(manifest.resolve()), "fasta_directory": str(args.fasta_dir.resolve())}
    atomic_json(args.output_dir / "ctcf_negative_pairs.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
