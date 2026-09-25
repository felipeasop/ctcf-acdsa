#!/usr/bin/env python3
"""Deterministic, overlap-safe train/test splitting for genomic intervals."""
# Stage-08 library: split window groups rather than sampling individual rows.
# Windows sharing bases remain on the same side of the split.
# This does not prevent homology between distant regions or sharing across cells.

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from genomic_intervals import IntervalIndex, Region
from robustness_io import atomic_json, atomic_tsv


@dataclass(frozen=True)
class Interval:
    # Preserve ID, source, and score so each window can be traced in the final manifest.
    record_id: str
    peak_id: str
    chrom: str
    start: int
    end: int
    score: float
    source_line: int


@dataclass(frozen=True)
class OverlapGroup:
    # Connected component: A overlaps B and B overlaps C, so all share one group.
    group_id: str
    chrom: str
    intervals: tuple[Interval, ...]
    score: float


def read_bed(path: Path) -> list[Interval]:
    # Read the window BED produced by stage 08; field 5 contains the selected score.
    intervals: list[Interval] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip() or raw_line.startswith(('#', 'track', 'browser')):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"line {line_number}: expected at least 3 BED columns")
            chrom = fields[0].strip()
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as error:
                raise ValueError(f"line {line_number}: non-integer BED coordinates") from error
            if not chrom or start < 0 or end <= start:
                raise ValueError(
                    f"line {line_number}: invalid interval {chrom}:{start}-{end}"
                )
            peak_id = fields[3].strip() if len(fields) >= 4 and fields[3].strip() else "."
            try:
                score = float(fields[4]) if len(fields) >= 5 else 0.0
            except ValueError as error:
                raise ValueError(f"line {line_number}: non-numeric BED score") from error
            if not math.isfinite(score):
                raise ValueError(f"line {line_number}: non-finite BED score")
            record_id = f"{peak_id}|line={line_number}"
            intervals.append(
                Interval(record_id, peak_id, chrom, start, end, score, line_number)
            )
    if len(intervals) < 2:
        raise ValueError("at least two BED intervals are required")
    return intervals


def _group_id(intervals: list[Interval]) -> str:
    # Deterministic coordinate identifier; it does not certify file integrity.
    payload = "\n".join(
        f"{item.chrom}\t{item.start}\t{item.end}\t{item.record_id}"
        for item in intervals
    )
    return "grp_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def group_overlaps(intervals: list[Interval]) -> list[OverlapGroup]:
    # The ordered scan keeps the group's largest end, capturing transitive overlap.
    # The internal finish function materializes a group when the chain ends.
    ordered = sorted(
        intervals, key=lambda item: (item.chrom, item.start, item.end, item.record_id)
    )
    grouped: list[OverlapGroup] = []
    current: list[Interval] = []
    current_chrom = ""
    maximum_end = -1

    def finish() -> None:
        if current:
            grouped.append(
                OverlapGroup(
                    _group_id(current),
                    current[0].chrom,
                    tuple(current),
                    float(statistics.median(item.score for item in current)),
                )
            )

    for item in ordered:
        if current and item.chrom == current_chrom and item.start < maximum_end:
            current.append(item)
            maximum_end = max(maximum_end, item.end)
        else:
            finish()
            current = [item]
            current_chrom = item.chrom
            maximum_end = item.end
    finish()
    return grouped


def _score_bins(groups: list[OverlapGroup], bins: int) -> dict[str, int]:
    # Score-ranked strata keep peak-confidence distributions similar across
    # partitions. They do not use motifs or MEME results.
    if bins < 1:
        raise ValueError("score_bins must be at least 1")
    ordered = sorted(groups, key=lambda group: (group.score, group.group_id))
    return {
        group.group_id: min(bins - 1, rank * bins // len(ordered))
        for rank, group in enumerate(ordered)
    }


def assign_groups(
    groups: list[OverlapGroup], test_fraction: float, seed: int, score_bins: int
) -> dict[str, str]:
    # Allocate the requested test fraction of GROUPS by chromosome/stratum. Since
    # groups have different sizes, the window fraction will not be exactly 20%.
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if len(groups) < 2:
        raise ValueError("at least two non-overlapping groups are required")

    bins = _score_bins(groups, score_bins)
    strata: dict[tuple[str, int], list[OverlapGroup]] = defaultdict(list)
    for group in groups:
        strata[(group.chrom, bins[group.group_id])].append(group)

    target = min(
        len(groups) - 1,
        max(1, math.floor(len(groups) * test_fraction + 0.5)),
    )
    allocation: dict[tuple[str, int], int] = {}
    remainders: list[tuple[float, tuple[str, int]]] = []
    for key, members in strata.items():
        exact = len(members) * test_fraction
        allocation[key] = math.floor(exact)
        remainders.append((exact - allocation[key], key))

    remaining = target - sum(allocation.values())
    # Distribute rounding remainders among the largest fractional parts without
    # groups than the requested total. Hash-plus-seed ranking is reproducible.
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocation[key] += 1

    assignments = {group.group_id: "train" for group in groups}
    for key, members in strata.items():
        ranked = sorted(
            members,
            key=lambda group: hashlib.sha256(
                f"{seed}\0{group.group_id}".encode()
            ).hexdigest(),
        )
        for group in ranked[: allocation[key]]:
            assignments[group.group_id] = "test"
    return assignments


def validate_no_cross_split_overlap(
    groups: list[OverlapGroup], assignments: dict[str, str]
) -> None:
    # Validity check: sharing bases between
    # train and test would inflate evaluation on held-out sequences.
    train = IntervalIndex([
        Region(item.chrom, item.start, item.end)
        for group in groups if assignments[group.group_id] == "train"
        for item in group.intervals
    ])
    for group in groups:
        if assignments[group.group_id] == "test":
            for item in group.intervals:
                if train.overlaps(Region(item.chrom, item.start, item.end)):
                    raise AssertionError(f"cross-split overlap: {item.record_id}")


def write_split(
    source: Path,
    manifest: Path,
    summary: Path,
    test_fraction: float,
    seed: int,
    score_bins: int,
) -> dict[str, object]:
    # Public function: read -> group -> split -> validate -> save the manifest
    # and counts. No discovery tool is called here.
    intervals = read_bed(source)
    groups = group_overlaps(intervals)
    assignments = assign_groups(groups, test_fraction, seed, score_bins)
    validate_no_cross_split_overlap(groups, assignments)

    rows = [
        {
            "split": assignments[group.group_id],
            "group_id": group.group_id,
            "record_id": item.record_id,
            "peak_id": item.peak_id,
            "chrom": item.chrom,
            "start": item.start,
            "end": item.end,
            "score": f"{item.score:.12g}",
            "source_line": item.source_line,
        }
        for group in groups
        for item in group.intervals
    ]
    rows.sort(key=lambda row: (row["split"], row["chrom"], row["start"], row["end"]))

    split_group_counts = {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "test")
    }
    split_interval_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "test")
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "input": str(source.resolve()),
        "seed": seed,
        "test_fraction_requested": test_fraction,
        "score_bins": score_bins,
        "groups": len(groups),
        "intervals": len(intervals),
        "group_counts": split_group_counts,
        "interval_counts": split_interval_counts,
        "cross_split_overlaps": 0,
    }

    atomic_tsv(manifest, rows, list(rows[0]))
    atomic_json(summary, report)
    return report
