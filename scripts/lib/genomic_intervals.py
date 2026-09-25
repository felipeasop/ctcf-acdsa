#!/usr/bin/env python3
"""Small interval indexes used to enforce genomic disjointness."""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class Region:
    # Immutable object: invalid regions are rejected at construction time.
    chrom: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.chrom or self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid region: {self.chrom}:{self.start}-{self.end}")


class IntervalIndex:
    """Merged half-open intervals with logarithmic overlap queries."""

    def __init__(self, regions: list[Region] | None = None) -> None:
        by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._merged: dict[str, list[tuple[int, int]]] = {}
        self._starts: dict[str, list[int]] = {}
        for region in regions or []:
            by_chrom[region.chrom].append((region.start, region.end))
        for chrom, intervals in by_chrom.items():
            merged: list[list[int]] = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            frozen = [(start, end) for start, end in merged]
            self._merged[chrom] = frozen
            self._starts[chrom] = [start for start, _ in frozen]

    def overlaps(self, region: Region) -> bool:
        intervals = self._merged.get(region.chrom, [])
        starts = self._starts.get(region.chrom, [])
        index = bisect.bisect_left(starts, region.end) - 1
        return index >= 0 and intervals[index][1] > region.start

    def contains(self, region: Region) -> bool:
        intervals = self._merged.get(region.chrom, [])
        starts = self._starts.get(region.chrom, [])
        index = bisect.bisect_right(starts, region.start) - 1
        return index >= 0 and intervals[index][1] >= region.end


class MutableIntervalIndex:
    """Bucketed index for efficient incremental insertion and overlap checks."""

    def __init__(self, bucket_size: int = 4096) -> None:
        if bucket_size < 1:
            raise ValueError("bucket_size must be positive")
        self.bucket_size = bucket_size
        self._buckets: dict[tuple[str, int], list[Region]] = defaultdict(list)

    def _keys(self, region: Region):
        first = region.start // self.bucket_size
        last = (region.end - 1) // self.bucket_size
        for bucket in range(first, last + 1):
            yield region.chrom, bucket

    def overlaps(self, region: Region) -> bool:
        for key in self._keys(region):
            for existing in self._buckets.get(key, []):
                if existing.start < region.end and region.start < existing.end:
                    return True
        return False

    def add(self, region: Region) -> None:
        for key in self._keys(region):
            self._buckets[key].append(region)


def read_bed_regions(path: Path) -> list[Region]:
    regions: list[Region] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.startswith(("#", "track", "browser")):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: expected at least 3 BED columns")
            try:
                regions.append(Region(fields[0], int(fields[1]), int(fields[2])))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid BED interval") from error
    return regions


def read_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            fields = raw.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: invalid chromosome sizes row")
            size = int(fields[1])
            if size <= 0 or fields[0] in sizes:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate chromosome")
            sizes[fields[0]] = size
    return sizes


def chromosome_alias(chrom: str, chrom_sizes: dict[str, int]) -> str | None:
    """Resolve only common UCSC/Ensembl aliases, never assembly coordinates."""
    bare = chrom.removeprefix("chr")
    aliases = [chrom, bare, f"chr{bare}"]
    if bare in ("M", "MT"):
        aliases.extend(["MT", "chrM"])
    return next((name for name in aliases if name in chrom_sizes), None)


def normalize_regions_to_chrom_sizes(
    regions: list[Region], chrom_sizes: dict[str, int]
) -> tuple[list[Region], int, int]:
    """Map common UCSC/Ensembl chromosome aliases and reject out-of-range rows."""
    normalized: list[Region] = []
    renamed = 0
    skipped = 0
    for region in regions:
        target = chromosome_alias(region.chrom, chrom_sizes)
        if target is None or region.end > chrom_sizes[target]:
            skipped += 1
            continue
        renamed += target != region.chrom
        normalized.append(Region(target, region.start, region.end))
    return normalized, renamed, skipped
