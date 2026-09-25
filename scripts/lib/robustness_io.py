#!/usr/bin/env python3
"""Deterministic and atomic I/O helpers for robustness experiments."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO


def atomic_write(path: Path, writer: Callable[[TextIO], None], overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: Any, overwrite: bool = False) -> None:
    def write(handle: TextIO) -> None:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")

    atomic_write(path, write, overwrite=overwrite)


def atomic_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str], overwrite: bool = False) -> None:
    def write(handle: TextIO) -> None:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(path, write, overwrite=overwrite)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
