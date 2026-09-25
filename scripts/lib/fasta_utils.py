#!/usr/bin/env python3
"""Read nonempty FASTA records with unique first-token identifiers."""
# Library, not a pipeline stage: convert FASTA headers and sequence lines into
# records. Callers decide how ambiguous bases are handled.

from collections.abc import Iterator
from pathlib import Path


def read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    # Yield one record at a time; retain seen IDs to prevent ambiguous joins.
    header: str | None = None
    chunks: list[str] = []
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    # A new header closes the previous sequence.
                    if not chunks:
                        raise ValueError(f"empty FASTA sequence: {header} in {path}")
                    yield header, "".join(chunks)
                identifier = line[1:].split()
                if not identifier or identifier[0] in identifiers:
                    raise ValueError(f"missing or duplicate FASTA identifier at {path}:{line_number}")
                identifiers.add(identifier[0])
                header, chunks = line, []
            elif header is None:
                raise ValueError(f"sequence before header at line {line_number}: {path}")
            else:
                chunks.append(line)
    if header is not None:
        # The final record has no following header and requires this flush.
        if not chunks:
            raise ValueError(f"empty FASTA sequence: {header} in {path}")
        yield header, "".join(chunks)
    else:
        raise ValueError(f"empty FASTA file: {path}")
