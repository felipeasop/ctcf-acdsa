#!/usr/bin/env python3
"""Parse MEME XML and score every sequence with a PWM on both strands."""

from __future__ import annotations

import math
import itertools
from decimal import Decimal
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from fasta_utils import read_fasta

BASES = "ACGT"


@dataclass(frozen=True)
class MemeMotif:
    # Immutable motif container. The E-value string and -log10(E) preserve
    # information that a floating-point value may round to zero.
    matrix: tuple[tuple[float, float, float, float], ...]
    sites: float
    e_value: float
    width: int
    meme_seed: int
    command_line: str
    primary_count: int
    e_value_text: str
    neg_log10_evalue: float


def parse_meme_xml(path: Path) -> MemeMotif:
    root = ET.parse(path).getroot()
    motif = root.find("./motifs/motif")
    if motif is None:
        raise ValueError(f"no motif in MEME XML: {path}")
    rows: list[tuple[float, float, float, float]] = []
    for array in motif.findall("./probabilities/alphabet_matrix/alphabet_array"):
        values = {value.attrib["letter_id"]: float(value.text or "nan") for value in array.findall("value")}
        if any(base not in values for base in BASES):
            raise ValueError(f"incomplete DNA PWM row in {path}")
        row = tuple(values[base] for base in BASES)
        if any(value < 0 or not math.isfinite(value) for value in row):
            raise ValueError(f"negative or non-finite PWM probability in {path}")
        if not math.isclose(sum(row), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"PWM row does not sum to one in {path}")
        rows.append(row)  # type: ignore[arg-type]
    width = int(motif.attrib["width"])
    if len(rows) != width:
        raise ValueError(f"PWM width mismatch in {path}")
    seed_text = root.findtext("./model/seed")
    command_line = (root.findtext("./model/command_line") or "").strip()
    if seed_text is None:
        raise ValueError(f"MEME seed absent from {path}")
    evalue_text = motif.attrib["e_value"]
    evalue = Decimal(evalue_text)
    if not evalue.is_finite() or evalue <= 0:
        raise ValueError(f"MEME E-value must be positive; literal zero cannot be ranked: {path}")
    return MemeMotif(
        tuple(rows),
        float(motif.attrib["sites"]),
        float(motif.attrib["e_value"]),
        width,
        int(seed_text),
        command_line,
        int(root.find("./training_set").attrib["primary_count"]),
        evalue_text,
        float(-evalue.log10()),
    )


def parse_markov(path: Path, order: int = 2) -> dict[str, float]:
    frequencies: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.split()
            if len(fields) != 2 or fields[0].startswith("#"):
                continue
            word = fields[0].upper()
            if 1 <= len(word) <= order + 1 and set(word) <= set(BASES):
                frequencies[word] = float(fields[1])
    expected = sum(4**length for length in range(1, order + 2))
    if len(frequencies) != expected or any(not math.isfinite(value) or value <= 0 for value in frequencies.values()):
        raise ValueError(f"incomplete or zero-valued order-{order} background: {path}")
    return frequencies


def markov_log_probability(word: str, frequencies: dict[str, float], order: int) -> float:
    if not 0 <= order <= 5:
        raise ValueError("supported background orders: 0 through 5")
    word = word.upper()
    if not word or set(word) - set(BASES):
        raise ValueError("Markov probability requires an unambiguous DNA word")
    prefix_length = min(order + 1, len(word))
    value = math.log(frequencies[word[:prefix_length]])
    for index in range(prefix_length, len(word)):
        context = word[index - order : index]
        value += math.log(frequencies[context + word[index]])
        if order:
            value -= math.log(frequencies[context])
    return value


def _pwm_log_probability(word: str, matrix: tuple[tuple[float, float, float, float], ...]) -> float:
    probabilities = (matrix[index][BASES.index(base)] for index, base in enumerate(word))
    total = 0.0
    for probability in probabilities:
        if probability == 0:
            return -math.inf
        total += math.log(probability)
    return total


def reverse_complement_matrix(
    matrix: tuple[tuple[float, float, float, float], ...]
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple((row[3], row[2], row[1], row[0]) for row in reversed(matrix))


def apply_pwm_pseudocount(
    matrix: tuple[tuple[float, float, float, float], ...],
    sites: float,
    pseudocount_per_base: float,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return a normalized PWM after adding a fixed count to every base."""
    if not math.isfinite(sites) or sites <= 0:
        raise ValueError("motif site count must be finite and positive")
    if not math.isfinite(pseudocount_per_base) or pseudocount_per_base < 0:
        raise ValueError("PWM pseudocount must be finite and non-negative")
    denominator = sites + 4 * pseudocount_per_base
    return tuple(
        tuple((probability * sites + pseudocount_per_base) / denominator for probability in row)
        for row in matrix
    )  # type: ignore[return-value]


def max_log_odds(
    sequence: str,
    matrix: tuple[tuple[float, float, float, float], ...],
    background: dict[str, float],
    order: int = 2,
) -> float:
    sequence = sequence.upper()
    width = len(matrix)
    if len(sequence) < width:
        raise ValueError("sequence shorter than PWM")
    reverse = reverse_complement_matrix(matrix)
    best = -math.inf
    valid = False
    for start in range(len(sequence) - width + 1):
        word = sequence[start : start + width]
        if set(word) - set(BASES):
            continue
        denominator = markov_log_probability(word, background, order)
        best = max(
            best,
            _pwm_log_probability(word, matrix) - denominator,
            _pwm_log_probability(word, reverse) - denominator,
        )
        valid = True
    if not valid:
        raise ValueError("sequence has no unambiguous PWM-width window")
    if not math.isfinite(best):
        raise ValueError("sequence has no finite PWM score")
    return best


def score_fasta(
    fasta: Path,
    matrix: tuple[tuple[float, float, float, float], ...],
    background: dict[str, float],
    order: int = 2,
) -> list[tuple[str, float]]:
    return [
        (header[1:].split()[0], max_log_odds(sequence, matrix, background, order))
        for header, sequence in read_fasta(fasta)
    ]


def score_fasta_numpy(
    fasta: Path,
    matrix: tuple[tuple[float, float, float, float], ...],
    background: dict[str, float],
    order: int = 2,
    batch_size: int = 2048,
) -> list[tuple[str, float]]:
    """Vectorized equivalent of :func:`score_fasta` for large held-out sets."""
    import numpy as np

    records = [(header[1:].split()[0], sequence.upper()) for header, sequence in read_fasta(fasta)]
    lengths = {len(sequence) for _, sequence in records}
    if len(lengths) != 1:
        raise ValueError("vectorized scanner requires equal-length sequences")
    sequence_length = next(iter(lengths))
    width = len(matrix)
    if sequence_length < width:
        raise ValueError("sequence shorter than PWM")

    lookup = np.full(256, -1, dtype=np.int8)
    for index, base in enumerate(BASES.encode()):
        lookup[base] = index
    with np.errstate(divide="ignore"):
        forward = np.log(np.asarray(matrix, dtype=np.float64))
        reverse = np.log(np.asarray(reverse_complement_matrix(matrix), dtype=np.float64))
    if not 0 <= order <= 5:
        raise ValueError("supported background orders: 0 through 5")
    logs = {length: np.asarray([math.log(background["".join(word)])
            for word in itertools.product(BASES, repeat=length)])
            for length in range(1, order + 2)}

    results: list[tuple[str, float]] = []
    window_count = sequence_length - width + 1
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        raw = np.frombuffer("".join(sequence for _, sequence in batch).encode(), dtype=np.uint8)
        encoded = lookup[raw].reshape(len(batch), sequence_length)
        valid = np.ones((len(batch), window_count), dtype=bool)
        numerator_forward = np.zeros((len(batch), window_count), dtype=np.float64)
        numerator_reverse = np.zeros_like(numerator_forward)
        for position in range(width):
            bases = encoded[:, position : position + window_count]
            valid &= bases >= 0
            safe = np.maximum(bases, 0)
            numerator_forward += forward[position, safe]
            numerator_reverse += reverse[position, safe]
        def indices(start, length):
            code = np.zeros((len(batch), window_count), dtype=np.int32)
            for position in range(start, start + length):
                code = 4 * code + np.maximum(encoded[:, position:position + window_count], 0)
            return code

        prefix = min(order + 1, width)
        denominator = logs[prefix][indices(0, prefix)]
        for position in range(prefix, width):
            transition = logs[order + 1][indices(position - order, order + 1)]
            if order:
                transition = transition - logs[order][indices(position - order, order)]
            denominator += transition
        values = np.maximum(numerator_forward, numerator_reverse) - denominator
        values[~valid] = -np.inf
        maxima = values.max(axis=1)
        if not np.isfinite(maxima).all():
            raise ValueError("at least one sequence has no finite PWM score")
        results.extend((identifier, float(score)) for (identifier, _), score in zip(batch, maxima))
    return results
