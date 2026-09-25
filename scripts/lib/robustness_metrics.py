#!/usr/bin/env python3
"""Dependency-free binary metrics and paired robustness statistics."""
# Statistical library: the caller supplies 0/1 labels. There is no CTCF concept
# here. Region AP and diagnostic AP are not interchangeable.

from __future__ import annotations

import itertools
import math
import random


def _validate(labels: list[int], scores: list[float]) -> tuple[int, int]:
    # Without both classes, benchmark AP/AUROC comparisons are not informative.
    # Do not replace missing data with zero or 0.5.
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and have equal length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("both classes are required")
    return positives, negatives


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Area under the right-continuous precision-recall staircase."""
    positives, _ = _validate(labels, scores)
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    prior_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        # Empates entram juntos; ordenar arbitrariamente seus labels inflaria AP.
        score = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1]:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - prior_recall) * precision
        prior_recall = recall
    return area


def auroc(labels: list[int], scores: list[float]) -> float:
    # Rank sums equal the probability that a positive outranks a negative;
    # Ties receive half credit. The metric does not require choosing a threshold.
    positives, negatives = _validate(labels, scores)
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (rank + (rank + end - index - 1)) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        rank += end - index
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def trapezoid_auc(xs: list[float], ys: list[float]) -> float:
    # Integrate AP versus nominal positive fraction. This is robustness AURC, not
    # a risk-coverage curve, although that other metric uses the same acronym.
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("at least two paired coordinates are required")
    points = sorted(zip(xs, ys))
    if len({x for x, _ in points}) != len(points):
        raise ValueError("x coordinates must be unique")
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )


def exact_sign_flip_test(differences: list[float]) -> dict[str, float | int]:
    """Two-sided exact paired randomization test over all sign assignments."""
    # Enumerate 2^n sign changes of paired differences. Interpretation depends on
    # exchangeability/symmetry under the null; this does not prove equivalence.
    if not differences or any(not math.isfinite(value) for value in differences):
        raise ValueError("finite paired differences are required")
    observed = abs(sum(differences) / len(differences))
    extreme = 0
    total = 1 << len(differences)
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(sum(sign * value for sign, value in zip(signs, differences)) / len(differences))
        extreme += statistic + tolerance >= observed
    return {
        "n_pairs": len(differences),
        "observed_mean_difference": sum(differences) / len(differences),
        "extreme_assignments": extreme,
        "total_assignments": total,
        "p_value_two_sided": extreme / total,
    }


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    # Linear interpolation between neighboring positions of a SORTED sample.
    if not sorted_values or not 0 <= probability <= 1:
        raise ValueError("a non-empty sample and probability in [0, 1] are required")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_mean_ci(
    differences: list[float],
    *,
    seed: int,
    replicates: int = 100_000,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Percentile interval for a paired mean, resampling pairs as intact units."""
    if not differences or any(not math.isfinite(value) for value in differences):
        raise ValueError("finite paired differences are required")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    # Resample trajectory differences, not individual curve points.
    # More bootstrap reduces Monte Carlo error; it does not add biological data.
    sample_size = len(differences)
    means = sorted(
        sum(differences[rng.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(replicates)
    )
    alpha = 1 - confidence
    return {
        "method": "paired percentile bootstrap of trajectory-level mean differences",
        "n_pairs": sample_size,
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "observed_mean_difference": sum(differences) / sample_size,
        "lower": _linear_quantile(means, alpha / 2),
        "upper": _linear_quantile(means, 1 - alpha / 2),
    }
