#!/usr/bin/env python3
"""Measure OOPS/ZOOPS disagreement; fit on development or apply frozen diagnostics.

Reads completed evaluation outputs. Never runs MEME or changes source results.
The outcome is weak reference recovery, not demonstrated biological invalidity.
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from motif_scoring import parse_meme_xml, reverse_complement_matrix
from experiments import cohort_settings, reference_catalog
from robustness_io import atomic_json, atomic_tsv, read_tsv

BASELINE = ["weakest_log_confidence", "zoops_site_fraction", "resampling_instability"]
MODELS = {"internal": BASELINE[:2], "baseline": BASELINE,
          "baseline_plus_disagreement": BASELINE + ["disagreement"]}


def motif_distance(left, right, fraction):
    """Minimum penalized mean column JSD (base 2), considering both strands.

    Require at least fraction of each PWM to overlap. Every unmatched column
    contributes the maximum divergence (1); divide by the alignment union.
    Uniform columns are handled without correlations or numerical pseudocounts.
    """
    a = np.asarray(left, dtype=float)
    # Normalize small rounding differences in exported rows.
    # Each row is a distribution over A,C,G,T, not a consensus letter.
    a = a / a.sum(axis=1, keepdims=True)
    minimum = math.ceil(fraction * max(len(left), len(right)))
    # With width 19 and fraction 0.8, require at least 16 shared positions.
    # Penalizing unaligned positions is an operational choice, not a universal
    # measure of biological distance. Uniform versus uniform therefore gives zero.
    best = (float("inf"), 0, "+", 0)
    for strand, matrix in (("+", right), ("-", reverse_complement_matrix(right))):
        b = np.asarray(matrix, dtype=float)
        b = b / b.sum(axis=1, keepdims=True)
        for offset in range(-len(b) + minimum, len(a) - minimum + 1):
            start, end = max(0, offset), min(len(a), offset + len(b))
            x, y = a[start:end], b[start - offset:end - offset]
            middle = (x + y) / 2
            # JSD(P,Q) = [KL(P||M)+KL(Q||M)]/2, M=(P+Q)/2.
            # Sum only positive terms; by convention, 0*log(0)=0.
            divergence = 0.0
            for value in (x, y):
                positive = value > 0
                divergence += float(np.sum(value[positive] * np.log2(value[positive] / middle[positive]))) / 2
            overlap = end - start
            unmatched = len(a) + len(b) - 2 * overlap
            score = (divergence + unmatched) / (overlap + unmatched)
            if score < best[0]:
                best = (max(0.0, min(1.0, score)), offset, strand, overlap)
    if not math.isfinite(best[0]):
        raise ValueError("PWM widths cannot satisfy the required overlap")
    return best


def reference_result(result, settings):
    # Operational label: p<=0.05 and the target-specific minimum overlap.
    # This p-value is not the probability that the motif is wrong and is not a
    # global FDR across all runs. The name 'full' does not guarantee every module:
    # the preserved file has 17 positions. No match does not prove absence of CTCF.
    # Other references require a documented protocol.
    matches = [m for m in result["tomtom"] if m["Target_ID"] == settings["reference_id"]
               and int(m["Overlap"]) >= settings["reference_minimum_overlap"]]
    if not matches:
        return 1.0, 0, 1
    best = min(matches, key=lambda m: float(m["p-value"]))
    pvalue = float(best["p-value"])
    if not math.isfinite(pvalue) or not 0 <= pvalue <= 1:
        raise ValueError("invalid reference p-value")
    return pvalue, int(best["Overlap"]), int(pvalue > settings["reference_p_threshold"])


def paired_records(evaluation, design):
    """Require the complete planned grid and matched discovery/evaluation inputs."""
    # Require the complete grid so favorable runs cannot be selected selectively.
    # Use individual JSON files, including empty match lists, rather than only
    # the aggregate TOMTOM table, which may omit motifs without a match.
    metadata = json.loads((evaluation / "evaluation.json").read_text())
    if (metadata["order"], metadata["pseudocount"]) != (2, 0.5):
        raise ValueError("the protocol requires scoring order 2 and pseudocount 0.5")
    records = read_tsv(evaluation / "metrics.tsv")
    cohorts = {r["cohort"] for r in records}
    if len(cohorts) != 1:
        raise ValueError("one nonempty cohort per analysis")
    cohort = next(iter(cohorts))
    discovery, target_settings = cohort_settings(design, cohort)
    reference_path = Path(metadata["inputs"]["reference"])
    matrix_headers = [re.search(r"\bw\s*=\s*(\d+)", line) for line in reference_path.read_text().splitlines()]
    reference_widths = [int(match.group(1)) for match in matrix_headers if match]
    if not reference_widths:
        raise ValueError(f"reference has no MEME matrix width: {reference_path}")
    reference_width = max(reference_widths)
    if not 0 < target_settings["reference_minimum_overlap"] <= reference_width:
        raise ValueError("invalid target-specific reference overlap")
    cells = {}
    for row in records:
        result = json.loads((evaluation / row["evaluation_id"] / "result.json").read_text())
        if result["protocol"] != metadata["protocol"] or result["run_id"] != row["run_id"]:
            raise ValueError("inconsistent evaluation records")
        if int(result["order_repeat"]) != 0 or int(result["meme_repeat"]) != 0:
            raise ValueError("optimizer/order audits are not primary observations")
        key = (int(result["sample"]), float(result["signal_percent"]),
               int(result["input_repeat"]), result["model"])
        if key in cells:
            raise ValueError(f"duplicate cell: {key}")
        if result["motif_width"] != discovery["width"] or result["sample_size"] != design["trajectories"]["sample_size"]:
            raise ValueError("discovery dimensions differ from protocol")
        if result["background"] != "markov2":
            raise ValueError("background controls are not primary observations")
        motif = parse_meme_xml(Path(result["motif"]))
        if motif.meme_seed != int(result["meme_seed"]):
            raise ValueError("motif seed differs from evaluation record")
        cells[key] = result, motif
    d = design["trajectories"]
    expected = {(s, float(level), r, model) for s in range(1, d["samples"] + 1)
                for level in d["levels_percent"] for r in (0, 1) for model in ("oops", "zoops")}
    if set(cells) != expected or d["input_repeats"] != 2:
        raise ValueError("the protocol requires the full grid, including one companion input per anchor")
    pairs = []
    fraction = target_settings["minimum_overlap_fraction"]
    for sample in range(1, d["samples"] + 1):
        for level in sorted(d["levels_percent"]):
            anchor = {m: cells[sample, level, 0, m] for m in ("oops", "zoops")}
            companion = {m: cells[sample, level, 1, m] for m in ("oops", "zoops")}
            for group in (anchor, companion):
                if len({(r["input"], r["meme_seed"], r["discovery_background"]) for r, _ in group.values()}) != 1:
                    raise ValueError("OOPS/ZOOPS were not run on the same input, seed and background")
            if anchor["oops"][0]["input"] == companion["oops"][0]["input"]:
                raise ValueError("companion must be a separate input resample")
            if anchor["oops"][0]["meme_seed"] != companion["oops"][0]["meme_seed"]:
                raise ValueError("keep the MEME seed fixed when measuring input-resampling instability")
            distance, offset, strand, overlap = motif_distance(anchor["oops"][1].matrix, anchor["zoops"][1].matrix, fraction)
            row = {"cohort": cohort, "factor": target_settings["factor"], "sample": sample, "signal_percent": level,
                   "contamination_percent": 100 - level,
                   "disagreement": distance, "alignment_offset": offset, "alignment_strand": strand,
                   "alignment_overlap": overlap}
            for model in ("oops", "zoops"):
                result, motif = anchor[model]
                pvalue, matched, weak = reference_result(result, target_settings)
                row.update({f"{model}_reference_p": pvalue, f"{model}_reference_overlap": matched,
                            f"{model}_weak_reference": weak, f"{model}_ap": result["ap"],
                            f"{model}_auroc": result["auroc"],
                            f"{model}_neg_log10_evalue": motif.neg_log10_evalue,
                            f"{model}_resampling": motif_distance(motif.matrix, companion[model][1].matrix, fraction)[0]})
            row.update(weak_reference=int(row["oops_weak_reference"] or row["zoops_weak_reference"]),
                       # Pair event: at least one member fails to recover the
                       # reference. This does not identify which member is correct.
                       weakest_log_confidence=math.log1p(max(0, min(row["oops_neg_log10_evalue"], row["zoops_neg_log10_evalue"]))),
                       zoops_site_fraction=anchor["zoops"][0]["site_fraction"],
                       resampling_instability=max(row["oops_resampling"], row["zoops_resampling"]),
                       minimum_ap=min(row["oops_ap"], row["zoops_ap"]))
            pairs.append(row)
    reference = reference_path.read_text()
    reference_ids = {line.split()[1] for line in reference.splitlines() if line.startswith("MOTIF ")}
    if reference_ids != {target_settings["reference_id"]}:
        raise ValueError("use only the declared target reference; a different database changes the protocol")
    if reference != reference_catalog(design)[target_settings["factor"]]:
        raise ValueError("evaluation reference differs from the declared reference contents")
    return pairs, reference


def matrix(rows, features):
    # Convert only preselected indicators to a numeric matrix.
    # AP, dilution level, and TOMTOM are not predictors: TOMTOM supplies the
    # outcome, AP is complementary evidence, and dilution is a known intervention.
    values = np.asarray([[r[f] for f in features] for r in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite diagnostic features")
    return values


def fit_models(rows, design, reference):
    # Fit on development data only. Serialize coefficients and scaling in readable
    # JSON; validation does not need to load executable pickle objects.
    # C=1 regularization limits coefficients but does not eliminate overfitting risk.
    labels = np.asarray([r["weak_reference"] for r in rows])
    if len(set(labels)) != 2:
        raise ValueError("development has only one reference-outcome class; tables are saved, but a diagnostic cannot be fitted")
    models = {}
    for name, features in MODELS.items():
        x = matrix(rows, features)
        mean, scale = x.mean(axis=0), x.std(axis=0)
        # Standardization puts variables on comparable scales. A constant column
        # receives divisor 1; after centering it is zero without division by zero.
        scale[scale == 0] = 1
        model = LogisticRegression(C=design["reliability"]["logistic_C"], solver="lbfgs", max_iter=2000)
        model.fit((x - mean) / scale, labels)
        if int(model.n_iter_[0]) >= 2000:
            raise ValueError("logistic fit did not converge; do not use a partial fit")
        models[name] = {"features": features, "mean": mean.tolist(), "scale": scale.tolist(),
                        "coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
    # Include GABPA in the frozen artifact as well: its reference cannot be
    # selected or changed after observing the external test.
    return {"schema_version": 2, "design": design, "reference": reference,
            "references": reference_catalog(design), "models": models,
            "development_cohort": rows[0]["cohort"], "sklearn_version": sklearn.__version__}


def predictions(rows, frozen):
    # Reuse development means, scales, and coefficients; never recompute them in
    # external contexts. The sigmoid gives a [0,1] score, but we do not claim it
    # is a biologically calibrated probability.
    scores = {"disagreement": np.asarray([r["disagreement"] for r in rows]),
              "resampling": np.asarray([r["resampling_instability"] for r in rows])}
    for name, model in frozen["models"].items():
        x = (matrix(rows, model["features"]) - model["mean"]) / model["scale"]
        z = np.clip(x @ np.asarray(model["coefficients"]) + model["intercept"], -700, 700)
        scores[name] = 1 / (1 + np.exp(-z))
    return scores


def choose_threshold(labels, scores, limit):
    """Lowest threshold meeting the development false-alert constraint; ties intact."""
    # Empirical false-alert limit in development. It does NOT guarantee 10% on
    # external data. A threshold above the largest score alerts on none.
    thresholds = np.append(np.unique(scores), np.nextafter(max(scores), np.inf))
    return float(next(t for t in thresholds if np.mean(scores[labels == 0] >= t) <= limit))


def performance(labels, scores, threshold):
    # Here the positive class is 'weak reference recovery', not a positive ChIP
    # sequence. Diagnostic AP and region AP use different units. A single class
    # makes AP/AUROC comparisons undefined.
    prediction = scores >= threshold
    positives, negatives = int(sum(labels)), int(sum(labels == 0))
    tp = int(sum(prediction & (labels == 1)))
    fp = int(sum(prediction & (labels == 0)))
    both = positives > 0 and negatives > 0
    return {"n": len(labels), "weak_reference": positives, "prevalence": float(np.mean(labels)),
            "ap": float(average_precision_score(labels, scores)) if both else None,
            "auroc": float(roc_auc_score(labels, scores)) if both else None,
            "threshold": threshold, "true_alerts": tp, "false_alerts": fp,
            "missed_weak_reference": positives - tp, "unflagged_matches": negatives - fp,
            "sensitivity": tp / positives if positives else None,
            "false_positive_rate": fp / negatives if negatives else None,
            "alert_precision": tp / (tp + fp) if tp + fp else None}


def incremental_interval(rows, labels, scores, settings):
    """Conditional paired bootstrap: whole six-level blocks, frozen coefficients."""
    # Resample whole blocks, keeping their six dependent levels together.
    # Do not refit the model: the interval is conditional on the fit, reference,
    # and data pool. Blocks may share sequences; this is not biological uncertainty.
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["sample"]].append(index)
    blocks = list(groups.values())
    rng = np.random.default_rng(settings["bootstrap_seed"])
    deltas = []
    baseline, augmented = scores["baseline"], scores["baseline_plus_disagreement"]
    for _ in range(settings["bootstrap"]):
        indices = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        y = labels[indices]
        if len(set(y)) == 2:
            deltas.append(average_precision_score(y, augmented[indices]) - average_precision_score(y, baseline[indices]))
    both = len(set(labels)) == 2
    return {"delta_ap": float(average_precision_score(labels, augmented) - average_precision_score(labels, baseline)) if both else None,
            "ci95": np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None,
            "requested": settings["bootstrap"], "usable": len(deltas), "blocks": len(blocks),
            "interpretation": "Conditional on this experiment, shared data pools and frozen development fit; not biological replication or a population CI."}


def main():
    # Ordem deliberada: tabular -> opcionalmente ajustar OU aplicar congelado.
    # A development set with one class saves the table but does not fabricate a fit.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/design.json")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fit", action="store_true", help="Fit only on the declared development cohort")
    mode.add_argument("--frozen", type=Path, help="Apply development coefficients/thresholds without refitting")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    design = json.loads(args.config.read_text())
    reliability_config = design["reliability"]
    # Parameter bounds prevent accidental JSON edits from producing a degenerate
    # distance, an empty bootstrap, or uninterpretable thresholds.
    if (not 0 < reliability_config["minimum_overlap_fraction"] <= 1
            or not 0 < reliability_config["reference_p_threshold"] < 1
            or not 0 <= reliability_config["development_false_positive_limit"] < 1
            or reliability_config["logistic_C"] <= 0 or reliability_config["bootstrap"] < 1):
        p.error("invalid reliability settings")
    rows, reference = paired_records(args.evaluation, design)
    cohort = rows[0]["cohort"]
    _, target_settings = cohort_settings(design, cohort)
    if args.fit and cohort != design["development_cohort"]:
        p.error("fit is restricted to the declared development cohort")
    if args.frozen and cohort not in design["validation_cohorts"]:
        p.error("frozen diagnostics require a declared external cohort")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_tsv(args.output / "pairs.tsv", rows, list(rows[0]))
    atomic_json(args.output / "protocol.json", {"design": design, "reference": reference,
        "source": str(args.evaluation.resolve()), "cohort": cohort,
        "outcome": {"event": "at least one anchor motif lacks qualifying reference match; not molecular ground truth",
                     "reference": target_settings["reference_id"], "p_threshold": target_settings["reference_p_threshold"],
                     "minimum_overlap": target_settings["reference_minimum_overlap"]}})
    if not args.fit and not args.frozen:
        print("Pair table saved. No diagnostic fitted or validated.")
        return
    labels = np.asarray([r["weak_reference"] for r in rows])
    if args.fit and len(set(labels)) != 2:
        report = {"cohort": cohort, "role": "development_not_fitted",
                  "status": "one_outcome_class",
                  "n": len(labels), "weak_reference": int(sum(labels))}
        atomic_json(args.output / "summary.json", report)
        print(json.dumps(report, indent=2))
        return
    frozen = fit_models(rows, design, reference) if args.fit else json.loads(args.frozen.read_text())
    if frozen["design"] != design or frozen.get("references") != reference_catalog(design):
        raise ValueError("validation protocol/reference differs from the frozen development artifact")
    scores = predictions(rows, frozen)
    if args.fit:
        frozen["thresholds"] = {name: choose_threshold(labels, value, design["reliability"]["development_false_positive_limit"])
                                for name, value in scores.items()}
        frozen["development_evaluation"] = str(args.evaluation.resolve())
        atomic_json(args.output / "frozen_diagnostic.json", frozen)
    report = {"cohort": cohort, "role": "development_apparent_not_validation" if args.fit else "external_no_refit",
              "overall": {name: performance(labels, value, frozen["thresholds"][name]) for name, value in scores.items()},
              "by_level": {}}
    for level in sorted({r["signal_percent"] for r in rows}):
        # A global association may merely track dilution. Stratification exposes
        # this issue; ten pairs per level limit evaluation precision.
        indices = np.asarray([i for i, r in enumerate(rows) if r["signal_percent"] == level])
        report["by_level"][str(level)] = {name: performance(labels[indices], value[indices], frozen["thresholds"][name])
                                           for name, value in scores.items()}
    if args.frozen:
        report["incremental"] = incremental_interval(rows, labels, scores, design["reliability"])
    output = [dict(row, **{f"risk_{name}": float(value[i]) for name, value in scores.items()}) for i, row in enumerate(rows)]
    atomic_tsv(args.output / "predictions.tsv", output, list(output[0]))
    atomic_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
