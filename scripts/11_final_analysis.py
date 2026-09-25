#!/usr/bin/env python3
"""Final, read-only analysis for the ACDSA manuscript.

Pairs anchor OOPS/ZOOPS motifs, evaluates disagreement with three bounded
column distances, residualizes quality within cohort-by-signal cells, and
bootstraps whole trajectories. It never reruns MEME or TOMTOM.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from motif_scoring import parse_meme_xml, reverse_complement_matrix
from robustness_io import atomic_json, atomic_tsv


ROOT = Path(__file__).resolve().parents[1]
PAIR_FIELDS = ["analysis", "cohort", "sample", "signal", "width", "jsd", "tv",
               "hellinger", "oops_ap", "zoops_ap", "oops_conf", "zoops_conf",
               "zoops_fraction", "oops_resampling", "zoops_resampling"]


def column_distance(a, b, metric):
    if metric == "tv":
        return np.abs(a - b).sum(axis=1) / 2
    if metric == "hellinger":
        return np.sqrt(((np.sqrt(a) - np.sqrt(b)) ** 2).sum(axis=1) / 2)
    m = (a + b) / 2
    terms = np.zeros(len(a))
    for x in (a, b):
        mask = x > 0
        terms += np.where(mask, x * np.log2(np.divide(x, m, out=np.ones_like(x), where=mask)), 0).sum(axis=1) / 2
    return terms


def distance(left, right, metric="jsd", fraction=.8):
    a = np.asarray(left, float); a /= a.sum(axis=1, keepdims=True)
    minimum = math.ceil(fraction * max(len(left), len(right)))
    best = float("inf")
    for raw in (right, reverse_complement_matrix(right)):
        b = np.asarray(raw, float); b /= b.sum(axis=1, keepdims=True)
        for offset in range(-len(b) + minimum, len(a) - minimum + 1):
            start, end = max(0, offset), min(len(a), offset + len(b))
            x, y = a[start:end], b[start-offset:end-offset]
            overlap = end - start
            unmatched = len(a) + len(b) - 2 * overlap
            score = (column_distance(x, y, metric).sum() + unmatched) / (overlap + unmatched)
            best = min(best, float(score))
    return best


def load_pairs(label, evaluation):
    rows = list(csv.DictReader((evaluation / "metrics.tsv").open(), delimiter="\t"))
    cells = {}
    for row in rows:
        if int(row["order_repeat"]) or int(row["meme_repeat"]):
            continue
        key = (int(row["sample"]), float(row["signal_percent"]), int(row["input_repeat"]), row["model"], int(row["motif_width"]))
        result = json.loads((evaluation / row["evaluation_id"] / "result.json").read_text())
        motif = parse_meme_xml(Path(result["motif"]))
        cells[key] = (result, motif)
    pairs = []
    anchors = sorted({(s, level, width) for s, level, repeat, model, width in cells if repeat == 0})
    for sample, level, width in anchors:
        try:
            o, z = cells[sample, level, 0, "oops", width], cells[sample, level, 0, "zoops", width]
            oc, zc = cells[sample, level, 1, "oops", width], cells[sample, level, 1, "zoops", width]
        except KeyError:
            continue
        rec = {"cohort": label, "sample": sample, "signal": level, "width": width,
               "oops_ap": float(o[0]["ap"]), "zoops_ap": float(z[0]["ap"]),
               "oops_conf": math.log1p(max(0, float(o[0]["motif_neg_log10_evalue"]))),
               "zoops_conf": math.log1p(max(0, float(z[0]["motif_neg_log10_evalue"]))),
               "zoops_fraction": float(z[0]["site_fraction"])}
        for metric in ("jsd", "tv", "hellinger"):
            rec[metric] = distance(o[1].matrix, z[1].matrix, metric)
        rec["oops_resampling"] = distance(o[1].matrix, oc[1].matrix)
        rec["zoops_resampling"] = distance(z[1].matrix, zc[1].matrix)
        pairs.append(rec)
    return pairs


def center_within_groups(values, groups):
    """Remove group means while retaining one common slope downstream."""
    out = np.asarray(values, float).copy()
    for group in sorted(set(groups)):
        idx = np.asarray([i for i, g in enumerate(groups) if g == group])
        out[idx] -= out[idx].mean()
    return out


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None


def ols_predict(x, y):
    """Return OLS predictions without estimator-state or serialization."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim == 1:
        x = x[:, None]
    design = np.column_stack([np.ones(len(y)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return design @ coefficients


def r_squared(y, prediction):
    y = np.asarray(y, float)
    denominator = np.square(y - y.mean()).sum()
    if denominator == 0:
        return None
    return float(1 - np.square(y - prediction).sum() / denominator)


def conditional_statistic(rows, model, metric="jsd"):
    """Use one coherent fixed-effect model for partial r and incremental R2.

    Outcome, disagreement, and every control are first centered within the
    cohort-by-signal-by-width cells. Common control slopes are then fitted on
    the centered data. This is the Frisch--Waugh--Lovell fixed-effect result;
    it does not fit a separate slope from only ten observations in each cell.
    """
    groups = [(r["cohort"], r["signal"], r["width"]) for r in rows]
    y = center_within_groups([1-r[f"{model}_ap"] for r in rows], groups)
    d = center_within_groups([r[metric] for r in rows], groups)
    baseline_names = [f"{model}_conf", f"{model}_resampling"] + (["zoops_fraction"] if model == "zoops" else [])
    x = np.column_stack([center_within_groups([r[n] for r in rows], groups) for n in baseline_names])
    prediction_y = ols_predict(x, y)
    prediction_d = ols_predict(x, d)
    residual_y = y - prediction_y
    residual_d = d - prediction_d
    augmented = np.column_stack([x, d])
    prediction_full = ols_predict(augmented, y)
    baseline_r2 = r_squared(y, prediction_y)
    full_r2 = r_squared(y, prediction_full)
    return {"n": len(rows), "within_cell_r": corr(d, y),
            "partial_r": corr(residual_d, residual_y),
            "baseline_r2": baseline_r2,
            "full_r2": full_r2,
            "delta_r2": float(full_r2 - baseline_r2)}


def resample_trajectories(rows, rng):
    """Stratified cluster bootstrap: resample whole trajectories per cohort."""
    sampled = []
    for cohort in sorted({r["cohort"] for r in rows}):
        trajectories = sorted({r["sample"] for r in rows if r["cohort"] == cohort})
        for chosen in rng.choice(trajectories, size=len(trajectories), replace=True):
            sampled.extend(r.copy() for r in rows
                           if r["cohort"] == cohort and r["sample"] == chosen)
    return sampled


def summarize(rows, model, metric="jsd", boot=10000, seed=20260925):
    result = conditional_statistic(rows, model, metric)
    if boot:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(boot):
            value = conditional_statistic(resample_trajectories(rows, rng), model, metric)["partial_r"]
            if value is not None and math.isfinite(value):
                values.append(value)
        if not values:
            raise ValueError("cluster bootstrap produced no finite partial correlations")
        result["partial_ci95"] = [float(np.quantile(values, .025)),
                                  float(np.quantile(values, .975))]
    return result


def leave_one_trajectory_out(rows, model):
    values = []
    for cohort, sample in sorted({(r["cohort"], r["sample"]) for r in rows}):
        kept = [r for r in rows if (r["cohort"], r["sample"]) != (cohort, sample)]
        values.append({"omitted_cohort": cohort, "omitted_sample": sample,
                       "partial_r": conditional_statistic(kept, model)["partial_r"]})
    observed = np.asarray([r["partial_r"] for r in values], float)
    return {"minimum": float(observed.min()), "median": float(np.median(observed)),
            "maximum": float(observed.max()), "values": values}


def load_coarse_pairs(work, labels):
    rows = []
    for label in labels:
        path = work / "results" / label / "reliability" / "pairs.tsv"
        for row in csv.DictReader(path.open(), delimiter="\t"):
            rows.append({"cohort": label, "sample": int(row["sample"]),
                         "signal": float(row["signal_percent"]), "width": 19,
                         "jsd": float(row["disagreement"]),
                         "oops_ap": float(row["oops_ap"]), "zoops_ap": float(row["zoops_ap"]),
                         "oops_conf": math.log1p(max(0, float(row["oops_neg_log10_evalue"]))),
                         "zoops_conf": math.log1p(max(0, float(row["zoops_neg_log10_evalue"]))),
                         "oops_resampling": float(row["oops_resampling"]),
                         "zoops_resampling": float(row["zoops_resampling"]),
                         "zoops_fraction": float(row["zoops_site_fraction"])})
    return rows


def write_pairs(path, groups):
    rows = []
    for analysis, values in groups.items():
        for value in values:
            rows.append({field: analysis if field == "analysis" else value.get(field, "")
                         for field in PAIR_FIELDS})
    atomic_tsv(path, rows, PAIR_FIELDS, overwrite=True)


def read_pairs(path):
    groups = {"ctcf_initial": [], "ctcf_fine": [], "gabpa_width": []}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            analysis = raw.pop("analysis")
            if analysis not in groups:
                raise ValueError(f"unknown analysis in compact pairs: {analysis}")
            row = {"cohort": raw.pop("cohort")}
            for field, value in raw.items():
                if value != "":
                    row[field] = int(value) if field in {"sample", "width"} else float(value)
            groups[analysis].append(row)
    if any(not values for values in groups.values()):
        raise ValueError("compact pairs file is missing one or more analyses")
    return groups


def gabpa_descriptive(evaluation):
    metrics = list(csv.DictReader((evaluation / "metrics.tsv").open(), delimiter="\t"))
    rows = [row for row in metrics if not int(row["order_repeat"]) and not int(row["meme_repeat"])]
    by_level = {}
    reference_at_full_signal = {}
    for model in ("oops", "zoops"):
        by_level[model] = {}
        for level in (100, 75, 50):
            selected = [row for row in rows
                        if row["model"] == model and float(row["signal_percent"]) == level]
            by_level[model][str(level)] = {
                "n": len(selected),
                "mean_ap": float(np.mean([float(row["ap"]) for row in selected])),
                "mean_site_fraction": float(np.mean([float(row["site_fraction"]) for row in selected])),
            }
        selected = [row for row in rows
                    if row["model"] == model and float(row["signal_percent"]) == 100]
        hits = 0
        for row in selected:
            result = json.loads((evaluation / row["evaluation_id"] / "result.json").read_text())
            hits += any(float(match["p-value"]) <= .05 and int(match["Overlap"]) >= 8
                        for match in result["tomtom"])
        reference_at_full_signal[model] = {"matches": hits, "total": len(selected),
                                           "p_threshold": .05, "minimum_overlap": 8}
    return {"by_level": by_level, "reference_at_100_percent": reference_at_full_signal}


def frozen_transfer(train, test, model):
    names=[f"{model}_conf",f"{model}_resampling"]+(["zoops_fraction"] if model=="zoops" else [])
    def arr(rows, full):
        return np.asarray([[r[n] for n in names]+([r["jsd"]] if full else []) for r in rows])
    ytr=np.asarray([1-r[f"{model}_ap"] for r in train]); yte=np.asarray([1-r[f"{model}_ap"] for r in test])
    out={}
    for key,full in (("baseline",False),("augmented",True)):
        xtr,xte=arr(train,full),arr(test,full)
        mean=xtr.mean(0); scale=xtr.std(0); scale[scale==0]=1
        fit=LinearRegression().fit((xtr-mean)/scale,ytr)
        pred=fit.predict((xte-mean)/scale)
        out[key]={"rmse":float(mean_squared_error(yte,pred)**.5),"r":corr(pred,yte)}
    out["delta_rmse"]=out["augmented"]["rmse"]-out["baseline"]["rmse"]
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "work/main")
    parser.add_argument("--output", type=Path, default=ROOT / "results/final_analysis.json")
    parser.add_argument("--pairs-input", type=Path,
                        help="reanalyze the compact repository table instead of raw MEME outputs")
    parser.add_argument("--pairs-output", type=Path,
                        default=ROOT / "results/final_analysis_pairs.tsv")
    parser.add_argument("--gabpa-diagnostics", type=Path,
                        default=ROOT / "results/gabpa_diagnostics.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260925)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap must be non-negative")
    labels = ("calu3-development", "a549-encsr035oxa", "mcf7-encsr000ahd")
    gabpa_evaluation = None
    if args.pairs_input:
        groups = read_pairs(args.pairs_input.resolve())
        coarse, fine, gabpa = (groups["ctcf_initial"], groups["ctcf_fine"], groups["gabpa_width"])
        gabpa_summary = json.loads(args.gabpa_diagnostics.resolve().read_text())
    else:
        work = args.work.resolve()
        fine=[]
        for label in labels:
            fine += load_pairs(label, work/f"results/extensions/fine-{label}/evaluation")
        coarse = load_coarse_pairs(work, labels)
        gabpa_evaluation = work/"results/extensions/gabpa-width/evaluation"
        gabpa=load_pairs("gabpa-a549-encsr000bpy", gabpa_evaluation)
        write_pairs(args.pairs_output.resolve(), {"ctcf_initial": coarse,
                                                  "ctcf_fine": fine,
                                                  "gabpa_width": gabpa})
        gabpa_summary = gabpa_descriptive(gabpa_evaluation)
        atomic_json(args.gabpa_diagnostics.resolve(), gabpa_summary, overwrite=True)
    calu=[r for r in fine if r["cohort"]=="calu3-development"]
    payload={"schema_version": 2,
             "status": "exploratory_post_initial_stress_test",
             "model": "cell-centered common control slopes; trajectory-cluster bootstrap",
             "bootstrap": {"replicates": args.bootstrap, "seed": args.seed},
             "ctcf_fine":{}, "ctcf_coarse":{}, "ctcf_by_cohort":{},
             "gabpa_width":{}, "sensitivity":{}, "leave_one_trajectory_out":{},
             "frozen_transfer":{}, "gabpa_descriptive": gabpa_summary}
    for model in ("oops","zoops"):
        payload["ctcf_fine"][model]=summarize(fine,model,boot=args.bootstrap,seed=args.seed)
        payload["ctcf_coarse"][model]=summarize(coarse,model,boot=args.bootstrap,seed=args.seed+1)
        payload["gabpa_width"][model]=summarize(gabpa,model,boot=args.bootstrap,seed=args.seed+2)
        payload["ctcf_by_cohort"][model] = {
            label: conditional_statistic([r for r in fine if r["cohort"] == label], model)
            for label in labels}
        payload["sensitivity"][model]={m:conditional_statistic(fine,model,m)["partial_r"]
                                       for m in ("jsd","tv","hellinger")}
        payload["leave_one_trajectory_out"][model]=leave_one_trajectory_out(fine,model)
        payload["frozen_transfer"][model]={}
        for label in ("a549-encsr035oxa","mcf7-encsr000ahd","gabpa-a549-encsr000bpy"):
            test=(gabpa if label.startswith("gabpa") else [r for r in fine if r["cohort"]==label])
            payload["frozen_transfer"][model][label]=frozen_transfer(calu,test,model)
    atomic_json(args.output, payload, overwrite=True)
    print(json.dumps({"output": str(args.output.resolve()),
                      "bootstrap_replicates": args.bootstrap,
                      "ctcf_initial_pairs": len(coarse),
                      "ctcf_fine_pairs": len(fine),
                      "gabpa_width_pairs": len(gabpa),
                      "pairs_output": None if args.pairs_input else str(args.pairs_output.resolve())}))


if __name__ == "__main__": main()
