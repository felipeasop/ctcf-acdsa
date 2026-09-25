"""Read executed cohorts and TOMTOM records without altering source artifacts."""
# Compatibility adapter: convert current and historical outputs to common
# records. It preserves the path and identity of each run.
import csv
import json
import math
from pathlib import Path


def cohort_settings(design, cohort):
    """Resolve the declared target; never infer it from the result."""
    factor = design["cohort_factors"][cohort]
    target = design["factors"][factor]
    discovery = dict(design["discovery"], width=target["width"])
    reliability = dict(design["reliability"])
    reliability.update(target)
    reliability["factor"] = factor
    return discovery, reliability


def reference_catalog(design):
    """Freeze all reference contents before cross-factor validation."""
    root = Path(__file__).resolve().parents[2] / "references"
    return {factor: (root / settings["reference_file"]).read_text()
            for factor, settings in design["factors"].items()}


def evaluation_protocol(inputs, order, pseudocount, minimum_overlap=None):
    """Explicit scoring contract; paths identify inputs, not immutable contents."""
    # Paths identify sources for this run but do not detect in-place edits;
    # inputs and results must therefore remain immutable.
    protocol = {"inputs": inputs, "order": order, "pseudocount": float(pseudocount),
        "score": "maximum_both_strands_log_odds_v1", "metrics": "AP_grouped_ties_AUROC_half_ties_v1"}
    if minimum_overlap is not None:
        protocol["tomtom_minimum_overlap"] = int(minimum_overlap)
    return json.dumps(protocol, sort_keys=True)


def tomtom_rows(path):
    # The required header distinguishes no match from a broken output. Original
    # values remain strings to preserve scientific notation.
    with path.open() as f:
        reader = csv.DictReader((line for line in f if line.strip() and not line.startswith("#")), delimiter="\t")
        required = {"Query_ID", "Target_ID", "p-value", "E-value", "q-value", "Overlap", "Optimal_offset", "Orientation"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"missing TOMTOM columns: {path}")
        rows = list(reader)
    for row in rows:
        for key in ("p-value", "E-value", "q-value"):
            value = float(row[key])
            if not math.isfinite(value) or value < 0 or (key != "E-value" and value > 1):
                raise ValueError(f"invalid TOMTOM {key}: {path}")
    return rows


def legacy_runs(source):
    # Adapter for the original Calu-3 layout only; it does not turn historical
    # reruns into prospective data or invent companion samples.
    root = Path(source) / "results/robustness"
    with (root / "experiment_log.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len({r["run_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate legacy run IDs")
    inputs = {role: str((root / "datasets" / f"test_{role}.fa").resolve())
              for role in ("positives", "negatives")}
    inputs.update(background=str((root / "background/train_negatives_markov2.txt").resolve()),
                  reference=str((Path(__file__).resolve().parents[2] / "references/CTCF_full.meme").resolve()))
    for row in rows:
        if row["status"] != "complete":
            continue
        directory = root / "experiments/runs" / row["run_id"]
        parameters = json.loads(row["hyperparameters_json"])
        yield dict(row, directory=directory, cohort="calu3-original", sample=int(row["trajectory"]),
            order_repeat=0, meme_repeat=0, ap=float(row["auprc"]), auroc=float(row["auroc"]),
            signal_percent=float(row["signal_percent"]), model=row["model"],
            protocol=evaluation_protocol(inputs, 2, parameters["scoring_pwm_pseudocount_per_base"]),
            input=str((root / "trajectories" / f"trajectory_{int(row['trajectory']):02d}" / f"signal_{int(row['signal_percent']):03d}.fa").resolve()),
            discovery_background=str((root / "background" / ("uniform_order0.txt" if row["background"] == "uniform" else "train_negatives_markov2.txt")).resolve()),
            tomtom=tomtom_rows(directory / "tomtom_ctcf_full/tomtom.tsv"))


def discovery_runs(source):
    # Require every planned task to finish; a directory with only favorable
    # results cannot pass as a complete experiment.
    source = Path(source)
    configs = sorted(source.glob("*/config.json"))
    plan = source / "plan.json"
    expected = json.loads(plan.read_text())["tasks"] if plan.exists() else None
    if expected is not None and len(configs) != len(expected):
        raise ValueError("discovery plan is incomplete; inspect failures before evaluation")
    completed = []
    for config in configs:
        path = config.parent / "result.json"
        if not path.exists():
            raise ValueError(f"discovery has no completed result: {config.parent}")
        result = json.loads(path.read_text())
        if result["status"] != "complete" or result["task"] != json.loads(config.read_text())["task"]:
            raise ValueError(f"inconsistent discovery result: {path}")
        completed.append(result)
    if expected is not None:
        serialize = lambda task: json.dumps(task, sort_keys=True)
        if sorted(map(serialize, expected)) != sorted(serialize(r["task"]) for r in completed):
            raise ValueError("completed discoveries differ from the saved execution plan")
    for result, config in zip(completed, configs):
        yield dict(result["task"], run_id=result["run_id"], directory=config.parent,
                   discovery_background=result["discovery_background"],
                   background=result["task"].get("background", f"markov{result['task']['markov_order']}"))
