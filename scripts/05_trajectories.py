#!/usr/bin/env python3
"""Make paired nested FASTAs and a discovery plan with separate random streams."""
import argparse
import filecmp
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from fasta_utils import read_fasta
from robustness_io import atomic_json, atomic_tsv, atomic_write, read_tsv
from experiments import cohort_settings, legacy_runs


def seed_for(base, stream, sample, repeat=0):
    text = f"{base}:{stream}:{sample}:{repeat}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % 2147483647


def nested_records(selected, pairs, positives, negatives, order_seed, level):
    size = len(selected)
    positive_count = round(size * level / 100)
    if abs(positive_count - size * level / 100) > 1e-8:
        raise ValueError("signal level cannot be represented exactly at this sample size")
    order = list(range(size))
    random.Random(order_seed).shuffle(order)
    replaced = set(order[:size - positive_count])
    records, manifest = [], []
    ranks = {slot: rank for rank, slot in enumerate(order)}
    for slot, pos_id in enumerate(selected):
        neg_id = pairs[pos_id]
        source = "negative" if slot in replaced else "positive"
        source_id = neg_id if source == "negative" else pos_id
        sequence = negatives[neg_id] if source == "negative" else positives[pos_id]
        records.append((f">slot_{slot:04d}|{source}|{source_id}", sequence))
        manifest.append({"slot": slot, "replacement_rank": ranks[slot], "positive_id": pos_id,
                         "negative_id": neg_id, "source": source, "source_id": source_id})
    return records, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=Path, required=True)
    p.add_argument("--pairs", type=Path, help="Required for a new or historical cohort; refinement uses frozen manifests")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cohort", help="Required identifier for a new external cohort")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/design.json")
    p.add_argument("--refine-from", type=Path, help="Stage-14 refinement.json for the frozen original trajectories")
    p.add_argument("--legacy-root", type=Path)
    p.add_argument("--original-plan", type=Path, help="Stage-10 historical plan from a fresh reproduction; alternative to --legacy-root")
    p.add_argument("--refine-samples", nargs="+", type=int, help="Defaults to refinement.samples in the design config")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--allow-partial-levels", action="store_true",
                   help="Extension mode: permit levels without 0% and 100% endpoints")
    p.add_argument("--historical", action="store_true", help="Reproduce the original Calu-3 shared-RNG design from robustness.json")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if not args.refine_from and args.pairs is None:
        p.error("--pairs is required when generating a cohort")
    if args.historical:
        if args.refine_from:
            p.error("historical regeneration and adaptive refinement are distinct modes")
        historical(args)
        return
    if args.refine_from:
        if bool(args.legacy_root) == bool(args.original_plan) or not 1 <= args.round <= config["refinement"]["maximum_rounds"]:
            p.error("refinement requires exactly one of --legacy-root/--original-plan and a round within budget")
        args.refine_samples = args.refine_samples or config["refinement"]["samples"]
        refine_original(args)
        return
    if not args.cohort:
        p.error("--cohort is required for a new external cohort")
    generate_cohort(args, config)


def generate_cohort(args, config):
    """Plan independent random streams while retaining paired nested inputs."""
    d = config["trajectories"]
    # Width is defined by the target reference, never by the best result.
    discovery, target = cohort_settings(config, args.cohort)
    levels = d["levels_percent"]
    if len(set(levels)) != len(levels) or any(not 0 <= x <= 100 for x in levels):
        raise ValueError("signal levels must be unique and within [0, 100]")
    if not args.allow_partial_levels and not {0, 100} <= set(levels):
        raise ValueError("include both endpoints")
    if d.get("input_repeats", 1) not in (1, 2):
        raise ValueError("use one historical input or one anchor plus one companion")
    positives = {h[1:].split()[0]: s for h, s in read_fasta(args.datasets / "train_positives.fa")}
    negatives = {h[1:].split()[0]: s for h, s in read_fasta(args.datasets / "train_negatives.fa")}
    pair_rows = [r for r in read_tsv(args.pairs) if r["split"] == "train" and int(r["negative_index"]) == 1]
    pairs = {r["positive_id"]: r["negative_id"] for r in pair_rows}
    if len(pairs) != len(pair_rows) or not set(pairs) <= positives.keys() or not set(pairs.values()) <= negatives.keys():
        raise ValueError("inconsistent or duplicate training pairs")
    if not 0 < d["sample_size"] <= len(pairs) or d["samples"] < 1:
        raise ValueError("invalid trajectory sample size/count")
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for sample, input_repeat in ((s, r) for s in range(1, d["samples"] + 1)
                                 for r in range(d.get("input_repeats", 1))):
        sample_seed = seed_for(d["sample_seed"], "sample", sample, input_repeat)
        selected = random.Random(sample_seed).sample(sorted(pairs), d["sample_size"])
        order_seed = seed_for(d["order_seed"], "replacement", sample, input_repeat)
        meme_seed = seed_for(d["meme_seed"], "meme", sample)
        trajectory = f"sample_{sample:02d}_input_{input_repeat:02d}_order_00"
        for level in levels:
            fasta = args.output / trajectory / f"signal_{level:g}.fa"
            records, manifest = nested_records(selected, pairs, positives, negatives, order_seed, level)
            atomic_write(fasta, lambda f, records=records: f.writelines(f"{h}\n{s}\n" for h, s in records))
            atomic_tsv(Path(str(fasta) + ".tsv"), manifest, list(manifest[0]))
            for model in ("oops", "zoops"):
                tasks.append({"cohort": args.cohort, "factor": target["factor"], "sample": sample, "trajectory": trajectory,
                    "input_repeat": input_repeat, "order_repeat": 0, "meme_repeat": 0, "signal_percent": level,
                    "sample_seed": sample_seed, "order_seed": order_seed, "meme_seed": meme_seed,
                    "model": model, "input": str(fasta.resolve()), "sample_size": d["sample_size"], **discovery})
    atomic_json(args.output / "plan.json", {"schema_version": 2, "config": config,
        "inputs": [str(p.resolve()) for p in (args.pairs, args.datasets / "train_positives.fa", args.datasets / "train_negatives.fa")],
        "tasks": tasks})
    print(json.dumps({"discovery_tasks": len(tasks), "plan": str(args.output / "plan.json"), "executed": 0}))


def refine_original(args):
    """Reuse original pair identities AND replacement ranks, not a newly sampled path."""
    proposals = json.loads(args.refine_from.read_text())
    if any(r["cohort"] != "calu3-original" for r in proposals):
        raise ValueError("historical refinement cannot reuse proposals from a different cohort")
    if args.legacy_root:
        for role in ("positives", "negatives"):
            name = f"train_{role}.fa"
            if not filecmp.cmp(args.datasets / name, args.legacy_root / "results/robustness/datasets" / name, shallow=False):
                raise ValueError("refinement training sequences differ from the original cohort")
    requested = sorted({(int(r["sample"]), float(r["next_signal_percent"])) for r in proposals
                        if int(r["sample"]) in args.refine_samples and int(r["order_repeat"]) == int(r["meme_repeat"]) == 0})
    if args.original_plan:
        anchors = json.loads(args.original_plan.read_text())["tasks"]
        if any(r["cohort"] != "calu3-original" for r in anchors):
            raise ValueError("refinement requires the historical Calu-3 plan")
        original = {(r["sample"], r["model"]): r for r in anchors if r["signal_percent"] == 100}
    else:
        original = {(r["sample"], r["model"]): r for r in legacy_runs(args.legacy_root) if r["background"] == "markov2"}
    sequences = {role: {h[1:].split()[0]: s for h, s in read_fasta(args.datasets / f"train_{role}.fa")}
                 for role in ("positives", "negatives")}
    if not requested:
        raise ValueError("no refinement proposals for the selected samples")
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for sample, level in requested:
        manifest = (Path(original[sample, "oops"]["input"] + ".tsv") if args.original_plan else
                    args.legacy_root / f"manifests/trajectories/trajectory_{sample:02d}.tsv")
        rows = read_tsv(manifest)
        slots = sorted((r for r in rows if args.original_plan or r["signal_percent"] == rows[0]["signal_percent"]), key=lambda r: int(r["slot"]))
        if [int(r["slot"]) for r in slots] != list(range(len(slots))) or sorted(int(r["replacement_rank"]) for r in slots) != list(range(len(slots))):
            raise ValueError("historical slots and replacement ranks must each be a permutation")
        if args.original_plan:
            for endpoint_level, role in ((100, "positives"), (0, "negatives")):
                endpoint = next(r for r in anchors if r["sample"] == sample and r["model"] == "oops" and r["signal_percent"] == endpoint_level)
                expected = [sequences[role][r["positive_id" if role == "positives" else "negative_id"]] for r in slots]
                if [s for _, s in read_fasta(Path(endpoint["input"]))] != expected:
                    raise ValueError("refinement sequences differ from the original endpoints")
        keep = round(len(slots) * level / 100)
        if abs(keep - len(slots) * level / 100) > 1e-8:
            raise ValueError("midpoint is not representable as an integer sequence count")
        records = []
        for row in slots:
            negative = int(row["replacement_rank"]) < len(slots) - keep
            role = "negatives" if negative else "positives"
            identifier = row["negative_id"] if negative else row["positive_id"]
            records.append((f">slot_{int(row['slot']):04d}|{identifier}", sequences[role][identifier]))
        fasta = args.output / f"trajectory_{sample:02d}_signal_{level:g}.fa"
        atomic_write(fasta, lambda f, records=records: f.writelines(f"{h}\n{s}\n" for h, s in records))
        for model in ("oops", "zoops"):
            base = original[sample, model]
            tasks.append({"cohort": "calu3-original", "role": "post_specified_refinement", "round": args.round,
                "sample": sample, "trajectory": str(sample), "order_repeat": 0, "meme_repeat": 0,
                "signal_percent": level, "sample_seed": int(base["sample_seed"] if args.original_plan else base["sampling_seed"]), "order_seed": "legacy shared stream; frozen ranks",
                "meme_seed": int(base["meme_seed"]), "model": model, "sample_size": len(slots),
                "input": str(fasta.resolve()), "source_manifest": str(manifest.resolve()),
                "width": int(base["width"] if args.original_plan else base["motif_width"]), "markov_order": int(base["markov_order"]), "searchsize": int(base["searchsize"]), "threads": 2})
    atomic_json(args.output / "plan.json", {"schema_version": 2, "role": "exploratory refinement; excluded from primary AURC grid", "tasks": tasks})
    print(json.dumps({"discovery_tasks": len(tasks), "executed": 0, "round": args.round}))


def historical(args):
    """Byte-for-byte FASTA reproduction without keeping a second legacy runner."""
    config_path = Path(__file__).resolve().parents[1] / "config/robustness.json"
    config = json.loads(config_path.read_text())
    design, meme = config["trajectories"], config["meme"]
    positives = {h[1:].split()[0]: s.upper() for h, s in read_fasta(args.datasets / "train_positives.fa")}
    negatives = {h[1:].split()[0]: s.upper() for h, s in read_fasta(args.datasets / "train_negatives.fa")}
    pairs = {r["positive_id"]: r["negative_id"] for r in read_tsv(args.pairs) if r["split"] == "train" and int(r["negative_index"]) == 1}
    args.output.mkdir(parents=True, exist_ok=False)
    tasks, uniform = [], []
    for sample in range(1, design["repetitions"] + 1):
        seed = int.from_bytes(hashlib.sha256(f"{design['seed']}\0trajectory={sample}".encode()).digest()[:8], "big")
        rng = random.Random(seed)
        selected = rng.sample(sorted(pairs), design["sample_size"])
        order = list(range(design["sample_size"]))
        rng.shuffle(order)
        ranks = {slot: rank for rank, slot in enumerate(order)}
        for level in design["levels_percent"]:
            replace = design["sample_size"] - design["sample_size"] * level // 100
            records = []
            for slot, positive_id in enumerate(selected):
                source = "negative" if ranks[slot] < replace else "positive"
                sequence = negatives[pairs[positive_id]] if source == "negative" else positives[positive_id]
                records.append((f">trajectory_{sample:02d}|level_{level:03d}|slot_{slot:04d}|{source}", sequence))
            fasta = args.output / f"trajectory_{sample:02d}" / f"signal_{level:03d}.fa"
            atomic_write(fasta, lambda f, records=records: f.writelines(f"{h}\n{s}\n" for h, s in records))
            if level == 100:
                slots = [{"slot": i, "replacement_rank": ranks[i], "positive_id": identifier,
                          "negative_id": pairs[identifier]} for i, identifier in enumerate(selected)]
                atomic_tsv(Path(str(fasta) + ".tsv"), slots, list(slots[0]))
            for model in ("oops", "zoops"):
                task = {"cohort": "calu3-original", "sample": sample, "trajectory": str(sample), "order_repeat": 0, "meme_repeat": 0,
                    "signal_percent": level, "sample_seed": seed, "order_seed": "historical shared RNG", "meme_seed": meme["seed"] + sample,
                    "model": model, "sample_size": design["sample_size"], "input": str(fasta.resolve()),
                    "width": meme["width"], "markov_order": meme["markov_order"], "searchsize": meme["searchsize"], "threads": meme["threads_per_run"]}
                tasks.append(task)
                if level == 100:
                    uniform.append(dict(task, markov_order=0, background="uniform"))
    for name, cells in (("plan.json", tasks), ("uniform_plan.json", uniform)):
        atomic_json(args.output / name, {"schema_version": 2, "role": "historical reproduction", "config": config, "tasks": cells})
    print(json.dumps({"discovery_tasks": len(tasks), "uniform_tasks": len(uniform), "executed": 0}))


if __name__ == "__main__":
    main()
