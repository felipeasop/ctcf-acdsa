#!/usr/bin/env python3
"""Discover motifs only. Plans are inspected by default; --execute starts MEME."""
# Convert each plan cell into an isolated MEME run. Without --execute this only
# inspects the plan; completed runs are reused after identity validation.
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from motif_scoring import parse_markov, parse_meme_xml
from robustness_io import atomic_json


def task_identity(task):
    """Readable ID within one cohort/protocol execution directory."""
    width = f"_width_{task['width']}" if task.get("role") == "width_sensitivity" else ""
    return (f"sample_{task['sample']:02d}_input_{task.get('input_repeat', 0):02d}_order_{task['order_repeat']:02d}"
            f"_repeat_{task['meme_repeat']:02d}_signal_{task['signal_percent']:g}_{task['model']}{width}")


def discover(task, args, version):
    # IDs include block, input, repeat, and level so paired model runs cannot
    # collide in the output directory.
    run_id = task_identity(task)
    directory = args.output / run_id
    temporary = Path(tempfile.mkdtemp(prefix=f"{run_id}.", dir=args.output / ".partial"))
    command = [args.meme, task["input"], "-dna", "-revcomp", "-mod", task["model"],
        # -revcomp searches both strands; -w fixes width; -nmotifs limits to one
        # candidate. -searchsize 0 disables internal subsampling by that option.
        "-nmotifs", "1", "-w", str(task["width"]), "-searchsize", str(task["searchsize"]),
        "-seed", str(task["meme_seed"]), "-p", str(task["threads"]), "-nostatus",
        "-bfile", str(args.background.resolve()), "-markov_order", str(task["markov_order"]), "-oc", str(temporary / "meme")]
    recorded_command = command[:-1] + [str(directory / "meme")]
    atomic_json(temporary / "config.json", {"task": task, "command": recorded_command, "meme_version": version,
        "discovery_background": str(args.background.resolve())})
    start = time.monotonic()
    try:
        # The timeout applies to this process, not to the complete plan.
        with (temporary / "stdout.log").open("x") as out, (temporary / "stderr.log").open("x") as err:
            subprocess.run(command, stdout=out, stderr=err, check=True, timeout=args.timeout)
        motif = parse_meme_xml(temporary / "meme/meme.xml")
        if (motif.width, motif.meme_seed, motif.primary_count) != (task["width"], task["meme_seed"], task["sample_size"]):
            raise ValueError("MEME result does not match planned width, seed or sample size")
        result = {"run_id": run_id, "status": "complete", "task": task, "meme_version": version,
            "discovery_background": str(args.background.resolve()),
            "runtime_seconds": time.monotonic() - start,
            "outputs": ["meme/meme.xml", "meme/meme.txt"]}
        atomic_json(temporary / "result.json", result)
        os.replace(temporary, directory)
        return {"run_id": run_id, "status": "complete"}
    except Exception as exc:
        # A missing complete result is a technical failure, never a biological label.
        failure = args.output / "failures" / f"{run_id}.{time.time_ns()}.json"
        atomic_json(failure, {"run_id": run_id, "task": task, "error": str(exc),
                              "runtime_seconds": time.monotonic() - start})
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def completed(task, output):
    """Accept reuse only when identity, XML and MEME metadata are consistent."""
    run_id = task_identity(task)
    directory = output / run_id
    result_path = directory / "result.json"
    if not result_path.exists():
        return False
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete" or result.get("task") != task or result.get("run_id") != run_id:
        raise ValueError(f"existing discovery conflicts with current plan: {directory}")
    motif = parse_meme_xml(directory / "meme/meme.xml")
    if (motif.width, motif.meme_seed, motif.primary_count) != (
            task["width"], task["meme_seed"], task["sample_size"]):
        raise ValueError(f"existing motif conflicts with current plan: {directory}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--background", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--meme", default="meme")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--timeout", type=int, default=3600, help="seconds per MEME process")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--samples", type=int, nargs="+", help="Restrict the supplied plan without resampling inputs")
    p.add_argument("--levels", type=float, nargs="+")
    p.add_argument("--meme-seeds", type=int, nargs="+", help="Replace only MEME seeds; repeat IDs start at one")
    args = p.parse_args()
    tasks = json.loads(args.plan.read_text())["tasks"]
    if args.samples:
        tasks = [t for t in tasks if t["sample"] in args.samples]
    if args.levels:
        tasks = [t for t in tasks if t["signal_percent"] in args.levels]
    if args.meme_seeds:
        # Optional MEME-seed audit; this is not the input-resampling control.
        if len(set(args.meme_seeds)) != len(args.meme_seeds):
            p.error("duplicate MEME seeds")
        tasks = [dict(t, meme_seed=seed, meme_repeat=i) for t in tasks for i, seed in enumerate(args.meme_seeds, 1)]
    if not tasks or args.jobs < 1 or args.timeout < 1:
        p.error("nonempty plan, positive jobs and timeout required")
    if len({t["cohort"] for t in tasks}) != 1:
        p.error("use one cohort per execution directory")
    identities = [task_identity(t) for t in tasks]
    if len(set(identities)) != len(identities):
        p.error("duplicate discovery cells; restrict seed overrides to baseline tasks")
    for task in tasks:
        if task["model"] not in ("oops", "zoops") or not 0 <= task["signal_percent"] <= 100:
            p.error("invalid occurrence model or signal percentage")
        if min(task["sample_size"], task["width"], task["threads"]) < 1 or not 0 <= task["meme_seed"] < 2147483647:
            p.error("invalid sample size, motif width, threads or MEME seed")
        if not Path(task["input"]).is_file() or not Path(str(task["input"]) + ".tsv").is_file():
            p.error(f"planned FASTA or its manifest is missing: {task['input']}")
    for order in {t["markov_order"] for t in tasks}:
        parse_markov(args.background, order)
    print(json.dumps({"tasks": len(tasks), "jobs": args.jobs, "execute": args.execute}))
    if not args.execute:
        return
    version = subprocess.run([args.meme, "-version"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".partial").mkdir(exist_ok=True)
    saved_plan = {"tasks": tasks, "meme_version": version,
        "source_plan": str(args.plan.resolve()), "discovery_background": str(args.background.resolve())}
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != saved_plan:
            raise ValueError("existing discovery directory belongs to another plan/environment")
    else:
        atomic_json(plan_path, saved_plan)
    reused = [task for task in tasks if completed(task, args.output)]
    pending = [task for task in tasks if task not in reused]
    print(json.dumps({"total": len(tasks), "reused": len(reused), "pending": len(pending)}))
    executor = ThreadPoolExecutor(max_workers=args.jobs)
    # Python threads coordinate external subprocesses. jobs * threads per MEME
    # approximates process-level parallelism; memory also limits this choice.
    try:
        results = list(executor.map(lambda task: discover(task, args, version), pending))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    # Each completed record retains its own cohort and trajectory identity.
    print(json.dumps({"reused": len(reused), "completed_now": len(results)}, indent=2))


if __name__ == "__main__":
    main()
