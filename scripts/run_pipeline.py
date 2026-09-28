"""
End-to-end MIM pipeline. Safe to re-run: finished steps are skipped and
interrupted training resumes from mim_latest.pt.

  1. Baseline probe (val): random ViT + old 1-epoch MIM checkpoint
  2. Mask-ratio ablation: 50 / 60 / 75%, 5 epochs each
  3. Full probe of the ablation checkpoints (val)
  4. Select the ratio by val Macro-F1 (accuracy breaks ties)
  5. 100-epoch MIM run at the selected ratio
  6. Final probes: test on the primary split, plus val/test on the plain
     split as a sensitivity analysis
  7. Write checkpoints/pipeline_summary.md
"""

import csv
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CKPT = ROOT / "checkpoints"
LOG_DIR = CKPT / "logs"

PRIMARY_SPLIT = "dataset/manifests/probe_geo_split.csv"
PLAIN_SPLIT = "dataset/manifests/probe_geo_split_plain.csv"
RESULTS = CKPT / "eval" / "probe_results.csv"
SELECTION = CKPT / "mim" / "selected_mask_ratio.json"
SUMMARY = CKPT / "pipeline_summary.md"

OLD_CHECKPOINT = "checkpoints/mim/mim_epoch_01.pt"

ABLATION_RATIOS = [0.50, 0.60, 0.75]
ABLATION_EPOCHS = 5
ABLATION_WARMUP = 1

MAIN_EPOCHS = 100
MAIN_WARMUP = 5

# Macro-F1 differences below this are treated as a tie -> accuracy decides
TIE_TOLERANCE = 0.005

MAX_ATTEMPTS = 4

# Resource safeguards
DISK_FLOOR_GB = 15          # stop the pipeline below this much free disk
RAM_FLOOR_GB = 3            # available system RAM considered too low
LOW_RAM_POLLS = 3           # consecutive low readings before acting
POLL_SECONDS = 15
RESOURCE_LOG_SECONDS = 300

# Data-loader workers per step type; halved after a low-RAM kill
WORKERS = {"train": 8, "probe": 4}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def log(message):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with open(LOG_DIR / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


class ResourceStop(Exception):
    """Disk is nearly full: stop the pipeline (it can be resumed later)."""


def disk_free_gb():
    return shutil.disk_usage(ROOT).free / 2**30


def ram_available_gb():
    return psutil.virtual_memory().available / 2**30


def tree_rss_gb(proc):
    try:
        procs = [proc] + proc.children(recursive=True)
        return sum(p.memory_info().rss for p in procs if p.is_running()) / 2**30
    except psutil.Error:
        return 0.0


def kill_tree(proc):
    procs = [proc] + proc.children(recursive=True)
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=30)


def log_resources(step, proc):
    path = LOG_DIR / "resources.csv"
    new_file = not path.exists()
    vm = psutil.virtual_memory()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "time", "step", "ram_available_gb", "ram_used_pct",
                "pagefile_used_gb", "step_rss_gb", "disk_free_gb",
            ])
        writer.writerow([
            f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            step,
            f"{vm.available / 2**30:.1f}",
            f"{vm.percent:.0f}",
            f"{psutil.swap_memory().used / 2**30:.1f}",
            f"{tree_rss_gb(proc):.1f}",
            f"{disk_free_gb():.0f}",
        ])


def check_disk(context):
    free = disk_free_gb()
    if free < DISK_FLOOR_GB:
        raise ResourceStop(
            f"Only {free:.1f} GB disk free (< {DISK_FLOOR_GB} GB) {context}"
        )


def run(name, args, attempts=MAX_ATTEMPTS):
    """
    Run a script step, logging to its own file; retry on failure.

    A watchdog polls RAM and disk. If available RAM stays below the floor,
    the step is killed and retried with half the data-loader workers
    (training resumes from its latest checkpoint). If disk runs low, the
    step is killed and the pipeline stops.
    """

    kind = "train" if "train_mim" in args[0] else "probe"

    for attempt in range(1, attempts + 1):
        check_disk(f"before {name}")

        env = {**os.environ, "NUM_WORKERS": str(WORKERS[kind])}

        log(
            f"START {name} (attempt {attempt}, {WORKERS[kind]} workers): "
            f"{' '.join(args)}"
        )
        start = time.time()

        with open(LOG_DIR / f"{name}.log", "a", encoding="utf-8") as out:
            child = subprocess.Popen(
                [sys.executable, "-u", *args],
                cwd=ROOT,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
            proc = psutil.Process(child.pid)

            low_ram_polls = 0
            last_resource_log = 0.0
            stop_reason = None

            while child.poll() is None:
                time.sleep(POLL_SECONDS)

                if time.time() - last_resource_log > RESOURCE_LOG_SECONDS:
                    log_resources(name, proc)
                    last_resource_log = time.time()

                if disk_free_gb() < DISK_FLOOR_GB:
                    stop_reason = "disk"
                    break

                if ram_available_gb() < RAM_FLOOR_GB:
                    low_ram_polls += 1
                    if low_ram_polls >= LOW_RAM_POLLS:
                        stop_reason = "ram"
                        break
                else:
                    low_ram_polls = 0

            if stop_reason:
                log_resources(name, proc)
                kill_tree(proc)
                child.wait()

            code = child.returncode

        minutes = (time.time() - start) / 60

        if stop_reason == "disk":
            log(f"STOP  {name}: disk below {DISK_FLOOR_GB} GB, step killed")
            check_disk(f"during {name}")

        if stop_reason == "ram":
            WORKERS[kind] = max(1, WORKERS[kind] // 2)
            log(
                f"KILL  {name}: available RAM below {RAM_FLOOR_GB} GB for "
                f"{LOW_RAM_POLLS * POLL_SECONDS}s; retrying with "
                f"{WORKERS[kind]} workers"
            )
        elif code == 0:
            log(f"DONE  {name} in {minutes:.1f} min")
            return
        else:
            log(f"FAIL  {name} exit {code} after {minutes:.1f} min")

    raise SystemExit(f"Step {name} failed {attempts} times; see logs/{name}.log")


def read_results():
    if not RESULTS.exists():
        return []
    with open(RESULTS, newline="") as f:
        return list(csv.DictReader(f))


def has_result(encoder, split_file, eval_split):
    return any(
        r["encoder"] == encoder
        and r["split_file"] == Path(split_file).name
        and r["eval_split"] == eval_split
        for r in read_results()
    )


def probe(name, encoders, split_file, eval_split):
    missing = [e for e in encoders if not has_result(e, split_file, eval_split)]

    if not missing:
        log(f"SKIP  {name}: results already recorded")
        return

    run(
        name,
        [
            "scripts/linear_probe.py",
            *missing,
            "--split-file", split_file,
            "--eval-split", eval_split,
        ],
    )


def completed_epochs(run_name):
    latest = CKPT / "mim" / run_name / "mim_latest.pt"
    if not latest.exists():
        return 0
    return torch.load(latest, map_location="cpu", weights_only=False)["epoch"]


def train(run_name, ratio, epochs, warmup):
    done = completed_epochs(run_name)

    if done >= epochs:
        log(f"SKIP  train {run_name}: {done}/{epochs} epochs already done")
        return

    args = [
        "scripts/train_mim.py",
        "--mask-ratio", str(ratio),
        "--epochs", str(epochs),
        "--warmup-epochs", str(warmup),
        "--run-name", run_name,
    ]

    # Retries resume from the latest checkpoint
    for attempt in range(1, MAX_ATTEMPTS + 1):
        resume = completed_epochs(run_name) > 0
        try:
            run(
                f"train_{run_name}",
                args + (["--resume"] if resume else []),
                attempts=1,
            )
            return
        except SystemExit:
            if attempt == MAX_ATTEMPTS:
                raise
            log(f"Retrying {run_name} from its latest checkpoint")


def ablation_name(ratio):
    return f"ablation_mr{round(ratio * 100)}"


def ablation_checkpoint(ratio):
    return f"checkpoints/mim/{ablation_name(ratio)}/mim_epoch_{ABLATION_EPOCHS:03d}.pt"


def main_checkpoint():
    return f"checkpoints/mim/main/mim_epoch_{MAIN_EPOCHS:03d}.pt"


# ---------------------------------------------------------
# Selection
# ---------------------------------------------------------

def select_ratio():
    rows = [
        r for r in read_results()
        if r["split_file"] == Path(PRIMARY_SPLIT).name
        and r["eval_split"] == "val"
        and r["region"] == "all"
    ]

    table = []

    for ratio in ABLATION_RATIOS:
        candidates = [r for r in rows if r["encoder"] == ablation_checkpoint(ratio)]
        best = max(
            candidates,
            key=lambda r: (float(r["macro_f1"]), float(r["accuracy"])),
        )
        table.append({
            "ratio": ratio,
            "pooling": best["pooling"],
            "macro_f1": float(best["macro_f1"]),
            "accuracy": float(best["accuracy"]),
        })

    top_f1 = max(t["macro_f1"] for t in table)
    tied = [t for t in table if top_f1 - t["macro_f1"] < TIE_TOLERANCE]
    chosen = max(tied, key=lambda t: t["accuracy"])

    selection = {
        "selected_ratio": chosen["ratio"],
        "rule": (
            "highest val Macro-F1 (best pooling per ratio); ratios within "
            f"{TIE_TOLERANCE} Macro-F1 of the best are tied and accuracy decides"
        ),
        "tied_with_best": [t["ratio"] for t in tied],
        "table": table,
    }

    SELECTION.write_text(json.dumps(selection, indent=2))

    log(f"Selected mask ratio {chosen['ratio']} ({json.dumps(table)})")

    return chosen["ratio"]


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def write_summary(ratio):
    rows = read_results()

    def table(split_file, eval_split, encoders, regions=("all",)):
        lines = [
            "| Encoder | Pooling | Region | Macro-F1 | Accuracy | Images |",
            "|---|---|---|---|---|---|",
        ]
        for encoder in encoders:
            for r in rows:
                if (
                    r["encoder"] == encoder
                    and r["split_file"] == Path(split_file).name
                    and r["eval_split"] == eval_split
                    and r["region"] in regions
                ):
                    lines.append(
                        f"| {encoder} | {r['pooling']} | {r['region']} | "
                        f"{r['macro_f1']} | {r['accuracy']} | {r['eval_images']} |"
                    )
        return "\n".join(lines)

    finals = ["random", OLD_CHECKPOINT, ablation_checkpoint(ratio), main_checkpoint()]
    ablations = [ablation_checkpoint(r) for r in ABLATION_RATIOS]
    all_regions = ("all", "Americas", "Europe/Africa", "Asia/Oceania")

    selection = json.loads(SELECTION.read_text())

    SUMMARY.write_text(
        f"# MIM pipeline summary\n\n"
        f"Generated {datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"## Baselines (val, primary split)\n\n"
        f"{table(PRIMARY_SPLIT, 'val', ['random', OLD_CHECKPOINT])}\n\n"
        f"## Mask-ratio ablation (val, primary split)\n\n"
        f"{table(PRIMARY_SPLIT, 'val', ablations)}\n\n"
        f"Selected ratio: **{ratio}**. Rule: {selection['rule']}. "
        f"Ratios tied with the best: {selection['tied_with_best']}.\n\n"
        f"## Final: 100-epoch encoder (test, primary split)\n\n"
        f"{table(PRIMARY_SPLIT, 'test', finals, all_regions)}\n\n"
        f"## Sensitivity: plain geographic split\n\n"
        f"### val\n\n{table(PLAIN_SPLIT, 'val', finals)}\n\n"
        f"### test\n\n{table(PLAIN_SPLIT, 'test', finals)}\n"
    )

    log(f"Summary written: {SUMMARY}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def keep_awake():
    # Ask Windows not to sleep while this process runs (released on exit)
    if sys.platform == "win32":
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    keep_awake()

    log("=" * 60)
    log("Pipeline started")

    try:
        run_steps()
    except ResourceStop as error:
        log(f"PIPELINE STOPPED: {error}. Free space, then re-run to resume.")
        raise SystemExit(1)
    except SystemExit as error:
        log(f"PIPELINE FAILED: {error}")
        raise


def run_steps():

    # 1. Baselines
    probe("probe_baseline_val", ["random", OLD_CHECKPOINT], PRIMARY_SPLIT, "val")

    # 2. Ablation
    for ratio in ABLATION_RATIOS:
        train(ablation_name(ratio), ratio, ABLATION_EPOCHS, ABLATION_WARMUP)

    # 3. Ablation probes
    probe(
        "probe_ablation_val",
        [ablation_checkpoint(r) for r in ABLATION_RATIOS],
        PRIMARY_SPLIT,
        "val",
    )

    # 4. Select
    ratio = select_ratio()

    # 5. Main run
    train("main", ratio, MAIN_EPOCHS, MAIN_WARMUP)

    # 6. Final probes
    finals = ["random", OLD_CHECKPOINT, ablation_checkpoint(ratio), main_checkpoint()]

    probe("probe_final_test", finals, PRIMARY_SPLIT, "test")
    probe("probe_plain_val", finals, PLAIN_SPLIT, "val")
    probe("probe_plain_test", finals, PLAIN_SPLIT, "test")

    # 7. Summary
    write_summary(ratio)

    log("Pipeline finished")


if __name__ == "__main__":
    main()
