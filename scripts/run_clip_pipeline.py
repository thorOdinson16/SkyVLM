"""
Unattended pipeline: CLIP alignment -> encoder probe -> VLM on the aligned ViT -> test eval.

    python scripts/run_clip_pipeline.py

Run it detached (it outlives the shell that started it). Safe to re-run: finished
steps are skipped, interrupted trainings resume from their last save. A RAM
watchdog kills and resumes a training when free memory stays critically low
(the trainers have been seen growing past 9 GB). New names are used everywhere,
so earlier checkpoints (checkpoints/mim, lm, vlm/stage1|stage2, ...) are never touched.
"""

import csv
import ctypes
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / "checkpoints" / "logs" / "clip_pipeline.log"

CLIP_NAME = "main"
CLIP_EPOCHS = 15
SWEEP_LRS = ["5e-5", "1e-4", "3e-4"]
DEFAULT_LR = "1e-4"

LOW_RAM_MB = 1200
LOW_RAM_POLLS = 3
MAX_ATTEMPTS = 8


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class MemStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_ram_mb():
    s = MemStatus()
    s.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys // 2**20


def kill_tree(proc):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_step(name, cmd, done, logfile=None, attempts=MAX_ATTEMPTS):
    """Run `cmd` until `done` (a Path) exists, restarting on crash / low RAM."""

    if done is not None and done.exists():
        log(f"SKIP {name} (already done)")
        return True

    for attempt in range(1, attempts + 1):
        log(f"START {name} (attempt {attempt}) free RAM {free_ram_mb()} MB")

        out = open(logfile or ROOT / "checkpoints" / "logs" / f"clip_pipeline_{name}.log", "a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)

        low = 0
        killed = False

        while proc.poll() is None:
            time.sleep(20)
            low = low + 1 if free_ram_mb() < LOW_RAM_MB else 0

            if low >= LOW_RAM_POLLS:
                log(f"LOW RAM ({free_ram_mb()} MB) during {name}; killing and resuming")
                kill_tree(proc)
                killed = True
                break

        out.close()
        rc = proc.poll()

        if not killed and rc == 0 and (done is None or done.exists()):
            log(f"DONE {name}")
            return True

        log(f"{name} ended early (rc={rc}, killed={killed}); retrying in 45 s")
        time.sleep(45)

    log(f"FAILED {name} after {attempts} attempts")
    return False


# ------------------------------------------------------------
# Step 0: choose the ViT learning rate from the sweep
# ------------------------------------------------------------

def sweep_score(lr):
    path = ROOT / "checkpoints" / "clip" / f"sweep_{lr}" / "train_log.csv"
    if not path.exists():
        return None, 0

    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None, 0

    last = rows[-1]
    return (float(last["i2t_R@1"]) + float(last["t2i_R@1"])) / 2, len(rows)


def choose_lr():
    log("Waiting for the learning-rate sweep (3 runs x 4 epochs)")
    idle_since = time.time()
    last_state = None

    while True:
        scores = {lr: sweep_score(lr) for lr in SWEEP_LRS}

        if all(n >= 4 for _, n in scores.values()):
            break

        state = tuple(n for _, n in scores.values())
        if state != last_state:
            last_state, idle_since = state, time.time()

        if time.time() - idle_since > 45 * 60:        # sweep stalled or killed
            log("Sweep stalled; using what finished")
            break

        time.sleep(60)

    done = {lr: s for lr, (s, n) in scores.items() if s is not None and n >= 4}
    log(f"Sweep results (mean R@1, val-1000): {scores}")

    if not done:
        log(f"No finished sweep runs; using default lr {DEFAULT_LR}")
        return DEFAULT_LR

    best = max(done, key=done.get)
    log(f"Chosen ViT lr: {best}")
    return best


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("CLIP pipeline started")

    clip_dir = ROOT / "checkpoints" / "clip" / CLIP_NAME
    vit_final = clip_dir / "vit_final.pt"

    # The sweep only matters while the alignment itself still has to run
    lr = DEFAULT_LR if vit_final.exists() else choose_lr()

    # 1. CLIP alignment (resumable per epoch)
    ok = run_step(
        "clip_align",
        [PY, "-u", "scripts/clip_align.py", "--name", CLIP_NAME, "--epochs", str(CLIP_EPOCHS),
         "--lr-vit", lr, "--workers", "6", "--resume"],
        vit_final,
    )
    if not ok:
        return

    # 2. Does the aligned encoder keep its scene features? (MIM encoder: 78.6% test accuracy)
    marker = clip_dir / "probe_done"
    if run_step("probe", [PY, "-u", "scripts/benchmark_probe.py", str(vit_final)], marker if marker.exists() else None, attempts=2):
        marker.write_text("done")

    # 3. VLM on the aligned ViT: same recipe as the MIM VLM
    s1 = ROOT / "checkpoints" / "vlm" / "clipvit_stage1" / "vlm_final.pt"
    s2 = ROOT / "checkpoints" / "vlm" / "clipvit_stage2" / "vlm_final.pt"

    ok = run_step(
        "vlm_stage1",
        [PY, "-u", "scripts/train_vlm.py", "--name", "clipvit_stage1", "--vit-ckpt", str(vit_final),
         "--max-steps", "3000", "--lr-projector", "1e-3", "--batch-size", "32",
         "--eval-every", "500", "--save-every", "500", "--warmup-steps", "200", "--workers", "6", "--resume"],
        s1,
    )
    if not ok:
        return

    ok = run_step(
        "vlm_stage2",
        [PY, "-u", "scripts/train_vlm.py", "--name", "clipvit_stage2", "--vit-ckpt", str(vit_final),
         "--init", str(s1), "--resume", "--train-lm", "--train-vit",
         "--lr-projector", "3e-4", "--lr-lm", "1e-4", "--lr-vit", "1e-5",
         "--batch-size", "16", "--grad-accum", "2", "--max-steps", "20000", "--warmup-steps", "500",
         "--eval-every", "1000", "--save-every", "1000", "--keep-every", "2500", "--workers", "6"],
        s2,
    )
    if not ok:
        return

    # 4. Caption evaluation on the same first 5k test images as the MIM VLM
    run_step(
        "eval_test5k",
        [PY, "-u", "scripts/eval_vlm.py", "--ckpt", str(s2), "--split", "test",
         "--start", "0", "--n", "5000", "--tag", "clipvit_test_s00"],
        ROOT / "checkpoints" / "eval" / "vlm" / "clipvit_test_s00_test.json",
    )

    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
