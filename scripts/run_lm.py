"""
Unattended LM pretraining: keeps Windows awake, watches RAM/disk (via
run_pipeline.run) and resumes from checkpoints/lm/lm_latest.pt after a crash.

    python scripts/run_lm.py
"""

import run_pipeline as rp


LATEST = rp.CKPT / "lm" / "lm_latest.pt"

# Must match between fresh start and resumes (see train_lm.py docstring)
ARGS = ["scripts/train_lm.py"]


def main():
    rp.LOG_DIR.mkdir(parents=True, exist_ok=True)
    rp.keep_awake()

    rp.log("=" * 60)
    rp.log("LM pretraining started")

    for attempt in range(1, rp.MAX_ATTEMPTS + 1):
        resume = LATEST.exists()
        try:
            rp.run("train_lm", ARGS + (["--resume"] if resume else []), attempts=1)
            rp.log("LM pretraining finished")
            return
        except rp.ResourceStop as error:
            rp.log(f"LM STOPPED: {error}. Free space, then re-run to resume.")
            raise SystemExit(1)
        except SystemExit:
            if attempt == rp.MAX_ATTEMPTS:
                rp.log("LM FAILED: giving up after repeated failures")
                raise
            rp.log("Retrying LM pretraining from its latest checkpoint")


if __name__ == "__main__":
    main()
