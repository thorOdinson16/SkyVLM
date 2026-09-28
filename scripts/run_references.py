"""
Reference points for the MIM encoder, run after run_pipeline.py finishes.
Safe to re-run: finished steps are skipped.

  1. Val probe of the final 100-epoch encoder (primary split)
  2. ImageNet-pretrained ViT-S/16 through the same probe
     (primary val/test, plain val/test)
  3. Supervised fine-tuning references (approximate ceiling):
     final MIM encoder, and the ImageNet ViT-S
  4. checkpoints/results_summary.md with every result
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import run_pipeline as rp


IMAGENET = "timm:vit_small_patch16_224.augreg_in21k_ft_in1k"

FINETUNES = {
    "mim100": rp.main_checkpoint(),
    "imagenet_vits": IMAGENET,
}

SUMMARY = rp.CKPT / "results_summary.md"

PIPELINE_WAIT_SECONDS = 60


def wait_for_pipeline():
    log_path = rp.LOG_DIR / "pipeline.log"
    rp.log("References: waiting for the main pipeline to finish")

    while True:
        text = log_path.read_text(encoding="utf-8")
        tail = text[text.rfind("Pipeline started"):]

        if "Pipeline finished" in tail:
            rp.log("References: main pipeline finished, starting")
            return

        if "PIPELINE FAILED" in tail or "PIPELINE STOPPED" in tail:
            raise SystemExit("Main pipeline did not finish; references not run")

        time.sleep(PIPELINE_WAIT_SECONDS)


def finetune(run_name, init):
    source = f"finetune:{init}"

    if rp.has_result(source, rp.PRIMARY_SPLIT, "test"):
        rp.log(f"SKIP  finetune_{run_name}: results already recorded")
        return

    rp.run(
        f"finetune_{run_name}",
        ["scripts/finetune_classifier.py", init, "--run-name", run_name],
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

LABELS = {
    "random": "Random ViT (untrained)",
    rp.OLD_CHECKPOINT: "Old MIM, 1 epoch (old recipe)",
    **{
        rp.ablation_checkpoint(r): f"MIM {round(r * 100)}%, 5 epochs"
        for r in rp.ABLATION_RATIOS
    },
    rp.main_checkpoint(): "**MIM 60%, 100 epochs (final)**",
    IMAGENET: "ImageNet ViT-S/16 (supervised, reference)",
    f"finetune:{rp.main_checkpoint()}": "Fine-tuned: final MIM encoder",
    f"finetune:{IMAGENET}": "Fine-tuned: ImageNet ViT-S/16",
}


def write_summary():
    rows = rp.read_results()

    def table(split_file, eval_split, encoders, region="all"):
        lines = [
            "| Encoder | Method | Pooling | Macro-F1 | Accuracy | Images |",
            "|---|---|---|---|---|---|",
        ]
        for encoder in encoders:
            for r in rows:
                if (
                    r["encoder"] == encoder
                    and r["split_file"] == Path(split_file).name
                    and r["eval_split"] == eval_split
                    and r["region"] == region
                ):
                    method = "linear probe" if r["probe"] == rp_probe() else "fine-tune"
                    lines.append(
                        f"| {LABELS.get(encoder, encoder)} | {method} | {r['pooling']} | "
                        f"{r['macro_f1']} | {r['accuracy']} | {r['eval_images']} |"
                    )
        return "\n".join(lines)

    def region_table(split_file, eval_split, encoders):
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
                    and r["region"] != "all"
                ):
                    lines.append(
                        f"| {LABELS.get(encoder, encoder)} | {r['pooling']} | {r['region']} | "
                        f"{r['macro_f1']} | {r['accuracy']} | {r['eval_images']} |"
                    )
        return "\n".join(lines)

    ablations = [rp.ablation_checkpoint(r) for r in rp.ABLATION_RATIOS]
    finetunes = [f"finetune:{init}" for init in FINETUNES.values()]
    probed = ["random", rp.OLD_CHECKPOINT, rp.main_checkpoint(), IMAGENET]

    SUMMARY.write_text(
        f"# MIM encoder results\n\n"
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Raw rows: `checkpoints/eval/probe_results.csv`.\n\n"
        f"**Linear probe:** multinomial logistic regression on frozen, standardized features, "
        f"solved to convergence with Newton's method; L2 strength chosen on val Macro-F1. "
        f"**Fine-tune:** full network trained on the category labels, best epoch chosen on val "
        f"Macro-F1. Split: fixed 1-degree geographically disjoint grid-cell split (dense cells "
        f">= 0.5% forced to train). Neighbouring cells can be close, so this is not a guaranteed "
        f"minimum distance. MIM pretraining saw all 200k images (not their labels).\n\n"
        f"## Mask-ratio ablation (val, primary split)\n\n"
        f"{table(rp.PRIMARY_SPLIT, 'val', ['random', rp.OLD_CHECKPOINT] + ablations)}\n\n"
        f"50% and 60% performed comparably (difference in best Macro-F1 about 0.0015); 75% was "
        f"about 1 point lower. 60% was used for the 100-epoch run. The original selection "
        f"(`selected_mask_ratio.json`) was made with an earlier, under-trained 5-epoch SGD probe "
        f"that could not separate the three ratios; those numbers are archived in "
        f"`probe_results_sgd5ep_archive.csv`.\n\n"
        f"## Final comparison, val (primary split)\n\n"
        f"{table(rp.PRIMARY_SPLIT, 'val', probed + finetunes)}\n\n"
        f"## Final comparison, test (primary split)\n\n"
        f"{table(rp.PRIMARY_SPLIT, 'test', probed + finetunes)}\n\n"
        f"### Test by region\n\n"
        f"{region_table(rp.PRIMARY_SPLIT, 'test', [rp.main_checkpoint(), IMAGENET] + finetunes)}\n\n"
        f"## Sensitivity: plain geographic split (no dense-cell rule)\n\n"
        f"### val\n\n{table(rp.PLAIN_SPLIT, 'val', probed)}\n\n"
        f"### test\n\n{table(rp.PLAIN_SPLIT, 'test', probed)}\n"
    )

    rp.log(f"Results summary written: {SUMMARY}")


def rp_probe():
    import linear_probe
    return linear_probe.PROBE_METHOD


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    rp.LOG_DIR.mkdir(parents=True, exist_ok=True)
    rp.keep_awake()

    wait_for_pipeline()

    try:
        rp.probe("probe_final_val", [rp.main_checkpoint()], rp.PRIMARY_SPLIT, "val")

        for split_file in (rp.PRIMARY_SPLIT, rp.PLAIN_SPLIT):
            for eval_split in ("val", "test"):
                rp.probe(
                    f"probe_imagenet_{Path(split_file).stem}_{eval_split}",
                    [IMAGENET],
                    split_file,
                    eval_split,
                )

        for run_name, init in FINETUNES.items():
            finetune(run_name, init)

        write_summary()

    except rp.ResourceStop as error:
        rp.log(f"REFERENCES STOPPED: {error}. Free space, then re-run to resume.")
        raise SystemExit(1)
    except SystemExit as error:
        rp.log(f"REFERENCES FAILED: {error}")
        raise

    rp.log("References finished")


if __name__ == "__main__":
    main()
