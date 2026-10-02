#!/bin/bash
# Wait for the random-ViT baseline stage 2 to finish (resume it if the process died),
# then evaluate on the first 5k test images.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
F=checkpoints/vlm/random_stage2/vlm_final.pt

while [ ! -f $F ]; do
  sleep 30
  if [ ! -f $F ] && ! tasklist | grep -qi python; then
    echo "training process gone before final checkpoint; resuming"
    bash scripts/run_vlm_baseline.sh 2>&1 | grep --line-buffered -v Warn >> checkpoints/logs/vlm_random_baseline.log
  fi
done

python scripts/eval_vlm.py --ckpt $F --split test --start 0 --n 5000 --tag random_vit_test_s00 2>&1 | grep -v Warn
