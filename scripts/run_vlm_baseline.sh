#!/bin/bash
# Random-ViT baseline: identical recipe to the main VLM (stage 1 then stage 2),
# but the ViT starts from random weights. Safe to re-run: skips finished stages
# and resumes stage 2 from its latest checkpoint.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
S1=checkpoints/vlm/random_stage1
S2=checkpoints/vlm/random_stage2

if [ ! -f $S1/vlm_final.pt ]; then
  python scripts/train_vlm.py --name random_stage1 --random-vit --max-steps 3000 \
    --lr-projector 1e-3 --batch-size 32 --eval-every 500 --save-every 500 --warmup-steps 200
fi

if [ ! -f $S2/vlm_final.pt ]; then
  python scripts/train_vlm.py --name random_stage2 --random-vit --resume \
    --init $S1/vlm_final.pt --train-lm --train-vit \
    --lr-projector 3e-4 --lr-lm 1e-4 --lr-vit 1e-5 \
    --batch-size 16 --grad-accum 2 --max-steps 20000 --warmup-steps 500 \
    --eval-every 1000 --save-every 1000 --keep-every 2500
fi
