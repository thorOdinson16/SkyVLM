#!/bin/sh
# Remaining 5k test shards for the aligned-encoder VLM (shard 0 = clipvit_test_s00 from run_clip_pipeline.py), then merge.
cd "$(dirname "$0")/.."
CK=checkpoints/vlm/clipvit_stage2/vlm_final.pt
for k in 1 2 3 4 5; do
  tag=clipvit_test_s0$k
  [ -f checkpoints/eval/vlm/${tag}_test.json ] && continue
  for try in 1 2 3; do
    python -u scripts/eval_vlm.py --ckpt $CK --split test --start $((k*5000)) --n 5000 --tag $tag && break
    sleep 60
  done
done
python scripts/eval_vlm.py --split test --merge clipvit_test_s00 clipvit_test_s01 clipvit_test_s02 clipvit_test_s03 clipvit_test_s04 clipvit_test_s05 --tag clipvit_merged_30k
