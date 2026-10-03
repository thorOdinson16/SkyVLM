# SkyVLM

A small vision-language model for satellite imagery, built from scratch on a single
RTX 5070 Laptop GPU (8 GB VRAM). It writes a structured caption for an aerial image, e.g.
`a satellite image of <main object>, surrounded by <object>; <object>`.

Everything is trained here: the vision encoder (masked image modeling), the language model
(FineWeb-Edu + captions), a CLIP-style alignment step, and the VLM that joins them.

## Results

Official SkyScript test captions, greedy decoding, all 30k images:

| Model | CIDEr-D | BLEU-4 | Main object exact | Surrounding F1 |
|---|---|---|---|---|
| **MIM ViT + CLIP alignment → VLM** | **1.90** | 0.475 | **21.3%** | 0.386 |
| MIM ViT → VLM | 1.75 | 0.464 | 19.6% | 0.372 |
| Random ViT → VLM (first 5k test images) | 0.69 | 0.323 | 5.9% | 0.264 |

Shuffled or blank images drop CIDEr-D to 0.09 / 0.04, so the model reads the image.

Sanity checks (MIM VLM, 30k test images): the main-object *category* (building, road, railway, ...) is right 61.5% of
the time versus 21% for always guessing the most common one, and the exact main object 19.6% versus 1.6%. All outputs
follow the caption template and 1.7% repeat a surrounding object. Typical errors are the right category with the wrong
specifics ("office" for an apartment block).

Image-text retrieval on 1,000 test pairs (a hit = same caption text). The VLM ranks captions by
log p(caption | image); image→text uses a PMI correction, because raw likelihood favours short captions.

| Model | Image→text R@1 | Text→image R@1 |
|---|---|---|
| Aligned VLM | 36.5 | 34.7 |
| MIM VLM | 30.2 | 30.7 |
| Our CLIP towers (ViT + LM) | 25.1 | 28.5 |
| SkyCLIP ViT-B/32 (external) | 8.6 | 6.8 |

### Vision encoder

RESISC45 linear probe (45 scene classes, 6,300 test images, frozen features, no RESISC45 data seen in training):

| Encoder | Test accuracy | Macro-F1 |
|---|---|---|
| Random ViT (untrained) | 43.5% | 0.425 |
| MIM (100 epochs, 200k unlabelled images) | 78.6% | 0.785 |
| **MIM + CLIP alignment** (mean pooling) | **88.4%** | **0.885** |
| MIM + CLIP alignment (CLS pooling) | 86.3% | 0.863 |
| ImageNet ViT-S/16 (supervised, ~14M labelled images) | 88.5% | 0.885 |

SkyScript 8-way category labels (geographic test split, Macro-F1; labels are noisy, so treat as secondary):

| Encoder | Frozen linear probe | Fine-tuned |
|---|---|---|
| Random ViT | 0.199 | n/a |
| MIM | 0.365 | 0.473 |
| ImageNet ViT-S/16 | 0.399 | 0.542 |

MIM mask-ratio ablation (5 epochs): 50% and 60% tied (Macro-F1 within 0.0015), 75% about one point lower; 60% used.

### Language model

| Metric | Value |
|---|---|
| Parameters / training tokens | 72.6M / 983M (30k steps) |
| FineWeb-Edu val perplexity | **21.7** (344 at step 500, 36.9 at 5k) |
| SkyScript caption val loss | 0.76 best (step 15k), 0.94 at the end |

Caption loss rose after step 15k because captions were 10% of every batch (~18 passes) and the model memorized them;
the final checkpoint is the better general LM, and `lm_step_015000.pt` is kept as the best-caption alternative.

### CLIP alignment

Val R@1 on 1,000 pairs went from 14.5% (epoch 0) to about 27% (epoch 4); it plateaued while train loss kept falling.
Final: image→text 27.8 / text→image 26.9 (val), 25.1 / 28.5 (test).

Caveats: one seed per condition. SkyCLIP is a general remote-sensing model and ours is a specialist
trained on these templated captions, so that row is a reference, not a claim of a better model.
MIM pretraining accounts for most of the VLM's accuracy; alignment adds a smaller gain.

## How it works

```
image ─► ViT (25.7M, MIM-pretrained, optionally CLIP-aligned)
          └─ 196 patch tokens ─► 2-layer MLP projector (512→1024→512)
                                   └─► LM (72.6M decoder) ─► caption
```

| Component | Parameters |
|---|---|
| Vision encoder (ViT) | 25.7M |
| MLP projector | 1.05M |
| Language model | 72.6M |
| **SkyVLM total** | **99.4M** |

- **Vision encoder** (`vit.py`, `mim.py`, `train_mim.py`): 8-layer ViT, 16×16 patches, 224 px; masked image
  modeling at a 60% mask ratio, 100 epochs on 200k SkyScript images.
- **Language model** (`lm.py`, `train_lm.py`): 20-layer decoder with RoPE and RMSNorm, 983M tokens
  (90% FineWeb-Edu, 10% SkyScript captions), 16k SentencePiece vocabulary.
- **CLIP alignment** (`clip_align.py`): symmetric InfoNCE between the ViT (mean of patch tokens) and a copy
  of the LM (EOS hidden state), with identical captions masked out of the negatives. The aligned ViT is saved
  in the MIM checkpoint format, so everything downstream loads it unchanged (`--vit-ckpt`).
- **VLM** (`vlm.py`, `train_vlm.py`): the LM sees `[img_start] 196 image tokens [img_end] caption` with a
  prefix-LM mask (image block bidirectional, caption causal); loss on caption tokens only.
  Stage 1 trains the projector and boundary tokens (3k steps); Stage 2 unfreezes everything (20k steps).

## Data and training cost

Data: 200k SkyScript training images, official val 5k / test 30k; 1M FineWeb-Edu documents (~1.15B tokens), 16k vocabulary.

| Stage | Compute | Peak VRAM |
|---|---|---|
| MIM encoder, 100 epochs | 13.6 h | n/a |
| Language model, 30k steps | 11.8 h | 2.8 GB |
| CLIP alignment, 15 epochs (11,715 steps) | ~5 h | 3.8 GB |
| VLM Stage 1 (3k steps) / Stage 2 (20k steps) | ~20 min / ~2.2 h | 5.9 GB |

## Repository layout

```
scripts/            models, training, evaluation, and the unattended pipelines
dataset/            images, manifests, tokenizer, FineWeb text (git-ignored; large)
checkpoints/        weights (git-ignored) plus logs and result tables (tracked)
  results_summary.md   encoder probe tables
  eval/                probe, VLM caption and retrieval results (json/csv)
select_skyscript_subset.py, finalize_skyscript.py, ...   200k-image selection
```

## Setup

Python 3.10, PyTorch with CUDA, plus `torchvision`, `pandas`, `sentencepiece`, `nltk`, `scikit-learn`, `timm`
(ImageNet reference). `open_clip_torch` is only needed to run the SkyCLIP comparison. Developed on Windows 11;
the long-run scripts use Windows APIs for keep-awake and RAM monitoring.

## Reproducing

Data (SkyScript, 200k balanced training images; official val 5k / test 30k fetched with HTTP range requests):

```
python select_skyscript_subset.py && python finalize_skyscript.py
python scripts/download_eval_images.py
python scripts/make_probe_split.py          # geographic probe splits
python scripts/train_tokenizer.py
python scripts/download_fineweb.py
```

Models:

```
python scripts/run_pipeline.py              # MIM mask-ratio ablation, 100-epoch MIM, probes
python scripts/run_lm.py                    # LM pretraining (~12 h)
python scripts/train_vlm.py --name stage1 ...        # VLM Stage 1, then Stage 2 (see run_clip_pipeline.py for the exact flags)
python scripts/run_clip_pipeline.py         # CLIP alignment → probe → aligned VLM → 5k test eval (resumable)
```

Evaluation:

```
python scripts/benchmark_probe.py <vit_ckpt>                          # RESISC45 linear probe
python scripts/eval_vlm.py --ckpt <vlm.pt> --split test --n 5000 [--controls]   # captions (--start / --merge for shards)
python scripts/eval_retrieval.py skyclip --split test --n 1000        # CLIP towers / SkyCLIP retrieval
python scripts/eval_vlm_retrieval.py --ckpt <vlm.pt> --split test --n 1000       # VLM-likelihood retrieval
```

Training scripts resume from their last save. The `run_*` drivers keep the machine awake, watch free RAM,
and restart a trainer that crashes or stalls.

## Notes and limitations

- References are OpenStreetMap-derived tags with several valid descriptions per image, so n-gram and exact-match
  scores understate quality. The main-object score is exact string match.
- SkyScript category labels are noisy; RESISC45 is the reliable measure of encoder quality.
- The LM saw the training captions ~18 times, so train loss is artificially low and val loss is a poor guide to
  caption quality; checkpoints were chosen on caption metrics.
