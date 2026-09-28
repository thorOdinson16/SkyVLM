# MIM encoder results

Generated 2026-09-28 19:01. Raw rows: `checkpoints/eval/probe_results.csv`.

**Linear probe:** multinomial logistic regression on frozen, standardized features, solved to convergence with Newton's method; L2 strength chosen on val Macro-F1. **Fine-tune:** full network trained on the category labels, best epoch chosen on val Macro-F1. Split: fixed 1-degree geographically disjoint grid-cell split (dense cells >= 0.5% forced to train). Neighbouring cells can be close, so this is not a guaranteed minimum distance. MIM pretraining saw all 200k images (not their labels).

## Mask-ratio ablation (val, primary split)

| Encoder | Method | Pooling | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| Random ViT (untrained) | linear probe | cls | 0.2036 | 0.2882 | 22583 |
| Random ViT (untrained) | linear probe | mean | 0.1943 | 0.2853 | 22583 |
| Old MIM, 1 epoch (old recipe) | linear probe | cls | 0.1980 | 0.2861 | 22583 |
| Old MIM, 1 epoch (old recipe) | linear probe | mean | 0.1972 | 0.2876 | 22583 |
| MIM 50%, 5 epochs | linear probe | cls | 0.2563 | 0.3138 | 22583 |
| MIM 50%, 5 epochs | linear probe | mean | 0.2833 | 0.3394 | 22583 |
| MIM 60%, 5 epochs | linear probe | cls | 0.2473 | 0.3096 | 22583 |
| MIM 60%, 5 epochs | linear probe | mean | 0.2818 | 0.3368 | 22583 |
| MIM 75%, 5 epochs | linear probe | cls | 0.2512 | 0.3114 | 22583 |
| MIM 75%, 5 epochs | linear probe | mean | 0.2733 | 0.3259 | 22583 |

50% and 60% performed comparably (difference in best Macro-F1 about 0.0015); 75% was about 1 point lower. 60% was used for the 100-epoch run. The original selection (`selected_mask_ratio.json`) was made with an earlier, under-trained 5-epoch SGD probe that could not separate the three ratios; those numbers are archived in `probe_results_sgd5ep_archive.csv`.

## Final comparison, val (primary split)

| Encoder | Method | Pooling | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| Random ViT (untrained) | linear probe | cls | 0.2036 | 0.2882 | 22583 |
| Random ViT (untrained) | linear probe | mean | 0.1943 | 0.2853 | 22583 |
| Old MIM, 1 epoch (old recipe) | linear probe | cls | 0.1980 | 0.2861 | 22583 |
| Old MIM, 1 epoch (old recipe) | linear probe | mean | 0.1972 | 0.2876 | 22583 |
| **MIM 60%, 100 epochs (final)** | linear probe | cls | 0.3633 | 0.4162 | 22583 |
| **MIM 60%, 100 epochs (final)** | linear probe | mean | 0.3671 | 0.4163 | 22583 |
| ImageNet ViT-S/16 (supervised, reference) | linear probe | cls | 0.3966 | 0.4343 | 22583 |
| ImageNet ViT-S/16 (supervised, reference) | linear probe | mean | 0.3868 | 0.4212 | 22583 |
| Fine-tuned: final MIM encoder | fine-tune | mean | 0.4649 | 0.4912 | 22583 |
| Fine-tuned: ImageNet ViT-S/16 | fine-tune | mean | 0.5340 | 0.5492 | 22583 |

## Final comparison, test (primary split)

| Encoder | Method | Pooling | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| Random ViT (untrained) | linear probe | cls | 0.1989 | 0.2824 | 39787 |
| Random ViT (untrained) | linear probe | mean | 0.1931 | 0.2794 | 39787 |
| Old MIM, 1 epoch (old recipe) | linear probe | cls | 0.1982 | 0.2822 | 39787 |
| Old MIM, 1 epoch (old recipe) | linear probe | mean | 0.1979 | 0.2844 | 39787 |
| **MIM 60%, 100 epochs (final)** | linear probe | cls | 0.3646 | 0.4178 | 39787 |
| **MIM 60%, 100 epochs (final)** | linear probe | mean | 0.3651 | 0.4172 | 39787 |
| ImageNet ViT-S/16 (supervised, reference) | linear probe | cls | 0.3986 | 0.4370 | 39787 |
| ImageNet ViT-S/16 (supervised, reference) | linear probe | mean | 0.3922 | 0.4257 | 39787 |
| Fine-tuned: final MIM encoder | fine-tune | mean | 0.4726 | 0.5019 | 39787 |
| Fine-tuned: ImageNet ViT-S/16 | fine-tune | mean | 0.5424 | 0.5568 | 39787 |

### Test by region

| Encoder | Pooling | Region | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| **MIM 60%, 100 epochs (final)** | cls | Americas | 0.3703 | 0.4214 | 31972 |
| **MIM 60%, 100 epochs (final)** | cls | Europe/Africa | 0.3246 | 0.4120 | 6124 |
| **MIM 60%, 100 epochs (final)** | cls | Asia/Oceania | 0.2066 | 0.3714 | 1691 |
| **MIM 60%, 100 epochs (final)** | mean | Americas | 0.3717 | 0.4218 | 31972 |
| **MIM 60%, 100 epochs (final)** | mean | Europe/Africa | 0.3215 | 0.4105 | 6124 |
| **MIM 60%, 100 epochs (final)** | mean | Asia/Oceania | 0.2075 | 0.3542 | 1691 |
| ImageNet ViT-S/16 (supervised, reference) | cls | Americas | 0.4058 | 0.4426 | 31972 |
| ImageNet ViT-S/16 (supervised, reference) | cls | Europe/Africa | 0.3533 | 0.4268 | 6124 |
| ImageNet ViT-S/16 (supervised, reference) | cls | Asia/Oceania | 0.2092 | 0.3666 | 1691 |
| ImageNet ViT-S/16 (supervised, reference) | mean | Americas | 0.3982 | 0.4292 | 31972 |
| ImageNet ViT-S/16 (supervised, reference) | mean | Europe/Africa | 0.3539 | 0.4250 | 6124 |
| ImageNet ViT-S/16 (supervised, reference) | mean | Asia/Oceania | 0.2027 | 0.3607 | 1691 |
| Fine-tuned: final MIM encoder | mean | Americas | 0.4798 | 0.5083 | 31972 |
| Fine-tuned: final MIM encoder | mean | Europe/Africa | 0.4286 | 0.4863 | 6124 |
| Fine-tuned: final MIM encoder | mean | Asia/Oceania | 0.2653 | 0.4370 | 1691 |
| Fine-tuned: ImageNet ViT-S/16 | mean | Americas | 0.5510 | 0.5646 | 31972 |
| Fine-tuned: ImageNet ViT-S/16 | mean | Europe/Africa | 0.4799 | 0.5310 | 6124 |
| Fine-tuned: ImageNet ViT-S/16 | mean | Asia/Oceania | 0.3407 | 0.5015 | 1691 |

## Sensitivity: plain geographic split (no dense-cell rule)

### val

| Encoder | Method | Pooling | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| Random ViT (untrained) | linear probe | cls | 0.2207 | 0.3200 | 24583 |
| Random ViT (untrained) | linear probe | mean | 0.2121 | 0.3193 | 24583 |
| Old MIM, 1 epoch (old recipe) | linear probe | cls | 0.2253 | 0.3262 | 24583 |
| Old MIM, 1 epoch (old recipe) | linear probe | mean | 0.2237 | 0.3248 | 24583 |
| **MIM 60%, 100 epochs (final)** | linear probe | cls | 0.3883 | 0.4625 | 24583 |
| **MIM 60%, 100 epochs (final)** | linear probe | mean | 0.3915 | 0.4600 | 24583 |
| ImageNet ViT-S/16 (supervised, reference) | linear probe | cls | 0.4243 | 0.4776 | 24583 |
| ImageNet ViT-S/16 (supervised, reference) | linear probe | mean | 0.4191 | 0.4709 | 24583 |

### test

| Encoder | Method | Pooling | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| Random ViT (untrained) | linear probe | cls | 0.2191 | 0.3133 | 51897 |
| Random ViT (untrained) | linear probe | mean | 0.2123 | 0.3111 | 51897 |
| Old MIM, 1 epoch (old recipe) | linear probe | cls | 0.2204 | 0.3168 | 51897 |
| Old MIM, 1 epoch (old recipe) | linear probe | mean | 0.2215 | 0.3193 | 51897 |
| **MIM 60%, 100 epochs (final)** | linear probe | cls | 0.3668 | 0.4336 | 51897 |
| **MIM 60%, 100 epochs (final)** | linear probe | mean | 0.3701 | 0.4306 | 51897 |
| ImageNet ViT-S/16 (supervised, reference) | linear probe | cls | 0.4038 | 0.4483 | 51897 |
| ImageNet ViT-S/16 (supervised, reference) | linear probe | mean | 0.3961 | 0.4401 | 51897 |

## External benchmark: NWPU-RESISC45 linear probe

45 aerial scene classes with clean labels; standard 18,900 / 6,300 / 6,300 train / val / test split
(`timm/resisc45`). Frozen features, standardized with train statistics; multinomial logistic
regression fitted to convergence with full-batch L-BFGS on the GPU (float32, then float64 to
max |grad| < 1e-6); L2 strength chosen on val accuracy over the same grid as above. All selected
fits converged. Raw rows: `checkpoints/eval/benchmark_results.csv`.

| Encoder | Pooling | Val accuracy | Test accuracy | Test Macro-F1 |
|---|---|---|---|---|
| Random ViT (untrained) | cls | 0.4522 | 0.4351 | 0.4246 |
| Random ViT (untrained) | mean | 0.4460 | 0.4341 | 0.4258 |
| **MIM 60%, 100 epochs (ours)** | cls | 0.7959 | 0.7848 | 0.7842 |
| **MIM 60%, 100 epochs (ours)** | mean | 0.8002 | **0.7859** | 0.7848 |
| ImageNet ViT-S/16 (supervised reference) | cls | 0.8960 | **0.8851** | 0.8845 |
| ImageNet ViT-S/16 (supervised reference) | mean | 0.9006 | 0.8790 | 0.8783 |

Our encoder closes about 78% of the gap between random features and the supervised ImageNet ViT-S
(0.786 vs 0.435 and 0.885 test accuracy). The ImageNet model was trained on roughly 70x more
images, all labelled; ours saw 200k unlabelled SkyScript images and no RESISC45 data.
