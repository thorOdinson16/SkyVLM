# MIM pipeline summary

Generated 2026-09-28 17:08

## Baselines (val, primary split)

| Encoder | Pooling | Region | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| random | cls | all | 0.2036 | 0.2882 | 22583 |
| random | mean | all | 0.1943 | 0.2853 | 22583 |
| checkpoints/mim/mim_epoch_01.pt | cls | all | 0.1980 | 0.2861 | 22583 |
| checkpoints/mim/mim_epoch_01.pt | mean | all | 0.1972 | 0.2876 | 22583 |

## Mask-ratio ablation (val, primary split)

| Encoder | Pooling | Region | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| checkpoints/mim/ablation_mr50/mim_epoch_005.pt | cls | all | 0.2563 | 0.3138 | 22583 |
| checkpoints/mim/ablation_mr50/mim_epoch_005.pt | mean | all | 0.2833 | 0.3394 | 22583 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | cls | all | 0.2473 | 0.3096 | 22583 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | mean | all | 0.2818 | 0.3368 | 22583 |
| checkpoints/mim/ablation_mr75/mim_epoch_005.pt | cls | all | 0.2512 | 0.3114 | 22583 |
| checkpoints/mim/ablation_mr75/mim_epoch_005.pt | mean | all | 0.2733 | 0.3259 | 22583 |

Selected ratio: **0.6**. Rule: highest val Macro-F1 (best pooling per ratio); ratios within 0.005 Macro-F1 of the best are tied and accuracy decides. Ratios tied with the best: [0.5, 0.6, 0.75].

## Final: 100-epoch encoder (test, primary split)

| Encoder | Pooling | Region | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| random | cls | all | 0.1989 | 0.2824 | 39787 |
| random | cls | Americas | 0.1988 | 0.2803 | 31972 |
| random | cls | Europe/Africa | 0.1973 | 0.2974 | 6124 |
| random | cls | Asia/Oceania | 0.1474 | 0.2679 | 1691 |
| random | mean | all | 0.1931 | 0.2794 | 39787 |
| random | mean | Americas | 0.1912 | 0.2752 | 31972 |
| random | mean | Europe/Africa | 0.1967 | 0.3014 | 6124 |
| random | mean | Asia/Oceania | 0.1460 | 0.2791 | 1691 |
| checkpoints/mim/mim_epoch_01.pt | cls | all | 0.1982 | 0.2822 | 39787 |
| checkpoints/mim/mim_epoch_01.pt | cls | Americas | 0.1978 | 0.2805 | 31972 |
| checkpoints/mim/mim_epoch_01.pt | cls | Europe/Africa | 0.1957 | 0.2936 | 6124 |
| checkpoints/mim/mim_epoch_01.pt | cls | Asia/Oceania | 0.1423 | 0.2732 | 1691 |
| checkpoints/mim/mim_epoch_01.pt | mean | all | 0.1979 | 0.2844 | 39787 |
| checkpoints/mim/mim_epoch_01.pt | mean | Americas | 0.1980 | 0.2830 | 31972 |
| checkpoints/mim/mim_epoch_01.pt | mean | Europe/Africa | 0.1927 | 0.2925 | 6124 |
| checkpoints/mim/mim_epoch_01.pt | mean | Asia/Oceania | 0.1436 | 0.2809 | 1691 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | cls | all | 0.2453 | 0.3067 | 39787 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | cls | Americas | 0.2458 | 0.3028 | 31972 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | cls | Europe/Africa | 0.2385 | 0.3325 | 6124 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | cls | Asia/Oceania | 0.1428 | 0.2868 | 1691 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | mean | all | 0.2799 | 0.3348 | 39787 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | mean | Americas | 0.2837 | 0.3345 | 31972 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | mean | Europe/Africa | 0.2654 | 0.3493 | 6124 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | mean | Asia/Oceania | 0.1444 | 0.2868 | 1691 |
| checkpoints/mim/main/mim_epoch_100.pt | cls | all | 0.3646 | 0.4178 | 39787 |
| checkpoints/mim/main/mim_epoch_100.pt | cls | Americas | 0.3703 | 0.4214 | 31972 |
| checkpoints/mim/main/mim_epoch_100.pt | cls | Europe/Africa | 0.3246 | 0.4120 | 6124 |
| checkpoints/mim/main/mim_epoch_100.pt | cls | Asia/Oceania | 0.2066 | 0.3714 | 1691 |
| checkpoints/mim/main/mim_epoch_100.pt | mean | all | 0.3651 | 0.4172 | 39787 |
| checkpoints/mim/main/mim_epoch_100.pt | mean | Americas | 0.3717 | 0.4218 | 31972 |
| checkpoints/mim/main/mim_epoch_100.pt | mean | Europe/Africa | 0.3215 | 0.4105 | 6124 |
| checkpoints/mim/main/mim_epoch_100.pt | mean | Asia/Oceania | 0.2075 | 0.3542 | 1691 |

## Sensitivity: plain geographic split

### val

| Encoder | Pooling | Region | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| random | cls | all | 0.2207 | 0.3200 | 24583 |
| random | mean | all | 0.2121 | 0.3193 | 24583 |
| checkpoints/mim/mim_epoch_01.pt | cls | all | 0.2253 | 0.3262 | 24583 |
| checkpoints/mim/mim_epoch_01.pt | mean | all | 0.2237 | 0.3248 | 24583 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | cls | all | 0.2621 | 0.3405 | 24583 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | mean | all | 0.3029 | 0.3748 | 24583 |
| checkpoints/mim/main/mim_epoch_100.pt | cls | all | 0.3883 | 0.4625 | 24583 |
| checkpoints/mim/main/mim_epoch_100.pt | mean | all | 0.3915 | 0.4600 | 24583 |

### test

| Encoder | Pooling | Region | Macro-F1 | Accuracy | Images |
|---|---|---|---|---|---|
| random | cls | all | 0.2191 | 0.3133 | 51897 |
| random | mean | all | 0.2123 | 0.3111 | 51897 |
| checkpoints/mim/mim_epoch_01.pt | cls | all | 0.2204 | 0.3168 | 51897 |
| checkpoints/mim/mim_epoch_01.pt | mean | all | 0.2215 | 0.3193 | 51897 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | cls | all | 0.2534 | 0.3279 | 51897 |
| checkpoints/mim/ablation_mr60/mim_epoch_005.pt | mean | all | 0.2897 | 0.3609 | 51897 |
| checkpoints/mim/main/mim_epoch_100.pt | cls | all | 0.3668 | 0.4336 | 51897 |
| checkpoints/mim/main/mim_epoch_100.pt | mean | all | 0.3701 | 0.4306 | 51897 |
