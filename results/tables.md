## A-OKVQA val (n=1145), restricted-letter accuracy %, [95% bootstrap CI]

| Method | k=288 (R=50%) | k=144 (R=75%) | k=58 (R=90%) | k=29 (R=95%) |
|---|---|---|---|---|
| Full 576 | 76.2 [73.7, 78.6] | | | |
| FastV (last-token attn) | 76.8 [74.2, 79.1] | 75.7 [73.2, 78.2] | 70.8 [68.1, 73.4] | 63.8 [61.1, 66.6] |
| CC-FastV 4x4 (proposed) | 76.6 [74.1, 79.0] | 76.2 [73.7, 78.7] | 72.8 [70.1, 75.3] | 67.7 [65.0, 70.4] |
| CC-FastV 2x2 (ablation) | 76.7 [74.2, 79.0] | 75.8 [73.3, 78.3] | 71.8 [69.2, 74.4] | 66.4 [63.7, 69.2] |
| Uniform grid (no attn) | 75.5 [73.1, 77.9] | 72.9 [70.4, 75.4] | 70.1 [67.6, 72.8] | 64.3 [61.7, 67.2] |
| Random (mean of 3 seeds) | 75.8 [73.5, 78.1] | 73.7 [71.3, 76.0] | 69.1 [66.8, 71.5] | 64.2 [61.9, 66.8] |
| FastV (paper text: mean received attn) | 72.8 [70.1, 75.3] | 67.2 [64.5, 70.0] | 58.0 [55.1, 61.0] | 54.8 [52.0, 57.8] |
| FastV, RoPE-free score (FEATHER-style) | 74.9 [72.4, 77.4] | 72.2 [69.5, 74.8] | 63.9 [61.1, 66.8] | 55.0 [52.1, 58.0] |

Paired difference vs FastV (points), [95% CI], McNemar p

| Method | k=288 | k=144 | k=58 | k=29 |
|---|---|---|---|---|
| CC-FastV 4x4 (proposed) | -0.2 [-0.9, +0.5] p=0.815 | +0.5 [-0.6, +1.7] p=0.441 | +1.9 [+0.5, +3.2] p=0.009 | +3.9 [+2.2, +5.7] p=0.000 |
| CC-FastV 2x2 (ablation) | -0.1 [-0.8, +0.7] p=1.000 | +0.1 [-0.9, +1.1] p=1.000 | +1.0 [-0.3, +2.3] p=0.200 | +2.6 [+1.0, +4.3] p=0.002 |
| Uniform grid (no attn) | -1.2 [-2.4, +0.0] p=0.070 | -2.8 [-4.6, -1.0] p=0.003 | -0.7 [-2.9, +1.4] p=0.575 | +0.5 [-1.9, +3.1] p=0.741 |
| Random (mean of 3 seeds) | -0.9 [-2.0, +0.1] | -2.0 [-3.5, -0.5] | -1.7 [-3.6, +0.2] | +0.5 [-1.6, +2.6] |
| FastV (paper text: mean received attn) | -4.0 [-5.6, -2.4] p=0.000 | -8.6 [-10.8, -6.3] p=0.000 | -12.8 [-15.6, -10.2] p=0.000 | -8.9 [-11.4, -6.5] p=0.000 |
| FastV, RoPE-free score (FEATHER-style) | -1.8 [-3.1, -0.5] p=0.006 | -3.5 [-5.2, -1.9] p=0.000 | -6.9 [-9.3, -4.5] p=0.000 | -8.7 [-11.1, -6.3] p=0.000 |

CC-FastV 4x4 minus attention-free controls (paired, points [95% CI]): grid@288: +1.0 [-0.1, +2.2]; rand@288: +0.8 [-0.1, +1.7]; grid@144: +3.3 [+1.6, +5.2]; rand@144: +2.6 [+1.1, +4.0]; grid@58: +2.6 [+0.6, +4.5]; rand@58: +3.6 [+1.8, +5.3]; grid@29: +3.4 [+1.0, +5.8]; rand@29: +3.5 [+1.4, +5.5]

Questions answered correctly without pruning but wrong after pruning / wrong without pruning but right after: fastv@288: 14/20; cc4@288: 11/15; grid@288: 29/21; fastv@144: 36/30; cc4@144: 29/29; grid@144: 75/37; fastv@58: 103/41; cc4@58: 79/39; grid@58: 108/38; fastv@29: 193/50; cc4@29: 149/51; grid@29: 193/56

Greedy first-token accuracy (official FastV metric: gold letter in decoded first token)

| Method | k=288 | k=144 | k=58 | k=29 |
|---|---|---|---|---|
| Full 576 | 76.2 | | | |
| FastV (last-token attn) | 76.8 | 75.7 | 70.8 | 63.8 |
| CC-FastV 4x4 (proposed) | 76.6 | 76.2 | 72.8 | 67.6 |
| Uniform grid (no attn) | 75.5 | 72.9 | 70.0 | 64.3 |
| FastV (paper text: mean received attn) | 72.8 | 67.0 | 57.9 | 54.7 |

Re-run on a second VM, share of identical predictions: fastv@288 100.0%, cc4@288 99.9%, fastv@144 99.9%, cc4@144 100.0%, fastv@58 99.8%, cc4@58 100.0%, fastv@29 99.8%, cc4@29 100.0%

## MME (n=2374): Perception / Cognition score (acc + acc+ summed over subtasks)

| Method | k=288 | k=144 | k=58 | k=29 |
|---|---|---|---|---|
| Full 576 | 1493 / 294 | | | |
| FastV (last-token attn) | 1487 / 307 | 1381 / 331 | 1178 / 315 | 953 / 291 |
| CC-FastV 4x4 (proposed) | 1504 / 303 | 1456 / 338 | 1279 / 307 | 1063 / 330 |
| Uniform grid (no attn) | 1513 / 324 | 1432 / 332 | 1251 / 319 | 1090 / 301 |
| Random (mean of 3 seeds) | 1491 / 311 | 1422 / 363 | 1254 / 317 | 1111 / 315 |

MME perception subtasks at k=58 (R=90%) and k=29 (R=95%)

| Method | existence | count | position | color | posters | celebrity | scene | landmark | artwork | OCR |
|---|---|---|---|---|---|---|---|---|---|---|
| Full 576 | 190 | 160 | 138 | 165 | 143 | 136 | 156 | 161 | 118 | 125 |
| fastv@58 | 125 | 97 | 82 | 133 | 120 | 132 | 137 | 120 | 99 | 132 |
| cc4@58 | 140 | 110 | 97 | 138 | 121 | 138 | 158 | 133 | 111 | 132 |
| grid@58 | 155 | 125 | 107 | 133 | 106 | 116 | 160 | 143 | 106 | 100 |
| rand@58 | 165 | 140 | 100 | 105 | 115 | 118 | 162 | 136 | 106 | 108 |
| fastv@29 | 110 | 62 | 63 | 92 | 110 | 105 | 113 | 99 | 82 | 118 |
| cc4@29 | 128 | 75 | 83 | 78 | 104 | 110 | 148 | 119 | 93 | 125 |
| grid@29 | 135 | 103 | 93 | 102 | 89 | 106 | 161 | 122 | 92 | 85 |
| rand@29 | 135 | 110 | 90 | 95 | 99 | 99 | 162 | 129 | 96 | 95 |

## Prefill latency (vision encoder + LLM prefill, batch 1, n=100 A-OKVQA questions, NVIDIA A100-SXM4-40GB)

| Method | k | median ms | p10-p90 ms | peak mem GiB |
|---|---|---|---|---|
| full | 576 | 81.9 | 81.1-82.7 | 13.23 |
| fastv | 288 | 69.0 | 66.4-70.1 | 13.30 |
| cc4 | 288 | 70.0 | 67.1-70.9 | 13.30 |
| grid | 288 | 67.3 | 64.8-68.2 | 13.23 |
| rand0 | 288 | 68.3 | 65.8-69.1 | 13.23 |
| fastv | 144 | 64.6 | 64.0-66.1 | 13.30 |
| cc4 | 144 | 65.5 | 64.7-66.6 | 13.30 |
| grid | 144 | 63.0 | 61.8-64.2 | 13.23 |
| rand0 | 144 | 63.7 | 63.0-64.8 | 13.23 |
| fastv | 58 | 64.2 | 63.1-65.6 | 13.30 |
| cc4 | 58 | 65.0 | 63.6-66.2 | 13.30 |
| grid | 58 | 62.7 | 61.2-64.8 | 13.23 |
| rand0 | 58 | 63.1 | 61.9-64.2 | 13.23 |
| fastv | 29 | 63.9 | 62.9-65.6 | 13.30 |
| cc4 | 29 | 65.0 | 63.8-66.7 | 13.30 |
| grid | 29 | 62.5 | 61.6-63.8 | 13.23 |
| rand0 | 29 | 63.5 | 62.5-65.3 | 13.23 |

## Theoretical LLM FLOPs ratio (FastV Eq.5 with mean sequence length n=678 tokens, K=2)

| k | 576 | 288 | 144 | 58 | 29 |
|---|---|---|---|---|---|
| FLOPs ratio | 100% | 59% | 40% | 28% | 24% |

## Spatial distribution of kept tokens on A-OKVQA (mean normalised row: 0=top, 1=bottom; share kept in top half)

| Method | k=288 | k=144 | k=58 | k=29 |
|---|---|---|---|---|
| fastv | 0.64 / 32% | 0.68 / 28% | 0.65 / 32% | 0.62 / 35% |
| fastv_avgq | 0.25 / 93% | 0.14 / 98% | 0.10 / 98% | 0.09 / 98% |
| cc4 | 0.51 / 50% | 0.51 / 50% | 0.56 / 44% | 0.58 / 41% |
| cc2 | 0.55 / 50% | 0.57 / 50% | 0.57 / 49% | 0.55 / 49% |
| grid | 0.50 / 50% | 0.50 / 50% | 0.50 / 50% | 0.50 / 48% |
| rand0 | 0.50 / 50% | 0.50 / 50% | 0.50 / 50% | 0.50 / 50% |
| fastvnr | 0.48 / 54% | 0.48 / 53% | 0.46 / 55% | 0.49 / 51% |

Last-token attention mass (layer 1, head-mean, per-image normalised) in top half: 42.0%; in last row: 6.3%
