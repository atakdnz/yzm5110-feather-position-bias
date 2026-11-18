# FastV reproduction + spatial-coverage-constrained visual token pruning (LLaVA-1.5-7B)

YZM5110 Multimodal Learning, research project. Report: `Akdeniz_YZM5110_Research.pdf` (also in this repository).

**Question.** At a fixed visual-token budget after layer 2 of LLaVA-1.5-7B, does constraining FastV's
attention-based token selection to cover the image spatially reduce its accuracy loss at high pruning ratios,
at matched latency?

- Paper reproduced: Chen et al., *An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference
  Acceleration for Large Vision-Language Models* (ECCV 2024), arXiv:2403.06764,
  official code https://github.com/pkunlp-icler/FastV (logic followed from commit `d165972`,
  `src/FastV/llava-hf/transformers/.../modeling_llama.py`, and `src/FastV/inference/eval/inference_aokvqa.py`).
- This repository contains an independent re-implementation for Hugging Face `transformers`; no FastV code is copied.

## What is in here

| Path | Content |
|---|---|
| `fastv_cov.py` | Model wrapper, FastV pruning forward pass, all selection rules, evaluation loops (A-OKVQA, MME), latency benchmark |
| `analyze.py` | Builds every table and figure in the report from `results/` |
| `scripts/setup_vm.sh`, `scripts/download.py` | Environment setup and data/model download on the GPU machine |
| `results/preds_aokvqa.jsonl` | Per-question predictions/correctness for every rule and budget (A-OKVQA val, 1145 q) |
| `results/preds_mme.jsonl` | Same for MME (2374 q) |
| `results/selection_aokvqa.npz` | How often each of the 576 patch positions was kept, per rule and budget; mean layer-1 attention map |
| `results/latency_aokvqa.json` | Prefill latency and peak memory per configuration |
| `results/norope/` | Extra control run (RoPE-free score) with its own predictions, selection counts and log |
| `results/meta_*.json`, `results/run_*.log` | Versions, model revision, GPU, runtime, sanity-check output |
| `results/summary.json`, `results/tables.md` | Output of `analyze.py` (all numbers quoted in the report) |
| `figures/` | Figures in the report |
| `report/report.html`, `report/fill.py` | Report source; `fill.py` inserts the tables from `results/summary.json`; `scripts/build_pdf.sh` renders the PDF with headless Chrome |

## Selection rules (all keep exactly *k* image tokens after layer *K*=2)

| name | rule |
|---|---|
| `fastv` | top-*k* head-averaged attention from the last prompt token at layer *K*-1 (official code) |
| `fastv_avgq` | top-*k* attention received averaged over all query rows (criterion as described in the paper text) |
| `fastvnr` | top-*k* last-token attention computed without RoPE (FEATHER's de-biased score; extra control) |
| `cc4` | **proposed**: 24x24 grid split into 4x4 blocks; each block keeps its floor(k/16) highest-`fastv`-score tokens, leftover budget by global score |
| `cc2` | same with 2x2 blocks (ablation) |
| `grid` | evenly spaced raster positions, no attention (coverage-only control) |
| `rand0..2` | uniform random positions, three seeds (attention-free control) |

Budgets *k* = 288, 144, 58, 29 (pruning ratio R = 50, 75, 90, 95 %). All hyper-parameters were fixed before any result was seen; nothing was tuned on the evaluation data.

## Environment (exact versions used for the reported runs)

Google Colab, NVIDIA A100-SXM4-40GB, CUDA 13.0, Python 3.13.16, torch 2.14.1, transformers 4.48.3,
tokenizers 0.21.4, huggingface-hub 0.35.3, datasets 4.8.5, accelerate 1.15.0, numpy 2.4.6
(see `requirements.txt`; `results/meta_*.json` records what each run actually used).
Model: `llava-hf/llava-1.5-7b-hf` at revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, fp16.

## Data access

Both datasets are public and are downloaded automatically from the Hugging Face hub by `scripts/download.py`:
- A-OKVQA (`HuggingFaceM4/A-OKVQA`, revision `d1b0efa3a436e9101dfbde3752db7607da696c35`, split `validation`, 1145 questions; annotations and COCO images under the licences stated on the dataset cards).
- MME (`lmms-lab/MME`, revision `d6c9023f017b564f7b3ccccf5348166bce8fdbcd`, split `test`, 2374 questions; use under the benchmark's terms).
No data is stored in this repository.

## Exact commands

On a CUDA GPU with >= 24 GB memory (all reported runs: one A100 40GB):

```bash
bash scripts/setup_vm.sh                       # pinned packages + model + datasets
python fastv_cov.py --task aokvqa --out results                                   # ~20 min on A100
python fastv_cov.py --task mme --methods fastv,cc4,grid,rand0 --sanity 0 --out results   # ~25 min
python fastv_cov.py --task aokvqa --latency 100 --out results                     # ~5 min, run on an idle GPU
```

Then, on any machine (CPU is enough):

```bash
pip install numpy scipy matplotlib
python analyze.py          # writes results/summary.json, results/tables.md, figures/*.png
```

**Small test** (a few minutes, also checks that the manual layer loop reproduces the model's own logits):

```bash
python fastv_cov.py --task aokvqa --limit 20 --out results/smoke
```

Extra control (RoPE-free FastV score, added after the main runs; FastV and CC-FastV are re-run in the same job as a determinism check):

```bash
python fastv_cov.py --task aokvqa --methods fastv,fastvnr,cc4 --sanity 2 --out results/norope
```

The A-OKVQA and MME runs in `results/` were run concurrently on one A100 (this affects only their runtime).
The control run and then the latency benchmark were run on a second, otherwise idle A100-40GB VM.
Reproducing the full numbers needs a GPU (about 1 A100-hour); `analyze.py` and every table/figure can be regenerated from the saved outputs without a GPU.
