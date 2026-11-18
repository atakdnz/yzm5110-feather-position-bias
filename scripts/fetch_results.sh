#!/bin/bash
# Copies the saved outputs from the two Colab VMs into results/ (used once after the runs).
set -u
cd "$(dirname "$0")/.."
mkdir -p results/norope
dl() { colab download -s "$1" "$2" "$3" 2>&1 | grep -i -e error -e fail; }
if [ "${1:-}" = "main" ]; then
  for f in preds_aokvqa.jsonl selection_aokvqa.npz meta_aokvqa.json run_aokvqa.log preds_mme.jsonl selection_mme.npz meta_mme.json run_mme.log; do
    dl akd-cov /content/fc/results/$f results/$f; done
fi
if [ "${1:-}" = "vm2" ]; then
  for f in preds_aokvqa.jsonl selection_aokvqa.npz meta_aokvqa.json run_aokvqa_norope.log; do
    dl akd-cov2 /content/fc/results/norope/$f results/norope/$f; done
  for f in latency_aokvqa.json run_latency.log; do dl akd-cov2 /content/fc/results/$f results/$f; done
fi
ls -la results results/norope
