#!/bin/bash
# Environment used for all reported runs (Colab A100 40GB).
set -eu
python -m pip install -q 'transformers==4.48.3' 'tokenizers==0.21.4' 'huggingface-hub==0.35.3'
python scripts/download.py
