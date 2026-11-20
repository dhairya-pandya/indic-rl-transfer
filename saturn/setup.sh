#!/usr/bin/env bash
# Run once inside a Saturn Cloud Jupyter server or job, from the repo root.
# Expects HF_TOKEN in the environment (Saturn secret); never echo it.
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q -r requirements.txt
pip install -q -e .
python -m pytest -q -m "not net"
python -c "import torch; print('cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported())"
