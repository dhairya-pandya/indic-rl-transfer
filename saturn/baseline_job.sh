#!/usr/bin/env bash
# Stage 1 baseline on a T4. Results are printed to the job log (SUMMARY_JSON / SAMPLE_JSON lines).
set -euo pipefail
bash saturn/setup.sh
for m in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct; do
  timeout 25m python scripts/eval_mgsm.py --model "$m" --dtype half --limit 250 --gpu-mem 0.9 \
    --print-samples 3 --out "outputs/eval/base_$(basename "$m")" || echo "EVAL_FAILED $m"
done
