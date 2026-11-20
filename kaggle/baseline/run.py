"""Kaggle runner for Stage 1 baseline. Clones the public repo, installs eval deps, runs MGSM eval.

Storage: big, throwaway files (model weights, HF and pip caches, the repo clone) go to /tmp,
which has more space but is not saved. Only results (and later, final adapters) go to
/kaggle/working/outputs, which is kept and downloadable from the Output tab.
"""
import os
import subprocess
import sys

os.environ.update(HF_HOME="/tmp/hf", PIP_CACHE_DIR="/tmp/pip", TMPDIR="/tmp")
REPO = "/tmp/repo"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("nvidia-smi --query-gpu=name,memory.total --format=csv && df -h /tmp /kaggle/working | tail -2")
sh(f"git clone -q --depth 1 https://github.com/dhairya-pandya/indic-rl-transfer.git {REPO}")
sh(f"{sys.executable} -m pip install -q vllm datasets pyyaml pytest")
sh(f"cd {REPO} && {sys.executable} -m pip install -q -e . && {sys.executable} -m pytest -q -m 'not net'")
for model in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]:
    name = model.split("/")[-1]
    try:
        sh(
            f"cd {REPO} && CUDA_VISIBLE_DEVICES=0 timeout 40m {sys.executable} scripts/eval_mgsm.py "
            f"--model {model} --dtype half --limit 250 --gpu-mem 0.9 --print-samples 3 "
            f"--out /kaggle/working/outputs/eval/base_{name}"
        )
    except subprocess.CalledProcessError as e:
        print(f"EVAL_FAILED {model}: {e}", flush=True)
