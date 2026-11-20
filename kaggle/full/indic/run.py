"""Kaggle runner for one full GRPO arm (Qwen2.5-3B, 4-bit, T4) plus its final MGSM eval.

Generated per arm by kaggle/full/make.py. Storage: caches and checkpoints in /tmp; final adapter,
logs and eval outputs in /kaggle/working/outputs. Training stops itself before Kaggle's 12 h limit
and falls back to the newest checkpoint if the time cap hits first.
"""
import glob
import os
import shutil
import subprocess
import sys

ARM = "indic"
STEPS = 300
TRAIN_TIMEOUT = "10h"
os.environ.update(HF_HOME="/tmp/hf", PIP_CACHE_DIR="/tmp/pip", TMPDIR="/tmp", CUDA_VISIBLE_DEVICES="0")
REPO, OUT, PY = "/tmp/repo", "/kaggle/working/outputs", sys.executable
RUN = f"/tmp/runs/full_{ARM}"
os.makedirs(OUT, exist_ok=True)


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


sh("nvidia-smi --query-gpu=name,memory.total --format=csv && df -h /tmp /kaggle/working | tail -2")
sh(f"git clone -q --depth 1 https://github.com/dhairya-pandya/indic-rl-transfer.git {REPO}")
sh(f"{PY} -m pip install -q unsloth vllm")
sh(f"cd {REPO} && {PY} -m pip install -q -e . pytest && {PY} -m pytest -q -m 'not net'")
sh(f"{PY} -m pip freeze > {OUT}/pip_freeze.txt")

with open("/tmp/steps.yaml", "w") as f:
    f.write(f"steps: {STEPS}\n")
cfgs = f"configs/base.yaml configs/arm_{ARM}.yaml configs/t4.yaml /tmp/steps.yaml"
rc = sh(f"cd {REPO} && timeout {TRAIN_TIMEOUT} {PY} scripts/train_grpo.py {cfgs} --out {RUN}", check=False)
print(f"TRAIN_RC {rc}", flush=True)

dest = f"{OUT}/full_{ARM}"
os.makedirs(dest, exist_ok=True)
for f in ["config.json", "step_times.json", "log_history.json"]:
    if os.path.exists(f"{RUN}/{f}"):
        shutil.copy(f"{RUN}/{f}", dest)
adapter = f"{RUN}/final_adapter"
if not os.path.isdir(adapter):
    ckpts = sorted(glob.glob(f"{RUN}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    adapter = ckpts[-1] if ckpts else None
    print(f"NO_FINAL_ADAPTER; using {adapter}", flush=True)
    if adapter and os.path.exists(f"{adapter}/trainer_state.json"):
        shutil.copy(f"{adapter}/trainer_state.json", dest)
if adapter:
    shutil.copytree(adapter, f"{dest}/adapter", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("optimizer.pt", "rng_state*", "scheduler.pt"))
    sh(f"cd {REPO} && timeout 40m {PY} scripts/eval_mgsm.py --model Qwen/Qwen2.5-3B-Instruct "
       f"--adapter {dest}/adapter --dtype half --limit 250 --gpu-mem 0.85 --print-samples 2 "
       f"--out {OUT}/eval_full_{ARM}", check=False)
