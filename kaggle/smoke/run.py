"""Kaggle runner for Stage 2 smoke test: 20 GRPO steps per arm on Qwen2.5-3B (4-bit), then a quick eval.

Storage: caches, repo clone and checkpoints in /tmp; logs, step times and final adapters copied
to /kaggle/working/outputs.
"""
import os
import shutil
import subprocess
import sys

os.environ.update(HF_HOME="/tmp/hf", PIP_CACHE_DIR="/tmp/pip", TMPDIR="/tmp", CUDA_VISIBLE_DEVICES="0")
REPO, OUT = "/tmp/repo", "/kaggle/working/outputs"
PY = sys.executable


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


sh("nvidia-smi --query-gpu=name,memory.total --format=csv && df -h /tmp /kaggle/working | tail -2")
sh(f"git clone -q --depth 1 https://github.com/dhairya-pandya/indic-rl-transfer.git {REPO}")
sh(f"{PY} -m pip install -q unsloth vllm")
sh(f"cd {REPO} && {PY} -m pip install -q -e . pytest && {PY} -m pytest -q -m 'not net'")
os.makedirs(OUT, exist_ok=True)
sh(f"{PY} -m pip freeze > {OUT}/pip_freeze.txt")

for arm in ["en", "indic"]:
    run = f"/tmp/runs/smoke_{arm}"
    cfgs = f"configs/base.yaml configs/arm_{arm}.yaml configs/t4.yaml configs/smoke.yaml"
    rc = sh(f"cd {REPO} && timeout 45m {PY} scripts/train_grpo.py {cfgs} --out {run}", check=False)
    if rc != 0:
        print(f"UNSLOTH_FAILED {arm} rc={rc}; retrying 5 steps without unsloth to get a timing", flush=True)
        rc = sh(f"cd {REPO} && printf 'steps: 5\\nload_in_4bit: false\\n' > /tmp/s5.yaml && timeout 25m {PY} "
                f"scripts/train_grpo.py {cfgs} /tmp/s5.yaml --no-unsloth --out {run}_nounsloth", check=False)
        run = f"{run}_nounsloth"
    dest = f"{OUT}/{os.path.basename(run)}"
    os.makedirs(dest, exist_ok=True)
    for f in ["config.json", "step_times.json", "log_history.json"]:
        if os.path.exists(f"{run}/{f}"):
            shutil.copy(f"{run}/{f}", dest)
    if os.path.isdir(f"{run}/final_adapter"):
        shutil.copytree(f"{run}/final_adapter", f"{dest}/final_adapter", dirs_exist_ok=True)
    print(f"ARM_DONE {arm} rc={rc}", flush=True)

adapter = f"{OUT}/smoke_en/final_adapter"
if os.path.isdir(adapter):
    sh(f"cd {REPO} && timeout 20m {PY} scripts/eval_mgsm.py --model Qwen/Qwen2.5-3B-Instruct --adapter {adapter} "
       f"--dtype half --limit 32 --gpu-mem 0.85 --out {OUT}/eval_smoke_en", check=False)
