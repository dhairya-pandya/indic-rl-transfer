# Saturn Cloud smoke-test plan

Every stage below needs an explicit go-ahead before launch. Free GPU hours only.

## Hardware (read from the Saturn API, 2026-11-20)

Community account, AWS. Single-GPU options relevant here (list prices; which ones draw on the
free-hour allowance is **not exposed by the API** and must be confirmed in the Saturn UI):

| Size          | GPU             | VRAM  | bf16 | List $/h |
|---------------|-----------------|-------|------|----------|
| g4dnxlarge    | T4              | 16 GB | no   | 0.795    |
| g5xlarge      | A10G            | 24 GB | yes  | 1.50     |
| k0rdent A100  | A100 80GB       | 80 GB | yes  | 1.65     |

**Chosen: g4dnxlarge (T4)**, the size covered by the free hours. Always add `configs/t4.yaml`.
Gemma 3 is unstable in fp16; if its baseline gives NaN or empty output, drop it and keep Qwen.
Previous default: whichever single GPU the free allowance covers. On a T4 the script falls back to
fp16 automatically; drop `max_completion_len` to 512 and `num_generations` to 6 if it OOMs.

## Setup (on the server)

```bash
git clone <repo> && cd indic-rl-transfer && bash saturn/setup.sh
```

`HF_TOKEN` comes from a Saturn secret attached to the resource. It is never typed into a notebook.

## Stage 1: baseline (~1 GPU h)

```bash
python scripts/eval_mgsm.py --model Qwen/Qwen2.5-1.5B-Instruct --limit 250 --gpu-mem 0.9 --out outputs/eval/base_qwen
python scripts/eval_mgsm.py --model google/gemma-3-1b-it      --limit 250 --gpu-mem 0.9 --out outputs/eval/base_gemma
```

Decision: keep the model with the higher mean bn/te accuracy. If both are under 8% on bn and te,
GRPO will see too few correct samples; move to a 3B model (Qwen2.5-3B-Instruct) and re-size.

## Stage 2: smoke train (~0.5 GPU h per arm)

```bash
python scripts/train_grpo.py configs/base.yaml configs/arm_en.yaml    configs/t4.yaml configs/smoke.yaml --out outputs/runs/smoke_en
python scripts/train_grpo.py configs/base.yaml configs/arm_indic.yaml configs/t4.yaml configs/smoke.yaml --out outputs/runs/smoke_indic
python scripts/eval_mgsm.py --model <model> --adapter outputs/runs/smoke_en/final_adapter --limit 32 --out outputs/eval/smoke_en
```

Pass: no OOM, mean correctness reward > 0 in `log_history.json`, `step_times.json` written,
adapter loads in the eval script.

## Stage 3: sizing (Kaggle budget: 30 T4 hours total, hard cap 26 h used)

Runs on Kaggle (account `dhairyaankurpandya`, T4, 12 h max per session). Budget:

| Item | Hours |
|------|-------|
| Stage 1 baseline (2 models) | ~1.5 |
| Stage 2 smoke (2 arms x 20 steps + quick eval) | ~1 |
| Full runs, 2 arms | <= 2 x 8 |
| Final evals (base + 2 adapters, 250 x 4 langs, greedy + 1 sampled seed) | ~3 |
| Headroom (reruns, crashes) | >= 4 |

Steps per arm = min(400, floor(8 h x 3600 / sec_per_step)), with 250 as the floor. Below 250 steps,
cut `max_completion_len` to 384 before cutting steps further. The consistency-reward arm (Arm C)
is dropped unless both arms finish well under budget. Each full run saves an adapter every 50
steps to `/kaggle/working`, so a session timeout loses at most 50 steps.

## Kaggle storage

Big throwaway files go to `/tmp`: HF cache (`HF_HOME=/tmp/hf`), pip cache, repo clone, and
intermediate GRPO checkpoints. Only eval outputs and final/latest adapters go to `/kaggle/working`
(kept after the session, ~20 GB limit).
