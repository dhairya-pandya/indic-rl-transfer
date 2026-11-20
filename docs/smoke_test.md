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

Default target: whichever single GPU the free allowance covers. On a T4 the script falls back to
fp16 automatically; drop `max_completion_len` to 512 and `num_generations` to 6 if it OOMs.

## Setup (on the server)

```bash
git clone <repo> && cd indic-rl-transfer && bash saturn/setup.sh
```

`HF_TOKEN` comes from a Saturn secret attached to the resource. It is never typed into a notebook.

## Stage 1: baseline (~1 GPU h)

```bash
python scripts/eval_mgsm.py --model Qwen/Qwen2.5-1.5B-Instruct --limit 250 --out outputs/eval/base_qwen
python scripts/eval_mgsm.py --model google/gemma-3-1b-it      --limit 250 --out outputs/eval/base_gemma
```

Decision: keep the model with the higher mean bn/te accuracy. If both are under 8% on bn and te,
GRPO will see too few correct samples; move to a 3B model (Qwen2.5-3B-Instruct) and re-size.

## Stage 2: smoke train (~0.5 GPU h per arm)

```bash
python scripts/train_grpo.py configs/base.yaml configs/arm_en.yaml    configs/smoke.yaml --out outputs/runs/smoke_en
python scripts/train_grpo.py configs/base.yaml configs/arm_indic.yaml configs/smoke.yaml --out outputs/runs/smoke_indic
python scripts/eval_mgsm.py --model <model> --adapter outputs/runs/smoke_en/final_adapter --limit 32 --out outputs/eval/smoke_en
```

Pass: no OOM, mean correctness reward > 0 in `log_history.json`, `step_times.json` written,
adapter loads in the eval script.

## Stage 3: sizing

Full-run hours per arm = 400 x mean sec/step / 3600. Proceed to full runs only if
2 arms + 3-seed evals of base and both adapters fit in 60 GPU h (keeps more than half of the
150 free hours spare for the optional consistency-reward arm and reruns).
