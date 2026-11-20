# indic-rl-transfer

Do reasoning gains from reinforcement learning on English math transfer to low-resource Indic
languages? This repo trains a small instruction model with GRPO + LoRA in two matched arms and
measures accuracy and reasoning-language drift on MGSM.

| Arm | Training prompts | Everything else |
|-----|------------------|-----------------|
| EN    | 1,069 GSM8K problems in English | identical |
| Indic | the same 1,069 problems, each in Bengali or Telugu (seeded 50/50) | identical |

The headline metric is the **transfer ratio**: the Bengali/Telugu gain from English-only RL,
divided by the gain from training on Bengali/Telugu directly.

**Status:** scaffold done (rewards, eval harness, training script). Baseline and smoke runs next.

## Data

- Train: [`ai4bharat/indic-gsm8k`](https://huggingface.co/datasets/ai4bharat/indic-gsm8k) (Apache-2.0),
  minus the 250 problems MGSM was built from. The overlap check runs every time data is loaded.
- Eval: [`juletxara/mgsm`](https://huggingface.co/datasets/juletxara/mgsm), test split: en, bn, te, sw.
  Swahili is a non-Indic low-resource control.

## Layout

```
src/indic_rl/   rewards.py (answer parsing, Indic digits), langid.py (script share),
                data.py (train pool, MGSM), evaluate.py (per-language metrics), config.py
scripts/        train_grpo.py, eval_mgsm.py
configs/        base.yaml + arm_en / arm_indic / smoke overrides
saturn/         setup.sh for a Saturn Cloud GPU server
docs/           design.md, plan.md, smoke_test.md
```

## Quick start

```bash
pip install -e ".[dev]" && pytest -m "not net"            # CPU tests
bash saturn/setup.sh                                       # on the GPU box
python scripts/eval_mgsm.py --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/eval/base
python scripts/train_grpo.py configs/base.yaml configs/arm_en.yaml configs/smoke.yaml --out outputs/runs/smoke_en
```

## Limitations

Reasoning-language drift is measured by script, which is exact for Bengali and Telugu. It cannot
separate Swahili from English, so drift is not reported for Swahili. The train pool is translated
GSM8K *test* data, so GSM8K test scores from these adapters are not meaningful. Only MGSM is reported.
