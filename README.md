# indic-rl-transfer

Do reasoning gains from reinforcement learning on English math transfer to low-resource Indic
languages? This repo trains a small instruction model with GRPO + LoRA in two matched arms and
measures accuracy and reasoning-language drift on MGSM.

| Arm | Training prompts | Everything else |
|-----|------------------|-----------------|
| EN    | 1,069 GSM8K problems in English | identical |
| Indic | the same 1,069 problems, each in Bengali or Telugu (seeded 50/50) | identical |
| Indic + consistency | as Indic, plus a reward for reasoning in the question's script | identical |

**Report:** [docs/report.md](docs/report.md)

## Results (Qwen2.5-3B-Instruct, 250 GRPO steps, one T4)

| MGSM accuracy (%) | en | bn | te | sw | bn reasoning in English |
|---|---|---|---|---|---|
| Base | 85.6 | 36.4 | 10.4 | 9.2 | 0% |
| + GRPO on English | 84.8 | 39.2 | 8.8 | 6.8 | 0% |
| + GRPO on Bengali/Telugu | 86.0 | **52.8** | 10.8 | 13.6 | **76%** |
| + GRPO on Bengali/Telugu + consistency reward | 82.0 | 40.0 | 10.8 | 10.0 | 0% |

- **No transfer.** English RL gave no significant change on Bengali or Telugu. The transfer
  ratio against the Indic arm on bn+te is 0.07 (95% CI [-0.43, 0.44]).
- **The Indic arm learned to reason in English.** Its +16.4-point Bengali gain (CI [+9.6, +23.2])
  sits on the 76% of answers that switched to English (57.4% correct, against 36.3% for the base
  model on the same items).
- **Block the switch and the gain goes.** With a native-script reward, reasoning stays in
  Bengali/Telugu, and Bengali is +3.6 (not significant), level with English-only RL.

Adapters: [EN](https://huggingface.co/dhairyapandya/indic-rl-transfer-qwen2.5-3b-grpo-en),
[Indic](https://huggingface.co/dhairyapandya/indic-rl-transfer-qwen2.5-3b-grpo-indic),
[Indic + consistency](https://huggingface.co/dhairyapandya/indic-rl-transfer-qwen2.5-3b-grpo-indic-consistency).
Raw completions: [dhairyapandya/indic-rl-transfer-evals](https://huggingface.co/datasets/dhairyapandya/indic-rl-transfer-evals).
All numbers are in `results/` (`summary.md`, `summary.json`, per-run `eval/completions.jsonl`).

## Data

- Train: [`ai4bharat/indic-gsm8k`](https://huggingface.co/datasets/ai4bharat/indic-gsm8k) (Apache-2.0),
  minus the 250 problems MGSM was built from. The overlap check runs every time data is loaded.
- Eval: [`juletxara/mgsm`](https://huggingface.co/datasets/juletxara/mgsm), test split: en, bn, te, sw.
  Swahili is a non-Indic low-resource control.

## Layout

```
src/indic_rl/   rewards.py (answer parsing, Indic digits), langid.py (script share),
                data.py (train pool, MGSM), evaluate.py (per-language metrics), config.py
src/indic_rl/   analysis.py (paired bootstrap gains, transfer ratio)
scripts/        train_grpo.py, eval_mgsm.py, analyze.py, upload_hf.py
configs/        base.yaml, t4.yaml + arm_en / arm_indic / arm_indic_consistency / smoke overrides
kaggle/         the exact Kaggle T4 runners used (baseline, smoke, full arms)
saturn/         setup.sh for a Saturn Cloud GPU server (not used: no free hours)
results/        baseline, smoke and full-run evals, training logs, summary tables, figure
docs/           report.md, design.md, plan.md, smoke_test.md
```

## Quick start

```bash
pip install -e ".[dev]" && pytest -m "not net"            # CPU tests
pip install unsloth==2026.11.7 vllm==0.26.0                # on a CUDA GPU (T4 works)
python scripts/train_grpo.py configs/base.yaml configs/arm_indic.yaml configs/t4.yaml --out outputs/runs/indic
python scripts/eval_mgsm.py --model Qwen/Qwen2.5-3B-Instruct --adapter outputs/runs/indic/final_adapter \
    --dtype half --out outputs/eval/indic
python scripts/analyze.py --base results/baseline/Qwen2.5-3B-Instruct --en results/full_en \
    --indic results/full_indic --cons results/full_indic_consistency --out results
```

## Limitations

Reasoning-language drift is measured by script, which is exact for Bengali and Telugu. It cannot
separate Swahili from English, so drift is not reported for Swahili. The train pool is translated
GSM8K *test* data, so GSM8K test scores from these adapters are not meaningful. Only MGSM is reported.
One seed per arm, one model size, 250 steps; see the report for the full list.
