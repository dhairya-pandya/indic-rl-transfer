# Stage 1 baseline (Kaggle T4, greedy, MGSM 250 per language, 2026-11-20)

| Model | en | bn | te | sw |
|-------|----|----|----|----|
| Qwen2.5-1.5B-Instruct | 76.0 | 14.4 | 3.2 | 5.6 |
| Qwen2.5-3B-Instruct   | 85.6 | 36.4 | 10.4 | 9.2 |

Reasoning stays in the question's script for bn (100%) and te (93-100%); Swahili prompts get
English reasoning (99-100%). Decision rule (both bn and te >= 8%): 1.5B fails on Telugu (3.2%),
3B passes. Training model: **Qwen2.5-3B-Instruct**.

Compute: 28 min of T4 time for both models, including install.
