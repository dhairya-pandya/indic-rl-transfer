# Stage 2 smoke test (Kaggle T4, Qwen2.5-3B 4-bit QLoRA, 20 steps per arm, 512-token completions)

| Arm | sec/step | correctness reward (step 1 -> 20) | completions clipped at 512 |
|-----|----------|-----------------------------------|----------------------------|
| en    | 49.3 | 0.38 -> 0.79 | 2.5% |
| indic | 80.5 | 0.08 -> 0.00 (noisy, 0.08-0.21) | 62.5% |

Both arms ran without errors and adapters load in the vLLM eval. The Indic arm was starved by the
512-token cap: Bengali/Telugu reasoning takes about 1.5x the tokens of English, so most Indic
completions were cut off before the final answer. Full runs use 768 tokens for both arms.

Session time: 65 min including install.
