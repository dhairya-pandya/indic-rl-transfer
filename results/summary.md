## MGSM accuracy (%, greedy, 250 items per language)

| Model | en | bn | te | sw |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 85.6 | 36.4 | 10.4 | 9.2 |
| + GRPO on English | 84.8 | 39.2 | 8.8 | 6.8 |
| + GRPO on Bengali/Telugu | 86.0 | 52.8 | 10.8 | 13.6 |

## Gain over base (points, 95% paired bootstrap CI)

| Arm | en | bn | te | sw | bn+te |
|---|---|---|---|---|---|
| + GRPO on English | -0.8 [-4.4, +3.2] | +2.8 [-2.8, +8.4] | -1.6 [-5.6, +2.0] | -2.4 [-6.0, +1.2] | +0.6 [-2.6, +3.8] |
| + GRPO on Bengali/Telugu | +0.4 [-2.8, +3.6] | +16.4 [+9.6, +23.2] | +0.4 [-4.0, +4.4] | +4.4 [+0.0, +8.8] | +8.4 [+4.4, +12.6] |

## Transfer ratio (English-arm gain / Indic-arm gain)

- bn+te: 0.07 [-0.43, 0.44]
- bn: 0.17 [-0.21, 0.47]
- te: -4.00 (CI undefined: Indic-arm gain not reliably positive)

## Share of completions reasoning mostly in English (%)

| Model | en | bn | te | sw |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 100.0 | 0.0 | 0.0 | 100.0 |
| + GRPO on English | 100.0 | 0.0 | 0.0 | 100.0 |
| + GRPO on Bengali/Telugu | 100.0 | 76.0 | 55.2 | 100.0 |

## Accuracy by reasoning language (%; n items; base accuracy on the same items)

| Model | bn, English reasoning | bn, Bengali reasoning | te, English reasoning | te, Telugu reasoning |
|---|---|---|---|---|
| + GRPO on English | - | 39.2 (n=250; base 36.4) | - | 8.8 (n=250; base 10.4) |
| + GRPO on Bengali/Telugu | 57.4 (n=190; base 36.3) | 38.3 (n=60; base 36.7) | 14.5 (n=138; base 11.6) | 6.2 (n=112; base 8.9) |
