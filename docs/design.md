# indic-rl-transfer: design

## Question
Do reasoning gains from RL (GRPO, verifiable rewards) on English math transfer to
low-resource Indic languages, and how close does English-only RL get to RL trained
directly on Indic-language data?

Headline target: "English RL recovers X% of the Bengali/Telugu gain of direct Indic RL."

## Goals
- Two matched GRPO runs, same model, same steps, same reward.
- Per-language MGSM accuracy before/after, plus reasoning-language drift.
- Short technical report, released LoRA adapters and raw eval outputs on HF (`codingpanda`).

## Non-goals / anti-goals
- No full grid of models/mixes. No claims beyond the measured languages.
- Anti-goal: a result driven by floor effects (base accuracy ~0 in bn/te) or by
  translation-test contamination. Either makes the headline meaningless.

## Constraints
- Saturn Cloud free hours only (~150 GPU h); no paid credits. GPU type to confirm on Saturn.
- Public/synthetic data only. Tokens via env vars / `hf auth login`, never committed.
- Public repo: no AI-tool mentions, plain commits.

## Approaches considered
1. Transfer only (EN RL, eval multilingual). Cheapest, but no reference point.
2. **Chosen:** transfer + direct-Indic arm at equal compute. Gives a ratio, not just a delta.
3. Full grid (2 models x 3 mixes x reward ablation). Too many runs for budget, dilutes story.

## Design
- **Step 0, baseline (~1 GPU h):** eval Qwen2.5-1.5B-Instruct and Gemma 3 1B-it on MGSM
  en/bn/te/sw (250 each, greedy). Pick model with clearly non-zero bn/te accuracy.
  If both near floor, step up to a ~3B model and re-size runs.
- **Train pool:** `ai4bharat/indic-gsm8k` (Apache-2.0; GSM8K test translated to Indic languages)
  minus the 250 problems that MGSM was built from = 1,069 problems. Both arms see the *same
  problems*; only the language differs.
- **Arm A (EN):** GRPO + LoRA on the 1,069 problems in English.
- **Arm B (Indic):** same 1,069 problems, each shown in Bengali or Telugu (seeded 50/50 split);
  same steps, group size, prompt count.
- **Optional Arm C:** Arm B + language-consistency reward, only if hours remain.
- **Rewards:** exact-match on final number (parsed, Indic/Arabic digits normalised);
  small format reward; consistency reward (Arm C) = fraction of reasoning in question language.
- **Eval:** MGSM test (en, bn, te, sw; sw = non-Indic low-resource control), 3 sampling seeds.
  Metrics: accuracy per language; share of reasoning traces in English (fastText/GlotLID);
  transfer ratio = (A_gain on bn/te) / (B_gain on bn/te).
- **Contamination check:** drop MGSM items by normalised English-question match (exactly 250
  removed), then assert no overlap at load time.
- **Stack:** TRL GRPOTrainer + Unsloth, vLLM generation if GPU allows. ~300-500 steps/run.
- **Smoke test first:** 20 steps, 32 eval items, to measure step time and size full runs.

## Left to the builder
Exact hyperparameters (lr, group size, max tokens), language-ID tool choice,
notebook vs script layout, Saturn job config.
