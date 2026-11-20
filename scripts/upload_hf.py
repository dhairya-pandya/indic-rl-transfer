"""Upload the two LoRA adapters and all eval outputs to the Hugging Face Hub.

Uses the token from `hf auth login`; never reads or prints it.

Example:
    python scripts/upload_hf.py --user dhairyapandya --results results
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

BASE = "Qwen/Qwen2.5-3B-Instruct"
GH = "https://github.com/dhairya-pandya/indic-rl-transfer"

CARD = """---
base_model: {base}
library_name: peft
license: apache-2.0
language: [{lang_tags}]
tags: [grpo, lora, math, multilingual, bengali, telugu]
datasets: [ai4bharat/indic-gsm8k, juletxara/mgsm]
---

# Qwen2.5-3B-Instruct + GRPO LoRA ({arm_name})

LoRA adapter (r=32, 4-bit QLoRA training on a single T4) trained with GRPO for {steps} steps on
1,069 GSM8K problems {arm_desc}. Rewards: exact final-answer match (1.0) and a `#### <number>`
format bonus (0.2).

Part of the [indic-rl-transfer]({gh}) study: does RL on English math transfer to Bengali and
Telugu, compared with RL on those languages directly? Both adapters see the same problems,
steps and hyperparameters; only the prompt language differs.

## MGSM accuracy (%, greedy, 250 items per language)

{table}

Full eval outputs: [dhairyapandya/indic-rl-transfer-evals](https://huggingface.co/datasets/{user}/indic-rl-transfer-evals).

## Use

```python
from vllm import LLM
from vllm.lora.request import LoRARequest
llm = LLM("{base}", enable_lora=True, max_lora_rank=64)
llm.chat([[{{"role": "user", "content": "..."}}]], lora_request=LoRARequest("a", 1, "{repo}"))
```

Training data excludes the 250 GSM8K test problems MGSM is built from (checked at load time).
GSM8K test scores from this adapter are not meaningful because the training pool comes from
GSM8K test.
"""

EVAL_CARD = """---
license: apache-2.0
language: [en, bn, te, sw]
tags: [mgsm, evaluation, grpo]
---

# indic-rl-transfer eval outputs

Raw MGSM completions (greedy, 250 items x en/bn/te/sw) for Qwen2.5-3B-Instruct and two GRPO LoRA
adapters (English-trained and Bengali/Telugu-trained), plus baseline runs and training logs.
Each `completions.jsonl` row: `seed, lang, question, answer, completion`.

Code and report: {gh}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--adapters", type=Path, default=Path("outputs"))
    a = p.parse_args()
    api = HfApi()
    summary_md = (a.results / "summary.md").read_text().split("## Gain")[0].split("\n", 2)[2].strip()

    for arm, arm_name, desc, langs in [
        ("en", "English arm", "in English", "en"),
        ("indic", "Bengali/Telugu arm", "each shown in Bengali or Telugu (50/50)", "bn, te"),
    ]:
        repo = f"{a.user}/indic-rl-transfer-qwen2.5-3b-grpo-{arm}"
        adapter = a.adapters / f"full_{arm}" / "outputs" / f"full_{arm}" / "adapter"
        cfg = __import__("json").loads((a.results / f"full_{arm}" / "config.json").read_text())
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(adapter), commit_message="Add LoRA adapter")
        card = CARD.format(base=BASE, lang_tags=langs, arm_name=arm_name, arm_desc=desc, steps=cfg["steps"],
                           gh=GH, table=summary_md, user=a.user, repo=repo)
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, commit_message="Add model card")
        print("uploaded", repo)

    ds = f"{a.user}/indic-rl-transfer-evals"
    api.create_repo(ds, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=ds, repo_type="dataset", folder_path=str(a.results), commit_message="Add eval outputs",
                      allow_patterns=["**/*.jsonl", "**/*.json", "**/*.md", "*.png"])
    api.upload_file(path_or_fileobj=EVAL_CARD.format(gh=GH).encode(), path_in_repo="README.md", repo_id=ds,
                    repo_type="dataset", commit_message="Add dataset card")
    print("uploaded", ds)


if __name__ == "__main__":
    main()
