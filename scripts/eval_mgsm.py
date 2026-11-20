"""Evaluate a model (optionally with a LoRA adapter) on MGSM.

Example:
    python scripts/eval_mgsm.py --model Qwen/Qwen2.5-1.5B-Instruct --langs en bn te sw \
        --limit 250 --seeds 0 --temperature 0.0 --out outputs/eval/qwen_base
"""

import argparse
import json
from pathlib import Path

from indic_rl.data import load_mgsm
from indic_rl.evaluate import score_rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="path or HF id of a LoRA adapter")
    p.add_argument("--langs", nargs="+", default=["en", "bn", "te", "sw"])
    p.add_argument("--limit", type=int, default=250)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams  # lazy: keeps --help and tests CPU-only
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.model,
        enable_lora=args.adapter is not None,
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048,
        seed=0,
    )
    lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None

    data = {lang: load_mgsm(lang, args.limit) for lang in args.langs}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {"args": vars(args), "seeds": {}}
    for seed in args.seeds:
        sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=seed)
        seed_rows = []
        for lang, ds in data.items():
            outputs = llm.chat(list(ds["prompt"]), sp, lora_request=lora, use_tqdm=True)
            for ex, o in zip(ds, outputs):
                seed_rows.append({"seed": seed, "lang": lang, "question": ex["question"], "answer": ex["answer"], "completion": o.outputs[0].text})
        summary["seeds"][seed] = score_rows(seed_rows)
        rows += seed_rows

    with open(out / "completions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for seed, s in summary["seeds"].items():
        print(seed, {l: round(v["acc"], 3) for l, v in s.items()})


if __name__ == "__main__":
    main()
