"""Build results/summary.{md,json} and figures from base + two arm evals.

Example:
    python scripts/analyze.py --base results/baseline/Qwen2.5-3B-Instruct \
        --en results/full_en --indic results/full_indic --out results
"""

import argparse
import json
from pathlib import Path

from indic_rl.analysis import item_scores, paired_gain, transfer_ratio

LANGS = ["en", "bn", "te", "sw"]


def load_rows(path: Path, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r.get("seed", 0) == seed]


def acc(scores: dict, lang: str, idx: int = 0) -> float:
    vals = [v[idx] for k, v in scores.items() if k[0] == lang]
    return 100 * sum(vals) / len(vals)


def curves(run_dir: Path):
    h = [x for x in json.load(open(run_dir / "log_history.json")) if "reward" in x]
    return [x["step"] for x in h], [x["rewards/correctness_reward/mean"] for x in h], [x.get("completions/clipped_ratio", 0) for x in h]


def smooth(ys, k=10):
    return [sum(ys[max(0, i - k + 1): i + 1]) / len(ys[max(0, i - k + 1): i + 1]) for i in range(len(ys))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--en", type=Path, required=True)
    p.add_argument("--indic", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    S = {name: item_scores(load_rows(d / "completions.jsonl" if (d / "completions.jsonl").exists() else d / "eval" / "completions.jsonl"))
         for name, d in [("base", a.base), ("en", a.en), ("indic", a.indic)]}
    table = {name: {l: round(acc(s, l), 1) for l in LANGS} for name, s in S.items()}
    drift = {name: {l: round(acc(s, l, 1), 1) for l in LANGS} for name, s in S.items()}
    gains = {arm: {l: paired_gain(S["base"], S[arm], [l]) for l in LANGS} | {"bn+te": paired_gain(S["base"], S[arm], ["bn", "te"])}
             for arm in ("en", "indic")}
    tr = {g: transfer_ratio(S["base"], S["en"], S["indic"], langs) for g, langs in [("bn+te", ["bn", "te"]), ("bn", ["bn"]), ("te", ["te"])]}

    summary = {"accuracy": table, "english_reasoning_share": drift, "gains": gains, "transfer_ratio": tr}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))

    f = lambda g: f"{100 * g['gain']:+.1f} [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}]"
    md = ["## MGSM accuracy (%, greedy, 250 items per language)", "", "| Model | " + " | ".join(LANGS) + " |", "|---" * (len(LANGS) + 1) + "|"]
    names = {"base": "Qwen2.5-3B-Instruct", "en": "+ GRPO on English", "indic": "+ GRPO on Bengali/Telugu"}
    md += [f"| {names[n]} | " + " | ".join(f"{table[n][l]:.1f}" for l in LANGS) + " |" for n in ("base", "en", "indic")]
    md += ["", "## Gain over base (points, 95% paired bootstrap CI)", "", "| Arm | " + " | ".join(LANGS + ["bn+te"]) + " |", "|---" * (len(LANGS) + 2) + "|"]
    md += [f"| {names[arm]} | " + " | ".join(f(gains[arm][l]) for l in LANGS + ["bn+te"]) + " |" for arm in ("en", "indic")]
    md += ["", "## Transfer ratio (English-arm gain / Indic-arm gain)", ""]
    for g, t in tr.items():
        ci = f"[{t['lo']:.2f}, {t['hi']:.2f}]" if t["lo"] is not None else "(CI undefined: Indic-arm gain not reliably positive)"
        md.append(f"- {g}: {t['ratio']:.2f} {ci}")
    md += ["", "## Share of completions reasoning mostly in English (%)", "", "| Model | " + " | ".join(LANGS) + " |", "|---" * (len(LANGS) + 1) + "|"]
    md += [f"| {names[n]} | " + " | ".join(f"{drift[n][l]:.1f}" for l in LANGS) + " |" for n in ("base", "en", "indic")]
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = range(len(LANGS))
    for i, (n, c) in enumerate([("base", "#9aa0a6"), ("en", "#1a73e8"), ("indic", "#e8710a")]):
        axes[0].bar([j + (i - 1) * 0.27 for j in x], [table[n][l] for l in LANGS], 0.27, label=names[n], color=c)
    axes[0].set_xticks(list(x), LANGS)
    axes[0].set_ylabel("MGSM accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)
    for arm, c in [("en", "#1a73e8"), ("indic", "#e8710a")]:
        steps, corr, _ = curves(a.en if arm == "en" else a.indic)
        axes[1].plot(steps, smooth(corr), color=c, label=names[arm])
    axes[1].set_xlabel("GRPO step")
    axes[1].set_ylabel("train correctness (10-step mean)")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "figure_main.png", dpi=160)


if __name__ == "__main__":
    main()
