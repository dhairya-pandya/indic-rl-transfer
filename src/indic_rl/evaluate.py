"""Score MGSM completions: accuracy and reasoning-language drift per language."""

from collections import defaultdict

from indic_rl.langid import SCRIPT_RANGES, english_ratio, script_ratio
from indic_rl.rewards import correctness_reward


def score_rows(rows: list[dict]) -> dict:
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    out = {}
    for lang, rs in sorted(by_lang.items()):
        comps = [r["completion"] for r in rs]
        correct = correctness_reward(comps, [r["answer"] for r in rs])
        n = len(rs)
        out[lang] = {
            "n": n,
            "acc": sum(correct) / n,
            "en_reasoning_share": sum(english_ratio(c) > 0.5 for c in comps) / n,
            # Only defined for languages with their own script.
            "native_script_share": (
                sum(script_ratio(c, lang) > 0.5 for c in comps) / n if lang in SCRIPT_RANGES else None
            ),
        }
    return out
