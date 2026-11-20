"""Paired comparisons between eval runs on the same MGSM items."""

import random

from indic_rl.langid import SCRIPT_RANGES, english_ratio, script_ratio
from indic_rl.rewards import correctness_reward


def item_scores(rows: list[dict]) -> dict:
    """{(lang, question): (correct 0/1, english_reasoning 0/1, native_script 0/1)}"""
    out = {}
    for r in rows:
        c = correctness_reward([r["completion"]], [r["answer"]])[0]
        en = float(english_ratio(r["completion"]) > 0.5)
        nat = float(script_ratio(r["completion"], r["lang"]) > 0.5) if r["lang"] in SCRIPT_RANGES else float("nan")
        out[(r["lang"], r["question"])] = (c, en, nat)
    return out


def paired_gain(base: dict, new: dict, langs: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean accuracy gain (new - base) over items in `langs`, with a 95% paired bootstrap interval."""
    keys = sorted(k for k in base if k[0] in langs and k in new)
    if not keys:
        raise ValueError(f"no shared items for {langs}")
    diffs = [new[k][0] - base[k][0] for k in keys]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n_boot))
    return {"gain": sum(diffs) / len(diffs), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1], "n": len(keys)}


def transfer_ratio(base: dict, en_arm: dict, indic_arm: dict, langs: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    """(gain of English-trained arm) / (gain of Indic-trained arm) on `langs`, bootstrap over items."""
    keys = sorted(k for k in base if k[0] in langs and k in en_arm and k in indic_arm)
    g = lambda arm, ks: sum(arm[k][0] - base[k][0] for k in ks) / len(ks)
    point_den = g(indic_arm, keys)
    point = g(en_arm, keys) / point_den if point_den else float("nan")
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ks = rng.choices(keys, k=len(keys))
        den = g(indic_arm, ks)
        if den > 0:
            boots.append(g(en_arm, ks) / den)
    boots.sort()
    ok = len(boots) >= 0.9 * n_boot  # interval only meaningful if the denominator is reliably positive
    return {
        "ratio": point,
        "lo": boots[int(0.025 * len(boots))] if ok else None,
        "hi": boots[int(0.975 * len(boots)) - 1] if ok else None,
        "denominator_positive_share": len(boots) / n_boot,
    }
