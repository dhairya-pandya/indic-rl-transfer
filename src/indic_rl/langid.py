"""Script-based language share. Exact for Bengali/Telugu/Hindi, which have their own scripts.

Swahili and English share the Latin script, so Swahili drift is not measured here.
"""

import unicodedata

from indic_rl.rewards import completion_text

SCRIPT_RANGES = {"bn": (0x0980, 0x09FF), "te": (0x0C00, 0x0C7F), "hi": (0x0900, 0x097F)}


def _letters(text: str) -> list[str]:
    # Letters plus combining marks (Indic vowel signs are category M*), minus digits.
    return [c for c in text if c.isalpha() or unicodedata.category(c).startswith("M")]


def _in_script(c: str, lang: str) -> bool:
    if lang == "en":
        return c.isascii()
    lo, hi = SCRIPT_RANGES[lang]
    return lo <= ord(c) <= hi


def script_ratio(text: str, lang: str) -> float:
    letters = _letters(text)
    if not letters:
        return 0.0
    return sum(_in_script(c, lang) for c in letters) / len(letters)


def english_ratio(text: str) -> float:
    return script_ratio(text, "en")


def consistency_reward(completions, lang, **kw) -> list[float]:
    return [
        0.3 * script_ratio(completion_text(c), l) if l in SCRIPT_RANGES else 0.0
        for c, l in zip(completions, lang)
    ]
