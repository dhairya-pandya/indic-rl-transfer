"""Verifiable rewards for math word problems in several scripts."""

import re

# Bengali, Telugu and Devanagari digit blocks map to ASCII 0-9.
_DIGIT_MAP = {}
for start in (0x09E6, 0x0C66, 0x0966):
    for i in range(10):
        _DIGIT_MAP[chr(start + i)] = str(i)
_DIGIT_TABLE = str.maketrans(_DIGIT_MAP)

# Minus only when not glued to a preceding word/digit ("3-4" is a range); commas only as thousands groups.
_NUM = r"(?<![\w])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def normalize_digits(s: str) -> str:
    return s.translate(_DIGIT_TABLE)


def completion_text(c) -> str:
    """TRL passes either a string or a list of chat messages."""
    if isinstance(c, str):
        return c
    return "".join(m.get("content", "") for m in c)


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    text = normalize_digits(text)
    for pattern in (rf"####\s*\$?\s*({_NUM})", rf"\\boxed\{{\s*\$?({_NUM})"):
        m = re.findall(pattern, text)
        if m:
            return _to_float(m[-1])
    nums = re.findall(_NUM, text)
    return _to_float(nums[-1]) if nums else None


def correctness_reward(completions, answer, **kw) -> list[float]:
    out = []
    for c, gold in zip(completions, answer):
        pred, ref = extract_answer(completion_text(c)), extract_answer(str(gold))
        out.append(1.0 if pred is not None and ref is not None and abs(pred - ref) < 1e-6 else 0.0)
    return out


def format_reward(completions, **kw) -> list[float]:
    pat = re.compile(rf"####\s*\$?\s*{_NUM}\s*\.?\s*$")
    return [0.2 if pat.search(normalize_digits(completion_text(c))) else 0.0 for c in completions]
