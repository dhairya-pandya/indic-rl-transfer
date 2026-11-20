"""Train pool (AI4Bharat Indic GSM8K minus MGSM items) and MGSM eval sets."""

import random

from datasets import Dataset, load_dataset

from indic_rl.rewards import extract_answer

SYSTEM_PROMPT = (
    "Solve the problem step by step in the same language as the question. "
    "End with '#### <number>'."
)
TRAIN_SOURCE = "ai4bharat/indic-gsm8k"
MGSM_SOURCE = "juletxara/mgsm"
_TRAIN_COLS = {"en": "question", "bn": "question_Bengali_translation", "te": "question_Telugu_translation"}


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


def build_prompt(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]


def assert_no_overlap(train: Dataset, test_questions: list[str]) -> None:
    test = {_norm(q) for q in test_questions}
    hits = [q for q in train["question_en"] if _norm(q) in test]
    if hits:
        raise ValueError(f"{len(hits)} train questions overlap the test set, e.g. {hits[0][:80]!r}")


def _train_pool() -> Dataset:
    pool = load_dataset(TRAIN_SOURCE, split="test")
    mgsm_en = load_dataset(MGSM_SOURCE, "en", split="test")["question"]
    test = {_norm(q) for q in mgsm_en}
    before = len(pool)
    pool = pool.filter(lambda r: _norm(r["question"]) not in test)
    if before - len(pool) != len(mgsm_en):
        raise ValueError(f"expected to drop {len(mgsm_en)} MGSM items, dropped {before - len(pool)}; leakage filter unreliable")
    pool = pool.map(lambda r: {"question_en": r["question"], "gold": str(extract_answer(r["answer"].split("####")[-1]))})
    assert_no_overlap(pool, mgsm_en)
    return pool


def load_train(arm: str, n: int | None = None, seed: int = 0) -> Dataset:
    """Same problems for both arms; `indic` shows each problem in bn or te (seeded 50/50)."""
    if arm not in ("en", "indic"):
        raise ValueError(f"arm must be 'en' or 'indic', got {arm!r}")
    pool = _train_pool().shuffle(seed=seed)
    if n is not None:
        pool = pool.select(range(min(n, len(pool))))
    k = len(pool)
    if arm == "en":
        langs = ["en"] * k
    else:
        langs = ["bn"] * (k // 2) + ["te"] * (k - k // 2)
        random.Random(seed).shuffle(langs)
    rows = [
        {"prompt": build_prompt(r[_TRAIN_COLS[l]]), "answer": r["gold"], "lang": l, "question_en": r["question_en"]}
        for r, l in zip(pool, langs)
    ]
    return Dataset.from_list(rows)


def load_mgsm(lang: str, limit: int | None = None) -> Dataset:
    ds = load_dataset(MGSM_SOURCE, lang, split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return Dataset.from_list(
        [{"prompt": build_prompt(r["question"]), "answer": str(r["answer_number"]), "lang": lang, "question": r["question"]} for r in ds]
    )
