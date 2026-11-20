import pytest
from datasets import Dataset

from indic_rl.data import SYSTEM_PROMPT, assert_no_overlap, build_prompt, load_mgsm, load_train
from indic_rl.langid import script_ratio
from indic_rl.rewards import extract_answer


def test_build_prompt():
    msgs = build_prompt("What is 2+2?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_overlap_raises():
    train = Dataset.from_dict({"question_en": ["A  cat has 2 legs?", "other"]})
    with pytest.raises(ValueError):
        assert_no_overlap(train, ["a cat has 2 legs?"])


@pytest.mark.net
def test_load_train_indic_split():
    ds = load_train("indic", 20)
    assert len(ds) == 20 and sorted(set(ds["lang"])) == ["bn", "te"]
    assert ds["lang"].count("bn") == 10
    for row in ds:
        assert script_ratio(row["prompt"][1]["content"], row["lang"]) > 0.5


@pytest.mark.net
def test_load_train_same_problems_both_arms():
    en, ind = load_train("en", 50), load_train("indic", 50)
    assert en["answer"] == ind["answer"]


@pytest.mark.net
def test_load_mgsm():
    ds = load_mgsm("bn", 5)
    assert len(ds) == 5 and all(extract_answer(a) is not None for a in ds["answer"])
