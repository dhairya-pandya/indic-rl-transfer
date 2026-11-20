import pytest

from indic_rl.analysis import item_scores, paired_gain, transfer_ratio


def _rows(lang, correct_flags):
    return [
        {"lang": lang, "question": f"q{i}", "answer": "1", "completion": "#### 1" if ok else "#### 2"}
        for i, ok in enumerate(correct_flags)
    ]


def test_paired_gain_point():
    base = item_scores(_rows("bn", [0, 0, 1, 1]))
    new = item_scores(_rows("bn", [1, 0, 1, 1]))
    g = paired_gain(base, new, ["bn"])
    assert g["gain"] == 0.25 and g["n"] == 4 and g["lo"] <= 0.25 <= g["hi"]


def test_transfer_ratio_half():
    base = item_scores(_rows("te", [0] * 8))
    en = item_scores(_rows("te", [1, 1] + [0] * 6))
    ind = item_scores(_rows("te", [1, 1, 1, 1] + [0] * 4))
    assert transfer_ratio(base, en, ind, ["te"])["ratio"] == pytest.approx(0.5)


def test_no_shared_items_raises():
    with pytest.raises(ValueError):
        paired_gain(item_scores(_rows("bn", [1])), item_scores(_rows("te", [1])), ["bn"])
