from indic_rl.langid import script_ratio, english_ratio, consistency_reward


def test_pure_bengali():
    assert script_ratio("আমি ভাত খাই", "bn") == 1.0


def test_mixed():
    assert 0.4 < script_ratio("আমি eat", "bn") < 0.7


def test_empty():
    assert script_ratio("", "te") == 0.0 and script_ratio("123", "te") == 0.0


def test_english_ratio():
    assert english_ratio("so the answer") == 1.0


def test_consistency_en_zero():
    assert consistency_reward(["hello"], ["en"]) == [0.0]


def test_consistency_bengali():
    assert consistency_reward(["আমি"], ["bn"]) == [0.3]
