from indic_rl.evaluate import score_rows


def test_score_rows():
    rows = [
        {"lang": "bn", "answer": "5", "completion": "আমি পাঁচ পেলাম #### 5"},
        {"lang": "bn", "answer": "3", "completion": "so the answer is #### 4"},
        {"lang": "en", "answer": "1", "completion": "#### 1"},
        {"lang": "en", "answer": "2", "completion": "#### 2"},
    ]
    s = score_rows(rows)
    assert s["bn"]["acc"] == 0.5 and s["bn"]["n"] == 2
    assert s["bn"]["en_reasoning_share"] == 0.5
    assert s["en"]["acc"] == 1.0
    assert s["bn"]["native_script_share"] == 0.5
