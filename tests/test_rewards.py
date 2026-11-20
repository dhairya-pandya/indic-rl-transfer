from indic_rl.rewards import extract_answer, correctness_reward, format_reward


def test_bengali_digits():
    assert extract_answer("উত্তর #### ১২৩") == 123


def test_telugu_digits():
    assert extract_answer("#### ౪౫") == 45


def test_commas_dollars():
    assert extract_answer("so $1,234.") == 1234


def test_boxed():
    assert extract_answer(r"\boxed{12.0}") == 12


def test_no_number():
    assert correctness_reward(["no idea"], ["5"]) == [0.0]


def test_correct():
    assert correctness_reward(["... #### 5"], ["5"]) == [1.0]


def test_format():
    assert format_reward(["x #### 7", "7"]) == [0.2, 0.0]


def test_chat_style_completion():
    assert correctness_reward([[{"role": "assistant", "content": "#### 5"}]], ["5"]) == [1.0]


def test_negative_and_hash_precedence():
    assert extract_answer("3 apples ... #### -4") == -4


def test_hyphen_range_not_negative():
    assert extract_answer("between 3-4 apples") == 4


def test_comma_list_not_merged():
    assert extract_answer("pick 2,3") == 3


def test_thousands_still_ok():
    assert extract_answer("#### 1,000.") == 1000
