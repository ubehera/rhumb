import multiturn_lib as ml


def test_norm_num_strips_commas_and_trailing_zero():
    assert ml._norm_num("1,234") == "1234"
    assert ml._norm_num("42.0") == "42"
    assert ml._norm_num("-7") == "-7"


def test_extract_prefers_hash_marker():
    assert ml.extract_answer("blah\n#### 18\n") == "18"


def test_extract_boxed():
    assert ml.extract_answer(r"so \boxed{150}.") == "150"


def test_extract_answer_is_phrase():
    assert ml.extract_answer("The answer is 391.") == "391"


def test_extract_falls_back_to_last_number():
    assert ml.extract_answer("first 5 then finally 13") == "13"


def test_extract_strips_think_block():
    assert ml.extract_answer("<think>320 total</think>\nThe answer is 150.") == "150"


def test_extract_none_when_no_number():
    assert ml.extract_answer("no digits here") is None


def test_grade_exact_match():
    assert ml.grade("150", "150") is True
    assert ml.grade("1,50", "150") is False
    assert ml.grade("150.0", 150) is True
    assert ml.grade(None, "150") is False


def test_reconstruct_with_reasoning_embeds_think():
    out = ml.reconstruct_assistant_content("step 1\nstep 2", "150")
    assert out == "<think>\nstep 1\nstep 2\n</think>\n\n150"


def test_reconstruct_without_reasoning_is_plain():
    assert ml.reconstruct_assistant_content("", "150") == "150"
    assert ml.reconstruct_assistant_content(None, " 150 ") == "150"


def test_ctk_builder_forces_thinking_on():
    assert ml.build_chat_template_kwargs(False) == {"enable_thinking": True, "preserve_thinking": False}
    assert ml.build_chat_template_kwargs(True) == {"enable_thinking": True, "preserve_thinking": True}
