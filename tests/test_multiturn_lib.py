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


def test_wilson_ci_bounds():
    lo, hi = ml.wilson_ci(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    assert ml.wilson_ci(0, 0) == (0.0, 0.0)


def test_aggregate_groups_by_condition_and_bucket():
    rows = [
        {"condition": "off", "bucket": "coupled", "correct": True, "prompt_tokens": 100, "total_tokens": 300},
        {"condition": "off", "bucket": "coupled", "correct": False, "prompt_tokens": 120, "total_tokens": 320},
        {"condition": "on", "bucket": "coupled", "correct": True, "prompt_tokens": 200, "total_tokens": 450},
        {"condition": "on", "bucket": "coupled", "correct": True, "prompt_tokens": 220, "total_tokens": 470},
    ]
    agg = ml.aggregate(rows)
    assert agg["off"]["coupled"]["accuracy"] == 0.5
    assert agg["off"]["coupled"]["n"] == 2
    assert agg["on"]["coupled"]["accuracy"] == 1.0
    assert agg["on"]["coupled"]["mean_prompt_tokens"] == 210.0


def test_headline_deltas():
    agg = {
        "off": {"coupled": {"accuracy": 0.6, "mean_total_tokens": 300.0},
                "control": {"accuracy": 0.9, "mean_total_tokens": 280.0}},
        "on": {"coupled": {"accuracy": 0.7, "mean_total_tokens": 480.0},
               "control": {"accuracy": 0.9, "mean_total_tokens": 460.0}},
    }
    h = ml.headline(agg)
    assert round(h["coupled_accuracy_delta_on_minus_off"], 4) == 0.1
    assert round(h["control_accuracy_delta_on_minus_off"], 4) == 0.0
    assert round(h["coupled_token_overhead_on_minus_off"], 1) == 180.0
    assert h["verdict"] in {"helps_when_coupled", "no_effect", "hurts"}


def test_headline_incomplete_when_cell_missing():
    # on/coupled has zero rows (e.g. all wedged) -> must NOT report a real verdict
    agg = {
        "off": {"coupled": {"accuracy": 0.6, "mean_total_tokens": 300.0},
                "control": {"accuracy": 0.9, "mean_total_tokens": 280.0}},
        "on": {"control": {"accuracy": 0.9, "mean_total_tokens": 460.0}},
    }
    h = ml.headline(agg)
    assert h["verdict"] == "incomplete"
    assert h["coupled_accuracy_delta_on_minus_off"] is None
    assert h["coupled_token_overhead_on_minus_off"] is None


# ---------------------------------------------------------------------------
# Tests for job_key and load_checkpoint (Part A)
# ---------------------------------------------------------------------------

def test_job_key_format():
    key = ml.job_key("item42", "on", 3407)
    assert key == "item42|on|3407"
    key2 = ml.job_key("item42", "off", 3408)
    assert key2 == "item42|off|3408"


def test_load_checkpoint_missing_file(tmp_path):
    rows, keys = ml.load_checkpoint(tmp_path / "nonexistent.jsonl")
    assert rows == []
    assert keys == set()


def test_load_checkpoint_roundtrip(tmp_path):
    import json
    ckpt = tmp_path / "run.partial.jsonl"
    row1 = {"id": "item1", "condition": "on", "seed": 3407, "correct": True}
    row2 = {"id": "item2", "condition": "off", "seed": 3408, "correct": False}
    ckpt.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n")
    rows, keys = ml.load_checkpoint(ckpt)
    assert len(rows) == 2
    assert rows[0] == row1
    assert rows[1] == row2
    assert "item1|on|3407" in keys
    assert "item2|off|3408" in keys
    assert len(keys) == 2


def test_load_checkpoint_skips_corrupt_trailing_line(tmp_path):
    import json
    ckpt = tmp_path / "run.partial.jsonl"
    row1 = {"id": "itemA", "condition": "on", "seed": 3407, "correct": True}
    # Last line is a truncated/corrupt write
    ckpt.write_text(json.dumps(row1) + "\n" + "{bad json\n")
    rows, keys = ml.load_checkpoint(ckpt)
    assert len(rows) == 1
    assert rows[0] == row1
    assert "itemA|on|3407" in keys
