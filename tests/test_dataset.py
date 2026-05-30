import json
import pathlib

import multiturn_lib as ml

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "tasks" / "preserve_thinking" / "items.jsonl"


def load_items():
    with open(DATA) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_file_exists_and_counts():
    items = load_items()
    buckets = [i["bucket"] for i in items]
    assert buckets.count("coupled") == 12
    assert buckets.count("control") == 12


def test_schema_and_grading_contract():
    seen_ids = set()
    for i in load_items():
        assert i["id"] not in seen_ids, f"duplicate id {i['id']}"
        seen_ids.add(i["id"])
        assert i["bucket"] in {"coupled", "control"}
        assert len(i["turns"]) >= 2
        for t in i["turns"]:
            assert t["role"] == "user"
            assert t["content"].strip()
        final = i["turns"][-1]
        assert "expected" in final, f"{i['id']} final turn needs expected"
        assert ml._norm_num(final["expected"]) == str(int(float(final["expected"])))


def test_final_turn_requests_answer_only():
    for i in load_items():
        last = i["turns"][-1]["content"].lower()
        assert "only" in last or "just" in last, f"{i['id']} final turn must request answer-only"
