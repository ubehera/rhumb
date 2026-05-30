# preserve_thinking multi-turn probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rhumb `multiturn_chat` runner that measures whether Qwen3.6's `preserve_thinking=true` improves answer accuracy (and at what token cost) versus the default strip, on the SGLang TP=2 daily driver.

**Architecture:** A new standalone runner (`runners/quality_multiturn.py`) drives genuine multi-turn conversations against the OpenAI-compatible SGLang endpoint via `httpx`. After each assistant turn it re-embeds the model's real `<think>` into history, then sends the *identical* messages under two conditions, flipping only `chat_template_kwargs.preserve_thinking`. Pure helpers (extraction, grading, reconstruction, aggregation) live in `runners/multiturn_lib.py` and are unit-tested. Grading is deterministic exact-match on a single-integer final answer (no judge).

**Tech Stack:** Python 3.12, `httpx` (already a dep), `pyyaml`, `pytest` (added as a dev group). Reuses rhumb's `models.yaml` / `endpoints.yaml` / `run.sh` dispatch and results layout.

---

## Design notes (read before implementing)

- **Thinking is forced ON.** `preserve_thinking` is meaningless with thinking off. The runner always sends `enable_thinking: true`; the only A/B variable is `preserve_thinking`.
- **Concurrency cap.** The 35B-A3B TP=2 cross-node stack wedges at 5+ concurrent long-output requests (`feedback_sglang_5concurrent_wedge_unsustainable`). The suite sets `num_concurrent: 2`. Do not raise it for this model.
- **No post-completion retries.** Per `feedback_httpx_retry_narrow_set`, `RemoteProtocolError`/`ReadError` can fire *after* the model finished generating; retrying duplicates minutes of expensive work. The driver does NOT retry generation requests. A connection-refused before any bytes is a hard fail (surfaced, not retried).
- **Honest effect-size expectation.** The *user* turns (problem statements) are always retained regardless of the flag — `preserve_thinking` only strips/keeps the *assistant's own* `<think>`. So the coupled effect is "reuse the prior worked steps vs re-derive them," not "the info is gone." A capable model may re-derive reliably, yielding a small or zero delta — which is itself a valid, reportable result. Items are designed so reuse *could* help; the measurement decides.
- **Grading contract.** Every item's FINAL turn must resolve to a single integer, so `extract_answer` + exact-match is unambiguous. Intermediate turns may carry an `expected` for diagnostics (e.g., "did turn 1 get the intermediate right").
- Spec this implements: `docs/specs/2026-05-29-preserve-thinking-probe-design.md`.

## File structure

- Create: `runners/multiturn_lib.py` — pure, unit-tested helpers (no I/O, no `main`).
- Create: `runners/quality_multiturn.py` — the runner `main()`: arg/config resolution + HTTP multi-turn loop + output JSON.
- Create: `tasks/preserve_thinking/items.jsonl` — 12 coupled + 12 control hand-authored items.
- Create: `suites/preserve_thinking.yaml` — suite config (runner, decoding, conditions, samples, concurrency).
- Create: `tests/conftest.py` — puts `runners/` on `sys.path` for imports.
- Create: `tests/test_multiturn_lib.py` — unit tests for the helpers.
- Create: `tests/test_dataset.py` — validates the dataset schema + grading contract.
- Modify: `run.sh:168-173` — add the `multiturn_chat` dispatch case.
- Modify: `pyproject.toml:15-17` — add a `dev` dependency group with `pytest`.

---

### Task 1: Test scaffolding (pytest dev group + conftest)

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add a dev dependency group to `pyproject.toml`**

Append after the `[tool.uv]` block (after line 17):

```toml
[dependency-groups]
dev = ["pytest>=8"]
```

- [ ] **Step 2: Create `tests/conftest.py`** so tests can `import multiturn_lib`

```python
import pathlib
import sys

# runners/ is not a package; put it on sys.path so tests can import the lib.
RUNNERS = pathlib.Path(__file__).resolve().parent.parent / "runners"
sys.path.insert(0, str(RUNNERS))
```

- [ ] **Step 3: Sync the dev group**

Run: `cd ~/projects/rhumb && uv sync --group dev`
Expected: resolves and installs `pytest` into `.venv` (no errors).

- [ ] **Step 4: Commit**

```bash
cd ~/projects/rhumb
git add pyproject.toml tests/conftest.py
git commit -m "test: add pytest dev group + conftest for runner-lib imports"
```

---

### Task 2: Answer extraction + exact-match grading (TDD)

**Files:**
- Create: `runners/multiturn_lib.py`
- Test: `tests/test_multiturn_lib.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_multiturn_lib.py`:

```python
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
    # a leaked <think> with a different number must not win over post-think content
    assert ml.extract_answer("<think>320 total</think>\nThe answer is 150.") == "150"


def test_extract_none_when_no_number():
    assert ml.extract_answer("no digits here") is None


def test_grade_exact_match():
    assert ml.grade("150", "150") is True
    assert ml.grade("1,50", "150") is False  # comma inside is not a thousands sep here
    assert ml.grade("150.0", 150) is True
    assert ml.grade(None, "150") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/projects/rhumb && uv run --group dev pytest tests/test_multiturn_lib.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'multiturn_lib'`.

- [ ] **Step 3: Implement extraction + grading in `runners/multiturn_lib.py`**

```python
"""Pure helpers for the multiturn preserve_thinking probe. No I/O, no network —
everything here is unit-tested in tests/test_multiturn_lib.py."""
from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict

_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)


def _norm_num(s: str) -> str:
    """Canonicalize a numeric string: drop thousands commas, collapse 42.0 -> 42."""
    s = str(s).strip().replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_answer(text: str | None) -> str | None:
    """Pull the final numeric answer out of model output.
    Priority: post-</think> content -> '#### N' -> \\boxed{N} -> 'answer is N' -> last number."""
    if not text:
        return None
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1]
    m = re.search(r"####\s*(" + _NUM + ")", t)
    if m:
        return _norm_num(m.group(1))
    m = re.search(r"\\boxed\{([^}]*)\}", t)
    if m:
        inner = _NUM_RE.search(m.group(1))
        if inner:
            return _norm_num(inner.group(0))
    m = re.search(r"answer\s*(?:is|:)?\s*\*{0,2}(" + _NUM + ")", t, re.I)
    if m:
        return _norm_num(m.group(1))
    nums = _NUM_RE.findall(t)
    return _norm_num(nums[-1]) if nums else None


def grade(pred: str | None, expected) -> bool:
    if pred is None:
        return False
    return _norm_num(pred) == _norm_num(expected)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd ~/projects/rhumb && uv run --group dev pytest tests/test_multiturn_lib.py -q`
Expected: PASS (all extraction/grading tests green).

- [ ] **Step 5: Commit**

```bash
cd ~/projects/rhumb
git add runners/multiturn_lib.py tests/test_multiturn_lib.py
git commit -m "feat: answer extraction + exact-match grading for multiturn probe"
```

---

### Task 3: History reconstruction + chat_template_kwargs builder (TDD)

**Files:**
- Modify: `runners/multiturn_lib.py`
- Test: `tests/test_multiturn_lib.py`

- [ ] **Step 1: Add failing tests** (append to `tests/test_multiturn_lib.py`)

```python
def test_reconstruct_with_reasoning_embeds_think():
    out = ml.reconstruct_assistant_content("step 1\nstep 2", "150")
    assert out == "<think>\nstep 1\nstep 2\n</think>\n\n150"


def test_reconstruct_without_reasoning_is_plain():
    assert ml.reconstruct_assistant_content("", "150") == "150"
    assert ml.reconstruct_assistant_content(None, " 150 ") == "150"


def test_ctk_builder_forces_thinking_on():
    assert ml.build_chat_template_kwargs(False) == {"enable_thinking": True, "preserve_thinking": False}
    assert ml.build_chat_template_kwargs(True) == {"enable_thinking": True, "preserve_thinking": True}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/projects/rhumb && uv run --group dev pytest tests/test_multiturn_lib.py -q`
Expected: FAIL — `AttributeError: module 'multiturn_lib' has no attribute 'reconstruct_assistant_content'`.

- [ ] **Step 3: Implement in `runners/multiturn_lib.py`** (append)

```python
def reconstruct_assistant_content(reasoning: str | None, content: str | None) -> str:
    """Rebuild the assistant turn so its <think> is present in history. The
    Qwen3.6 template then keeps it (preserve_thinking=true) or strips it (false)."""
    content = (content or "").strip()
    if reasoning and reasoning.strip():
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
    return content


def build_chat_template_kwargs(preserve: bool) -> dict:
    return {"enable_thinking": True, "preserve_thinking": bool(preserve)}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd ~/projects/rhumb && uv run --group dev pytest tests/test_multiturn_lib.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/rhumb
git add runners/multiturn_lib.py tests/test_multiturn_lib.py
git commit -m "feat: <think> history reconstruction + chat_template_kwargs builder"
```

---

### Task 4: Aggregation — accuracy, Wilson CI, token means (TDD)

**Files:**
- Modify: `runners/multiturn_lib.py`
- Test: `tests/test_multiturn_lib.py`

- [ ] **Step 1: Add failing tests** (append to `tests/test_multiturn_lib.py`)

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/projects/rhumb && uv run --group dev pytest tests/test_multiturn_lib.py -q`
Expected: FAIL — `AttributeError: module 'multiturn_lib' has no attribute 'wilson_ci'`.

- [ ] **Step 3: Implement in `runners/multiturn_lib.py`** (append)

```python
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(rows: list[dict]) -> dict:
    """rows: per-(item,condition,sample) dicts with keys
    condition, bucket, correct(bool), prompt_tokens, total_tokens."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["bucket"])].append(r)
    out: dict = {}
    for (cond, bucket), rs in groups.items():
        n = len(rs)
        k = sum(1 for r in rs if r["correct"])
        lo, hi = wilson_ci(k, n)
        out.setdefault(cond, {})[bucket] = {
            "accuracy": round(k / n, 4) if n else None,
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "n": n,
            "n_correct": k,
            "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rs), 1) if n else None,
            "mean_total_tokens": round(statistics.mean(r["total_tokens"] for r in rs), 1) if n else None,
        }
    return out


def headline(agg: dict) -> dict:
    def acc(cond, bucket):
        return agg.get(cond, {}).get(bucket, {}).get("accuracy")

    def tok(cond, bucket):
        return agg.get(cond, {}).get(bucket, {}).get("mean_total_tokens")

    coupled_delta = (acc("on", "coupled") or 0.0) - (acc("off", "coupled") or 0.0)
    control_delta = (acc("on", "control") or 0.0) - (acc("off", "control") or 0.0)
    net = coupled_delta - control_delta
    if net > 0.02:
        verdict = "helps_when_coupled"
    elif coupled_delta < -0.02:
        verdict = "hurts"
    else:
        verdict = "no_effect"
    return {
        "coupled_accuracy_delta_on_minus_off": round(coupled_delta, 4),
        "control_accuracy_delta_on_minus_off": round(control_delta, 4),
        "coupled_minus_control_delta": round(net, 4),
        "coupled_token_overhead_on_minus_off": (
            round((tok("on", "coupled") or 0.0) - (tok("off", "coupled") or 0.0), 1)
        ),
        "verdict": verdict,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd ~/projects/rhumb && uv run --group dev pytest tests/test_multiturn_lib.py -q`
Expected: PASS (all helper tests green).

- [ ] **Step 5: Commit**

```bash
cd ~/projects/rhumb
git add runners/multiturn_lib.py tests/test_multiturn_lib.py
git commit -m "feat: aggregation, Wilson CI, headline deltas for multiturn probe"
```

---

### Task 5: Author the dataset (12 coupled + 12 control) + validate (TDD)

**Files:**
- Create: `tasks/preserve_thinking/items.jsonl`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing dataset-contract test**

Create `tests/test_dataset.py`:

```python
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
        # final answer must be a clean integer so exact-match is unambiguous
        assert ml._norm_num(final["expected"]) == str(int(float(final["expected"])))


def test_final_turn_requests_answer_only():
    # guards against method/leakage: final prompt must constrain to a bare answer
    for i in load_items():
        last = i["turns"][-1]["content"].lower()
        assert "only" in last or "just" in last, f"{i['id']} final turn must request answer-only"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/projects/rhumb && uv run --group dev pytest tests/test_dataset.py -q`
Expected: FAIL — `FileNotFoundError` / items file missing.

- [ ] **Step 3: Create `tasks/preserve_thinking/items.jsonl`**

Authoring rules (every item must satisfy these — `tests/test_dataset.py` enforces the mechanical ones):
- `role` is always `"user"` for authored turns (assistant turns are generated).
- Turn 1 asks for an **answer-only** terse reply, so the working/intermediates land in `<think>`, not the visible content.
- **Coupled:** the final turn reuses an *intermediate the model worked out in turn 1* (e.g. "using the total you computed", "using the same method"), and the final answer is NOT equal to turn 1's visible answer.
- **Control:** the final turn is an independent problem needing nothing from turn 1.
- Final turn resolves to a single integer; include an `expected` on the final turn (and optionally on turn 1 as a diagnostic).
- 12 coupled + 12 control.

Start with these 8 worked exemplars (4 coupled, 4 control), then author 16 more (8+8) to the same pattern. Each line is one JSON object:

```json
{"id": "coupled-01", "bucket": "coupled", "turns": [{"content": "A train goes 120 km in 1.5 h, then 200 km in 2 h. Reply with ONLY the average speed for the whole trip in km/h, rounded to the nearest integer.", "expected": "91"}, {"content": "Using the total distance you computed in the previous step, reply with ONLY the number of whole minutes it would take to cover that same distance at 80 km/h (round down).", "expected": "240"}]}
{"id": "coupled-02", "bucket": "coupled", "turns": [{"content": "Compute the discriminant of 3x^2 - 11x + 6. Reply with ONLY the integer.", "expected": "73"}, {"content": "Using the two intermediate products b^2 and 4ac you computed, reply with ONLY their sum (b^2 + 4ac).", "expected": "193"}]}
{"id": "coupled-03", "bucket": "coupled", "turns": [{"content": "A shop sells 37 widgets at $14 each and 19 gadgets at $23 each. Reply with ONLY the total revenue in dollars.", "expected": "955"}, {"content": "Using the widget subtotal you computed (not the grand total), reply with ONLY that subtotal divided by 2, rounded down to the nearest integer.", "expected": "259"}]}
{"id": "coupled-04", "bucket": "coupled", "turns": [{"content": "Find the least common multiple of 18 and 24. Reply with ONLY the integer.", "expected": "72"}, {"content": "Using the greatest common divisor you found while computing that LCM, reply with ONLY 100 minus that gcd.", "expected": "94"}]}
{"id": "control-01", "bucket": "control", "turns": [{"content": "Compute 17 * 23. Reply with ONLY the integer.", "expected": "391"}, {"content": "Separately: reply with ONLY the 6th prime number.", "expected": "13"}]}
{"id": "control-02", "bucket": "control", "turns": [{"content": "Compute the sum of the first 9 positive integers. Reply with ONLY the integer.", "expected": "45"}, {"content": "Unrelated: reply with ONLY the number of days in the months April through July inclusive (non-leap year).", "expected": "122"}]}
{"id": "control-03", "bucket": "control", "turns": [{"content": "Compute 2^10. Reply with ONLY the integer.", "expected": "1024"}, {"content": "Separately: a rectangle is 13 by 9. Reply with ONLY its perimeter.", "expected": "44"}]}
{"id": "control-04", "bucket": "control", "turns": [{"content": "Compute 144 / 12 + 7. Reply with ONLY the integer.", "expected": "19"}, {"content": "Unrelated: reply with ONLY the number of edges on a cube.", "expected": "12"}]}
```

Author `coupled-05..12` and `control-05..12` (8 more each) following the identical rules. Vary domains (arithmetic word problems, algebra intermediates, gcd/lcm, counting/combinatorics) and difficulty so turn-1 visible accuracy is unlikely to be at ceiling. Verify each coupled item's turn-2 truly references a turn-1 *intermediate* (not turn-1's printed answer).

- [ ] **Step 4: Run to verify the dataset contract passes**

Run: `cd ~/projects/rhumb && uv run --group dev pytest tests/test_dataset.py -q`
Expected: PASS (counts = 12/12, schema + answer-only + integer-final all green).

- [ ] **Step 5: Manually re-verify coupling (cross-check discipline)**

Read every `coupled-*` item and confirm turn 2 needs an intermediate from turn 1's reasoning, not its visible answer. Fix any that leak. This is a human/judgment check, not automated.

- [ ] **Step 6: Commit**

```bash
cd ~/projects/rhumb
git add tasks/preserve_thinking/items.jsonl tests/test_dataset.py
git commit -m "feat: coupled/control multiturn dataset (12+12) + contract tests"
```

---

### Task 6: The runner `quality_multiturn.py` (multi-turn HTTP loop)

**Files:**
- Create: `runners/quality_multiturn.py`

This is integration code (network). There is no unit test; it is validated by the smoke run in Task 8. Keep config/endpoint resolution identical to `quality_chat_lm_eval.py` so `run.sh` passthrough works unchanged.

- [ ] **Step 1: Create `runners/quality_multiturn.py`**

```python
"""Quality runner (multi-turn variant): measures Qwen3.6 preserve_thinking ON vs
OFF on genuine multi-turn conversations against an OpenAI-compatible endpoint.

Unlike quality_chat_lm_eval.py (single-target lm-eval tasks), this runner drives
each conversation turn-by-turn, re-embedding the model's <think> into history so
the server-side Qwen3.6 template can keep it (preserve_thinking=true) or strip it
(false). The ONLY variable between conditions is that flag.

Usage (via run.sh, suite routes here when runner: multiturn_chat):
    ./run.sh --model qwen3.6-35b-a3b-nvfp4 --suite preserve_thinking \\
        --only quality --endpoint sglang-tp2 --detach
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import json
import os
import pathlib
import sys
import time

import httpx
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "runners"))
import multiturn_lib as ml  # noqa: E402


def load_yaml(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_conversation(client, url, headers, model_id, item, preserve, decoding, seed, timeout):
    """Drive one conversation under one condition+seed. Returns the final-turn row
    plus per-turn detail. No retries (post-completion errors must not duplicate work)."""
    history: list[dict] = []
    turns_out = []
    final_correct = False
    final_prompt_tokens = 0
    final_total_tokens = 0
    for ti, turn in enumerate(item["turns"]):
        history.append({"role": "user", "content": turn["content"]})
        body = {
            "model": model_id,
            "messages": history,
            "temperature": decoding["temperature"],
            "top_p": decoding["top_p"],
            "top_k": decoding["top_k"],
            "max_tokens": decoding["max_tokens"],
            "seed": seed,
            "chat_template_kwargs": ml.build_chat_template_kwargs(preserve),
        }
        r = client.post(url, headers=headers, json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        reasoning = msg.get("reasoning_content", "") or ""
        content = msg.get("content", "") or ""
        usage = data.get("usage", {}) or {}
        is_final = ti == len(item["turns"]) - 1
        pred = ml.extract_answer(content)
        expected = turn.get("expected")
        correct = ml.grade(pred, expected) if expected is not None else None
        turns_out.append({
            "turn": ti,
            "reasoning_chars": len(reasoning),
            "visible_chars": len(content.strip()),
            "pred": pred,
            "expected": expected,
            "correct": correct,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })
        if is_final:
            final_correct = bool(correct)
            final_prompt_tokens = usage.get("prompt_tokens", 0)
            final_total_tokens = usage.get("total_tokens",
                                           usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
        history.append({"role": "assistant",
                        "content": ml.reconstruct_assistant_content(reasoning, content)})
    return {
        "id": item["id"],
        "bucket": item["bucket"],
        "condition": "on" if preserve else "off",
        "seed": seed,
        "correct": final_correct,
        "prompt_tokens": final_prompt_tokens,
        "total_tokens": final_total_tokens,
        "turns": turns_out,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--suite", default="preserve_thinking")
    p.add_argument("--out", required=True)
    p.add_argument("--thinking", choices=["on", "off", "auto"], default="auto",
                   help="Ignored for this probe — thinking is always ON (preserve is the only variable). A warning prints if 'off' is passed.")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--base-url", dest="base_url", default=None)
    args = p.parse_args()

    if args.thinking == "off":
        print("[multiturn] WARNING: --thinking off ignored; preserve_thinking probe forces thinking ON", file=sys.stderr)

    models = load_yaml(REPO / "models.yaml")["models"]
    if args.model not in models:
        sys.exit(f"unknown model '{args.model}' (known: {list(models)})")
    model_cfg = models[args.model]

    suite_yaml = load_yaml(REPO / "suites" / f"{args.suite}.yaml")
    suite = suite_yaml["quality"]
    decoding = suite["decoding"]
    samples = int(suite.get("samples", 5))
    seed_base = int(suite.get("seed_base", 3407))
    num_concurrent = int(suite.get("num_concurrent", 2))
    timeout = float(suite.get("timeout", 600))
    conditions = suite.get("conditions", ["off", "on"])
    limit = suite.get("limit")

    api_key = os.environ.get(model_cfg.get("api_key_env", "VLLM_API_KEY"), "")
    endpoint_root = (args.base_url or model_cfg["endpoint"]).rstrip("/")
    url = endpoint_root + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    model_id = model_cfg["model_id"]
    print(f"[multiturn] model={model_id} endpoint={args.endpoint or 'model-default'} base={url}", flush=True)
    print(f"[multiturn] conditions={conditions} samples={samples} concurrency={num_concurrent} decoding={decoding}", flush=True)

    items = []
    with open(REPO / suite["dataset"]) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if limit:
        items = items[:limit]
    print(f"[multiturn] {len(items)} items x {len(conditions)} conditions x {samples} samples = "
          f"{len(items) * len(conditions) * samples} conversations", flush=True)

    jobs = []
    for item in items:
        for cond in conditions:
            preserve = cond == "on"
            for s in range(samples):
                seed = seed_base + s  # shared across conditions for the same (item,s) => paired
                jobs.append((item, preserve, seed))

    rows = []
    started_at = datetime.datetime.now(datetime.timezone.utc)
    t0 = time.perf_counter()
    with httpx.Client() as client:
        with cf.ThreadPoolExecutor(max_workers=num_concurrent) as ex:
            futs = {ex.submit(run_conversation, client, url, headers, model_id,
                              item, preserve, decoding, seed, timeout): (item["id"], preserve, seed)
                    for (item, preserve, seed) in jobs}
            done = 0
            for fut in cf.as_completed(futs):
                ident = futs[fut]
                try:
                    rows.append(fut.result())
                except Exception as e:  # noqa: BLE001 — surface, do not retry (would duplicate generation)
                    print(f"[multiturn] FAILED {ident}: {type(e).__name__}: {e}", file=sys.stderr)
                done += 1
                if done % 20 == 0:
                    print(f"[multiturn] {done}/{len(jobs)} conversations done", flush=True)
    wall = time.perf_counter() - t0
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    agg = ml.aggregate(rows)
    head = ml.headline(agg)

    # Leakage diagnostic: coupled turn-1 visible answers should be terse.
    leak = [r["id"] for r in rows
            if r["bucket"] == "coupled" and r["turns"] and r["turns"][0]["visible_chars"] > 40]
    out = {
        "runner": "multiturn_chat",
        "model": args.model,
        "model_id": model_id,
        "endpoint": endpoint_root,
        "endpoint_name": args.endpoint or "model-default",
        "suite": args.suite,
        "thinking": "on",
        "decoding": decoding,
        "samples": samples,
        "seed_base": seed_base,
        "conditions": conditions,
        "summary": agg,
        "headline": head,
        "leakage_suspect_ids": sorted(set(leak)),
        "n_rows": len(rows),
        "n_failed": len(jobs) - len(rows),
        "per_row": rows,
        "wall_seconds": round(wall, 2),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }
    out_path = pathlib.Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[multiturn] wrote {out_path}", flush=True)
    print(f"[multiturn] verdict={head['verdict']} "
          f"coupled_delta={head['coupled_accuracy_delta_on_minus_off']} "
          f"control_delta={head['control_accuracy_delta_on_minus_off']} "
          f"token_overhead={head['coupled_token_overhead_on_minus_off']}", flush=True)
    if leak:
        print(f"[multiturn] WARNING: {len(set(leak))} coupled items show long turn-1 answers (possible leakage): {sorted(set(leak))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Byte-compile to catch syntax errors**

Run: `cd ~/projects/rhumb && uv run python -m py_compile runners/quality_multiturn.py`
Expected: no output (exit 0).

- [ ] **Step 3: Commit**

```bash
cd ~/projects/rhumb
git add runners/quality_multiturn.py
git commit -m "feat: multiturn_chat runner — preserve_thinking A/B over live conversations"
```

---

### Task 7: Wire dispatch (`run.sh`) + suite YAML

**Files:**
- Modify: `run.sh` (the runner-dispatch `case`, lines 168-173)
- Create: `suites/preserve_thinking.yaml`

- [ ] **Step 1: Add the `multiturn_chat` dispatch case in `run.sh`**

Find (lines 168-173):

```bash
case "$RUNNER" in
  lm_eval)      QUALITY_OUT="$OUT_DIR/quality.json";       QUALITY_SCRIPT="quality_lm_eval.py";;
  chat_lm_eval) QUALITY_OUT="$OUT_DIR/quality_chat.json";  QUALITY_SCRIPT="quality_chat_lm_eval.py";;
  vlmeval)      QUALITY_OUT="$OUT_DIR/quality_vlm.json";   QUALITY_SCRIPT="quality_vlmeval.py";;
  *) echo "unknown runner '$RUNNER' in $SUITE_YAML (expected lm_eval, chat_lm_eval, or vlmeval)" >&2; exit 1;;
esac
```

Replace with (adds the `multiturn_chat` line and updates the error text):

```bash
case "$RUNNER" in
  lm_eval)        QUALITY_OUT="$OUT_DIR/quality.json";            QUALITY_SCRIPT="quality_lm_eval.py";;
  chat_lm_eval)   QUALITY_OUT="$OUT_DIR/quality_chat.json";       QUALITY_SCRIPT="quality_chat_lm_eval.py";;
  vlmeval)        QUALITY_OUT="$OUT_DIR/quality_vlm.json";        QUALITY_SCRIPT="quality_vlmeval.py";;
  multiturn_chat) QUALITY_OUT="$OUT_DIR/quality_multiturn.json";  QUALITY_SCRIPT="quality_multiturn.py";;
  *) echo "unknown runner '$RUNNER' in $SUITE_YAML (expected lm_eval, chat_lm_eval, vlmeval, or multiturn_chat)" >&2; exit 1;;
esac
```

- [ ] **Step 2: Create `suites/preserve_thinking.yaml`**

```yaml
# preserve_thinking probe — measures Qwen3.6 preserve_thinking ON vs OFF on
# genuine multi-turn conversations. Routes to runners/quality_multiturn.py.
# See docs/specs/2026-05-29-preserve-thinking-probe-design.md.
runner: multiturn_chat

# Thinking is always ON for this probe (preserve is meaningless otherwise);
# the runner enforces it. force_thinking is informational here.
force_thinking: true

quality:
  dataset: tasks/preserve_thinking/items.jsonl
  conditions: [off, on]        # preserve_thinking false / true
  samples: 5                   # samples per item per condition; paired seeds across conditions
  seed_base: 3407
  # 35B-A3B TP=2 cross-node wedges at 5+ concurrent long-output requests
  # (feedback_sglang_5concurrent_wedge_unsustainable). Keep this at 2.
  num_concurrent: 2
  timeout: 600                 # thinking-on multi-turn can take minutes/turn
  decoding:
    temperature: 0.6           # Qwen thinking-mode recommendation
    top_p: 0.95
    top_k: 20
    max_tokens: 4096
  # limit: 2                   # uncomment for the smoke run in Task 8

# Speed bench unused for this probe — always run with `--only quality`.
speed:
  num_prompts: 1
  output_tokens: 1
  concurrency_levels: [1]
```

- [ ] **Step 3: Verify run.sh still parses and resolves the runner**

Run: `cd ~/projects/rhumb && bash -n run.sh && python3 -c "import yaml; print(yaml.safe_load(open('suites/preserve_thinking.yaml'))['runner'])"`
Expected: `bash -n` prints nothing (valid); python prints `multiturn_chat`.

- [ ] **Step 4: Commit**

```bash
cd ~/projects/rhumb
git add run.sh suites/preserve_thinking.yaml
git commit -m "feat: route preserve_thinking suite to the multiturn_chat runner"
```

---

### Task 8: Endpoint smoke test, then full detached run + read results

**Files:** none created; this runs the probe.

- [ ] **Step 1: Confirm the live endpoint serves the expected model**

Run: `source ~/projects/sglang-server/sglang.env 2>/dev/null; curl -fsS http://umanks-mac-mini:30000/v1/models -H "Authorization: Bearer ${SGLANG_API_KEY:-$VLLM_API_KEY}" | python3 -m json.tool`
Expected: JSON listing a model id matching `RedHatAI/Qwen3.6-35B-A3B-NVFP4`. If the served id differs (e.g. the 27B-FP8 or the FP8 35B is currently up), either restart the 35B-A3B-NVFP4 SGLang service or switch `--model` to the model key that matches what is actually served. Do NOT run the probe against the wrong model.

- [ ] **Step 2: Smoke run (2 items) end-to-end**

Temporarily uncomment `limit: 2` in `suites/preserve_thinking.yaml`, then:

Run: `cd ~/projects/rhumb && ./run.sh --model qwen3.6-35b-a3b-nvfp4 --suite preserve_thinking --only quality --endpoint sglang-tp2`
Expected: completes in a few minutes; writes `results/qwen3.6-35b-a3b-nvfp4/<date>/sglang-tp2/quality_multiturn.json`. Open it and confirm: `n_failed: 0`, `per_row` shows non-empty `reasoning_chars` on turn 0 (thinking actually happened), and `summary` has both `off` and `on` with both buckets. Re-comment `limit: 2`.

- [ ] **Step 3: Sanity-check the mechanism is live**

In the smoke `quality_multiturn.json`, find an `on` row and an `off` row for the same coupled `id`+`seed`. Confirm the `on` row's final-turn `prompt_tokens` is **higher** than the `off` row's (preserve_thinking is retaining the prior `<think>`, so the turn-2 prompt is longer). If they're equal, the flag isn't taking effect — stop and debug (check `tokenizer_backend: none`, check the served template actually has `preserve_thinking`).

- [ ] **Step 4: Full run, detached**

Per `feedback_detach_long_jobs`, the full run (24 items x 2 x 5 = 240 conversations, thinking-on) will exceed ~10 min — detach it:

Run: `cd ~/projects/rhumb && ./run.sh --model qwen3.6-35b-a3b-nvfp4 --suite preserve_thinking --only quality --endpoint sglang-tp2 --detach`
Then monitor: `tail -f results/qwen3.6-35b-a3b-nvfp4/<date>/sglang-tp2/run.log`
Expected: progress lines every 20 conversations; ends with `[multiturn] wrote ...` and a `verdict=...` line.

- [ ] **Step 5: Read the headline + verify validity**

Open the final `quality_multiturn.json`:
- `headline`: `coupled_accuracy_delta_on_minus_off`, `control_accuracy_delta_on_minus_off`, `coupled_minus_control_delta`, `coupled_token_overhead_on_minus_off`, `verdict`.
- Check `leakage_suspect_ids` is empty (if not, those coupled items leaked the method into the visible turn-1 answer — fix the items and re-run).
- Spot-check 3-4 `per_row.turns` transcripts by eye to confirm grading is sane (cross-check-benchmarks discipline).

- [ ] **Step 6: Commit any dataset fixes made during validation**

```bash
cd ~/projects/rhumb
git add tasks/preserve_thinking/items.jsonl
git commit -m "fix: tighten coupled items flagged by leakage diagnostic" || echo "no dataset changes"
```

---

## Self-review

**Spec coverage:** live multi-turn loop (Task 6) ✓; single-variable flag flip (Task 6 `build_chat_template_kwargs`, enforced thinking-on) ✓; coupled vs control buckets + terse turn-1 (Task 5) ✓; exact-match no-judge grading (Task 2) ✓; temp 0.6 + N=5 paired seeds (Task 6/7) ✓; accuracy delta + token overhead by bucket (Task 4) ✓; "preserve hurts" outcome path (`headline` verdict) ✓; risk mitigations — leakage diagnostic (`leakage_suspect_ids`), extraction spot-check (Task 8 Step 5), ceiling guidance (Task 5 authoring rule), paired-seed noise control ✓.

**Placeholder scan:** the only deferred content is dataset items `coupled-05..12` / `control-05..12` — not a code placeholder; it is a content-authoring task bounded by explicit rules + 8 worked exemplars + a contract test (`tests/test_dataset.py`) that fails until 12+12 valid items exist.

**Type consistency:** `aggregate` consumes rows with keys `condition/bucket/correct/prompt_tokens/total_tokens`; `run_conversation` returns exactly those keys ✓. `headline` reads `accuracy`/`mean_total_tokens` which `aggregate` produces ✓. `build_chat_template_kwargs`, `extract_answer`, `grade`, `reconstruct_assistant_content` names match between `multiturn_lib.py`, the runner, and the tests ✓.

## Known limitation carried from the spec

Because user turns are always retained, `preserve_thinking` only saves the model from *re-deriving* its own prior work — it does not hide information. A genuine null result (verdict `no_effect`) is therefore a likely and valid outcome, and the recommendation stands either way: keep preserve OFF by default unless the coupled delta clearly clears the control delta and justifies the token overhead.
