"""Pure helpers for the multiturn preserve_thinking probe. No I/O, no network —
everything here is unit-tested in tests/test_multiturn_lib.py."""
from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict

_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)


_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$")


def _norm_num(s: str) -> str:
    """Canonicalize a numeric string: drop thousands commas, collapse 42.0 -> 42."""
    s = str(s).strip()
    if _THOUSANDS_RE.match(s):
        s = s.replace(",", "")
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


def reconstruct_assistant_content(reasoning: str | None, content: str | None) -> str:
    """Rebuild the assistant turn so its <think> is present in history. The
    Qwen3.6 template then keeps it (preserve_thinking=true) or strips it (false)."""
    content = (content or "").strip()
    if reasoning and reasoning.strip():
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
    return content


def build_chat_template_kwargs(preserve: bool) -> dict:
    return {"enable_thinking": True, "preserve_thinking": bool(preserve)}


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

    cells = [acc("on", "coupled"), acc("off", "coupled"), acc("on", "control"), acc("off", "control")]
    if any(c is None for c in cells):
        # A condition x bucket has no data (e.g. all conversations wedged/failed).
        # Do NOT emit a real verdict — "no data" is not "measured 0%".
        return {
            "coupled_accuracy_delta_on_minus_off": None,
            "control_accuracy_delta_on_minus_off": None,
            "coupled_minus_control_delta": None,
            "coupled_token_overhead_on_minus_off": None,
            "verdict": "incomplete",
        }
    coupled_delta = acc("on", "coupled") - acc("off", "coupled")
    control_delta = acc("on", "control") - acc("off", "control")
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
        "coupled_token_overhead_on_minus_off": round((tok("on", "coupled") or 0.0) - (tok("off", "coupled") or 0.0), 1),
        "verdict": verdict,
    }
