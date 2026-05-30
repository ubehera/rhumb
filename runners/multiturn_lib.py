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
