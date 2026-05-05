"""Pairwise diff between two rhumb result directories.

Each input is a results/<model>/<date>/ directory containing quality.json and/or
speed.json. The output is a comparison table showing per-metric deltas.

Usage:
    python diff.py results/qwen-32b-awq/2026-05-04 results/qwen-32b-bf16/2026-05-04
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from rich.console import Console
from rich.table import Table


def load_dir(p: pathlib.Path) -> dict:
    out: dict = {"path": str(p)}
    for name in ("quality.json", "speed.json", "meta.json"):
        f = p / name
        if f.exists():
            out[name.removesuffix(".json")] = json.loads(f.read_text())
    return out


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def fmt_delta(a, b) -> str:
    if a is None or b is None:
        return ""
    try:
        d = float(b) - float(a)
    except (ValueError, TypeError):
        return ""
    sign = "+" if d > 0 else ""
    return f"({sign}{d:.4f})" if abs(d) < 10 else f"({sign}{d:.2f})"


def diff_quality(con: Console, a: dict, b: dict, name_a: str, name_b: str) -> None:
    qa = a.get("quality", {}).get("tasks", {})
    qb = b.get("quality", {}).get("tasks", {})
    if not qa and not qb:
        return
    t = Table(title=f"Quality: {name_a} vs {name_b}", show_header=True, header_style="bold")
    t.add_column("Task")
    t.add_column("Metric")
    t.add_column(name_a, justify="right")
    t.add_column(name_b, justify="right")
    t.add_column("Δ", justify="right")
    for task in sorted(set(qa) | set(qb)):
        ra, rb = qa.get(task, {}), qb.get(task, {})
        metric = ra.get("metric") or rb.get("metric") or "—"
        sa, sb = ra.get("score"), rb.get("score")
        t.add_row(task, metric, fmt(sa), fmt(sb), fmt_delta(sa, sb))
    con.print(t)


def diff_speed(con: Console, a: dict, b: dict, name_a: str, name_b: str) -> None:
    sa = a.get("speed", {}).get("concurrency_results", {})
    sb = b.get("speed", {}).get("concurrency_results", {})
    if not sa and not sb:
        return
    metrics = ["tok_per_sec_overall", "ttft_p50_ms", "ttft_p95_ms",
               "itl_p50_ms", "itl_p95_ms"]
    t = Table(title=f"Speed: {name_a} vs {name_b}", show_header=True, header_style="bold")
    t.add_column("Concurrency")
    t.add_column("Metric")
    t.add_column(name_a, justify="right")
    t.add_column(name_b, justify="right")
    t.add_column("Δ", justify="right")
    levels = sorted(set(sa) | set(sb), key=lambda x: int(x))
    for c in levels:
        for m in metrics:
            va = sa.get(c, {}).get(m)
            vb = sb.get(c, {}).get(m)
            t.add_row(c, m, fmt(va), fmt(vb), fmt_delta(va, vb))
        t.add_section()
    con.print(t)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="First result directory (results/<model>/<date>)")
    p.add_argument("b", help="Second result directory")
    args = p.parse_args()

    pa, pb = pathlib.Path(args.a), pathlib.Path(args.b)
    if not pa.is_dir() or not pb.is_dir():
        sys.exit("both arguments must be result directories")

    da, db = load_dir(pa), load_dir(pb)
    name_a = da.get("meta", {}).get("model", pa.parent.name)
    name_b = db.get("meta", {}).get("model", pb.parent.name)

    con = Console()
    diff_quality(con, da, db, name_a, name_b)
    diff_speed(con, da, db, name_a, name_b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
