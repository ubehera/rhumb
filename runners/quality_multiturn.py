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

    leak = [r["id"] for r in rows
            if r["bucket"] == "coupled" and r["turns"] and r["turns"][0]["visible_chars"] > 40]
    think_fail = sorted({r["id"] for r in rows
                         if any(t["reasoning_chars"] == 0 for t in r["turns"])})
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
        "thinking_failure_ids": think_fail,
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
    if think_fail:
        print(f"[multiturn] WARNING: {len(think_fail)} items had a turn with EMPTY reasoning (thinking didn't fire; preserve is inert there): {think_fail}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
