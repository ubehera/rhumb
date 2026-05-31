"""Judge-graded multiturn variant: measures Qwen3.6 preserve_thinking ON vs OFF
on OPEN-ENDED coupled conversations, scored 1-10 by a third-family LLM judge
(Kimi K2.6 / Moonshot), blind to condition.

The exact-match probe (quality_multiturn.py) hit a structural 100% ceiling: for
deterministic, fully-specified problems the OFF arm just RE-DERIVES the needed
intermediate from the still-visible turn-1 prompt. This runner targets the
realistic regime instead — turn-1 produces a TERSE conclusion (rich reasoning
stays in the <think>), turn-2 builds on it, and we ask whether access to that
preserved REASONING (vs only the visible conclusion) improves turn-2 quality.

The judge sees an IDENTICAL view in both arms — turn-1 user prompt, turn-1
VISIBLE answer, turn-2 user prompt, turn-2 answer — so the only thing that can
move the score is turn-2's content. preserve_thinking is the sole variable in
generation (server-side Qwen template keeps/strips the prior <think>).

Usage (via run.sh, suite routes here when runner: multiturn_judge):
    ./run.sh --model qwen3.6-27b-fp8 --suite preserve_thinking_judge \\
        --only quality --endpoint sglang-tp2 --thinking on --detach
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import json
import os
import pathlib
import re
import statistics
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


# ---------- judge ----------

JUDGE_TMPL = """You are a strict, expert evaluator. Score ONLY the assistant's FINAL answer (turn 2) for QUALITY, on an integer 1-10 scale.

=== CONVERSATION ===
[Turn 1 - user]
{t1u}

[Turn 1 - assistant]
{t1a}

[Turn 2 - user]
{t2u}

[Turn 2 - assistant]   <-- EVALUATE THIS
{t2a}
=== END ===

Rubric for the turn-2 answer:
{rubric}

Judge on: correctness, technical depth, specificity, and COHERENCE with the assistant's own turn-1 answer (it must build on turn 1, not contradict or ignore it). Penalize vagueness, hand-waving, and inconsistency with turn 1. Do NOT reward mere length.

Reply with ONLY a JSON object, no prose before or after:
{{"score": <integer 1-10>, "reason": "<one concise sentence>"}}"""


def parse_score(content: str | None):
    """Extract {score, reason} from judge output. Robust to code fences and to a
    reasoning preamble before the JSON: take the LAST json object that has a
    'score'. Returns (int|None, reason_str)."""
    if not content:
        return None, ""
    t = content.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    best = None
    for m in re.finditer(r"\{[^{}]*\}", t, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "score" in obj:
            best = obj  # keep last
    if best is not None:
        try:
            sc = int(round(float(best["score"])))
            return max(1, min(10, sc)), str(best.get("reason", ""))[:300]
        except Exception:
            pass
    m = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', t, re.I)
    if m:
        sc = int(round(float(m.group(1))))
        return max(1, min(10, sc)), ""
    return None, ""


def judge_one(jclient, jurl, jheaders, jmodel, prompt, temp, seed, timeout):
    body = {"model": jmodel, "messages": [{"role": "user", "content": prompt}],
            "temperature": temp, "seed": seed, "max_tokens": 8000}
    r = jclient.post(jurl, headers=jheaders, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = (data["choices"][0]["message"].get("content") or "")
    return parse_score(content)


def judge_turn(jcfg, item, t1u, t1a, t2u, t2a):
    prompt = JUDGE_TMPL.format(t1u=t1u, t1a=t1a, t2u=t2u, t2a=t2a, rubric=item["judge_rubric"])
    scores, reasons = [], []
    with httpx.Client() as jclient, cf.ThreadPoolExecutor(max_workers=jcfg["samples"]) as ex:
        futs = [ex.submit(judge_one, jclient, jcfg["url"], jcfg["headers"], jcfg["model"],
                          prompt, jcfg["temperature"], jcfg["seed_base"] + j, jcfg["timeout"])
                for j in range(jcfg["samples"])]
        for fut in cf.as_completed(futs):
            try:
                sc, rs = fut.result()
            except Exception as e:  # noqa: BLE001
                sc, rs = None, f"JUDGE_ERROR {type(e).__name__}"
            if sc is not None:
                scores.append(sc)
                reasons.append(rs)
    mean_score = statistics.mean(scores) if scores else None
    return mean_score, scores, reasons


# ---------- generation ----------

def post_stream(client, url, headers, body, timeout):
    """STREAMING chat completion. rhumb is otherwise a non-streaming client, but
    SGLang's --stream-interval 32 (the detokenizer-stall fix) ONLY governs
    req.stream=True; the non-streaming path falls back to the weaker
    SGLANG_FORCE_STREAM_INTERVAL and stalls on very long generations (the judge
    items think for thousands of tokens). Streaming keeps the detokenizer
    flushing every 32 tokens. Generation is identical (same tokens/seed) — only
    delivery differs. Returns (reasoning, content, usage, finish_reason)."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    reasoning, content = [], []
    usage, finish = {}, None
    to = httpx.Timeout(timeout, connect=30.0, read=timeout, write=30.0, pool=30.0)
    with client.stream("POST", url, headers=headers, json=body, timeout=to) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []) or []:
                delta = ch.get("delta", {}) or {}
                rc = delta.get("reasoning_content")
                if rc:
                    reasoning.append(rc)
                c = delta.get("content")
                if c:
                    content.append(c)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return "".join(reasoning), "".join(content), usage, finish


def run_conversation(client, url, headers, model_id, item, preserve, decoding, seed, timeout, jcfg):
    """Drive one open-ended conversation under one condition+seed, then judge the
    final turn (blind). No retries on generation (post-completion errors must not
    duplicate work)."""
    history: list[dict] = []
    turns_out = []
    t1_visible = ""
    final_visible = ""
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
        reasoning, content_raw, usage, finish = post_stream(client, url, headers, body, timeout)
        content = (content_raw or "").strip()
        is_final = ti == len(item["turns"]) - 1
        if ti == 0:
            t1_visible = content
        turns_out.append({
            "turn": ti,
            "reasoning_chars": len(reasoning),
            "visible_chars": len(content),
            "content": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "finish_reason": finish,
        })
        if is_final:
            final_visible = content
            final_prompt_tokens = usage.get("prompt_tokens", 0)
            final_total_tokens = usage.get("total_tokens",
                                           usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
        history.append({"role": "assistant",
                        "content": ml.reconstruct_assistant_content(reasoning, content)})

    mean_score, scores, reasons = judge_turn(
        jcfg, item, item["turns"][0]["content"], t1_visible,
        item["turns"][-1]["content"], final_visible)

    return {
        "id": item["id"],
        "bucket": item["bucket"],
        "condition": "on" if preserve else "off",
        "seed": seed,
        "judge_score": mean_score,
        "judge_scores": scores,
        "judge_reasons": reasons,
        "prompt_tokens": final_prompt_tokens,
        "total_tokens": final_total_tokens,
        "turns": turns_out,
    }


# ---------- aggregation ----------

def aggregate_scores(rows: list[dict]) -> dict:
    groups: dict = {}
    for r in rows:
        if r.get("judge_score") is None:
            continue
        if r["turns"] and r["turns"][0]["visible_chars"] == 0:
            continue  # empty turn-1 conclusion: OFF would have nothing to build on => confound
        groups.setdefault((r["condition"], r["bucket"]), []).append(r)
    out: dict = {}
    for (cond, bucket), rs in groups.items():
        vals = [r["judge_score"] for r in rs]
        n = len(vals)
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if n > 1 else 0.0
        sem = (sd / (n ** 0.5)) if n else 0.0
        out.setdefault(cond, {})[bucket] = {
            "mean_score": round(mean, 3),
            "sd": round(sd, 3),
            "ci_low": round(mean - 1.96 * sem, 3),
            "ci_high": round(mean + 1.96 * sem, 3),
            "n": n,
            "mean_total_tokens": round(statistics.mean(r["total_tokens"] for r in rs), 1) if n else None,
        }
    return out


def paired_delta(rows: list[dict], bucket: str) -> dict:
    """Paired ON-OFF delta on matched (id, seed). More powerful than the
    between-group mean diff since the same item+seed is compared across arms."""
    by_key: dict = {}
    for r in rows:
        if r["bucket"] != bucket or r.get("judge_score") is None:
            continue
        if r["turns"] and r["turns"][0]["visible_chars"] == 0:
            continue
        by_key.setdefault((r["id"], r["seed"]), {})[r["condition"]] = r["judge_score"]
    deltas = [v["on"] - v["off"] for v in by_key.values() if "on" in v and "off" in v]
    if not deltas:
        return {"n_pairs": 0, "mean_delta": None, "ci_low": None, "ci_high": None}
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.pstdev(deltas) if n > 1 else 0.0
    sem = sd / (n ** 0.5) if n else 0.0
    return {"n_pairs": n, "mean_delta": round(mean, 3),
            "ci_low": round(mean - 1.96 * sem, 3), "ci_high": round(mean + 1.96 * sem, 3),
            "sd": round(sd, 3)}


def judge_headline(rows: list[dict], agg: dict) -> dict:
    def mean(cond, bucket):
        return agg.get(cond, {}).get(bucket, {}).get("mean_score")

    cells = [mean("on", "coupled"), mean("off", "coupled"), mean("on", "control"), mean("off", "control")]
    cp = paired_delta(rows, "coupled")
    ct = paired_delta(rows, "control")
    if any(c is None for c in cells):
        return {"verdict": "incomplete", "coupled_paired_delta": cp, "control_paired_delta": ct,
                "coupled_minus_control": None}
    coupled_d = cp["mean_delta"]
    control_d = ct["mean_delta"]
    net = (coupled_d or 0) - (control_d or 0)
    # On a 1-10 scale: call an effect real only if the paired coupled CI excludes 0
    # AND it exceeds the control effect (net > 0.3). Conservative by design.
    coupled_sig = cp["ci_low"] is not None and cp["ci_low"] > 0
    if coupled_sig and net > 0.3:
        verdict = "helps_when_coupled"
    elif cp["ci_high"] is not None and cp["ci_high"] < -0.3:
        verdict = "hurts"
    else:
        verdict = "no_effect"
    return {
        "verdict": verdict,
        "coupled_paired_delta": cp,
        "control_paired_delta": ct,
        "coupled_minus_control": round(net, 3),
        "coupled_token_overhead_on_minus_off": round(
            (mean_tok(agg, "on", "coupled") or 0) - (mean_tok(agg, "off", "coupled") or 0), 1),
    }


def mean_tok(agg, cond, bucket):
    return agg.get(cond, {}).get(bucket, {}).get("mean_total_tokens")


# ---------- judge endpoint resolution ----------

def resolve_judge(suite_yaml: dict) -> dict:
    jc = suite_yaml.get("judge")
    if not jc:
        sys.exit("suite has no judge: block (required for multiturn_judge runner)")
    key_env = jc.get("api_key_env", "OLLAMA_API_KEY")
    key = os.environ.get(key_env, "")
    if not key:
        env_path = pathlib.Path(jc.get(
            "env_file", f"~/.localbench/{key_env.lower().replace('_api_key','')}.env")).expanduser()
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(f"{key_env}="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        sys.exit(f"judge key {key_env} not set and not found in env_file")
    base = jc["base_url"].rstrip("/")
    return {
        "model": jc["model"],
        "url": base + "/chat/completions",
        "headers": {"Authorization": f"Bearer {key}"},
        "samples": int(jc.get("samples", 3)),
        "temperature": float(jc.get("temperature", 0.3)),
        "seed_base": int(jc.get("seed_base", 11)),
        "timeout": float(jc.get("timeout", 180)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--suite", default="preserve_thinking_judge")
    p.add_argument("--out", required=True)
    p.add_argument("--thinking", choices=["on", "off", "auto"], default="auto",
                   help="Ignored — thinking is always ON (preserve is the only variable).")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--base-url", dest="base_url", default=None)
    args = p.parse_args()

    models = load_yaml(REPO / "models.yaml")["models"]
    if args.model not in models:
        sys.exit(f"unknown model '{args.model}' (known: {list(models)})")
    model_cfg = models[args.model]

    suite_yaml = load_yaml(REPO / "suites" / f"{args.suite}.yaml")
    suite = suite_yaml["quality"]
    decoding = suite["decoding"]
    samples = int(suite.get("samples", 4))
    seed_base = int(suite.get("seed_base", 3407))
    num_concurrent = int(suite.get("num_concurrent", 1))
    timeout = float(suite.get("timeout", 600))
    conditions = suite.get("conditions", ["off", "on"])
    limit = suite.get("limit")

    api_key = os.environ.get(model_cfg.get("api_key_env", "VLLM_API_KEY"), "")
    endpoint_root = (args.base_url or model_cfg["endpoint"]).rstrip("/")
    url = endpoint_root + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    model_id = model_cfg["model_id"]
    jcfg = resolve_judge(suite_yaml)
    print(f"[mt-judge] model={model_id} endpoint={args.endpoint or 'model-default'} base={url}", flush=True)
    print(f"[mt-judge] judge={jcfg['model']} judge_samples={jcfg['samples']} judge_temp={jcfg['temperature']}", flush=True)
    print(f"[mt-judge] conditions={conditions} samples={samples} concurrency={num_concurrent} decoding={decoding}", flush=True)

    out_path = pathlib.Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = pathlib.Path(str(out_path) + ".partial.jsonl")

    done_rows, done_keys = ml.load_checkpoint(ckpt)
    rows = list(done_rows)
    n_resumed = len(rows)
    if n_resumed:
        print(f"[mt-judge] resuming: {n_resumed} completed conversations loaded", flush=True)

    items = []
    with open(REPO / suite["dataset"]) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if limit:
        items = items[:limit]
    print(f"[mt-judge] {len(items)} items x {len(conditions)} conditions x {samples} samples = "
          f"{len(items) * len(conditions) * samples} conversations", flush=True)

    jobs = []
    for item in items:
        for cond in conditions:
            preserve = cond == "on"
            for s in range(samples):
                seed = seed_base + s
                if ml.job_key(item["id"], "on" if preserve else "off", seed) in done_keys:
                    continue
                jobs.append((item, preserve, seed))

    new_rows: list[dict] = []
    started_at = datetime.datetime.now(datetime.timezone.utc)
    t0 = time.perf_counter()
    with httpx.Client() as client:
        with cf.ThreadPoolExecutor(max_workers=num_concurrent) as ex:
            futs = {ex.submit(run_conversation, client, url, headers, model_id,
                              item, preserve, decoding, seed, timeout, jcfg): (item["id"], preserve, seed)
                    for (item, preserve, seed) in jobs}
            done = 0
            for fut in cf.as_completed(futs):
                ident = futs[fut]
                try:
                    row = fut.result()
                    with open(ckpt, "a") as cf_f:
                        cf_f.write(json.dumps(row) + "\n")
                        cf_f.flush()
                    rows.append(row)
                    new_rows.append(row)
                except Exception as e:  # noqa: BLE001
                    print(f"[mt-judge] FAILED {ident}: {type(e).__name__}: {e}", file=sys.stderr)
                done += 1
                if done % 10 == 0:
                    print(f"[mt-judge] {done}/{len(jobs)} conversations done this run", flush=True)
    wall = time.perf_counter() - t0
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    agg = aggregate_scores(rows)
    head = judge_headline(rows, agg)

    # Validity gates
    think_fail = sorted({r["id"] for r in rows if any(t["reasoning_chars"] == 0 for t in r["turns"])})
    judge_fail = sorted({r["id"] for r in rows if r.get("judge_score") is None})
    leak = sorted({r["id"] for r in rows if r["bucket"] == "coupled" and r["turns"]
                   and r["turns"][0]["visible_chars"] > 400})  # terse-conclusion guard (open-ended => looser than 40)
    turn1_empty = sorted([(r["id"], r["condition"], r["seed"]) for r in rows
                          if r["turns"] and r["turns"][0]["visible_chars"] == 0])  # excluded from scoring
    all_scores = [s for r in rows for s in (r.get("judge_scores") or [])]
    score_spread = (max(all_scores) - min(all_scores)) if all_scores else 0

    out = {
        "runner": "multiturn_judge",
        "model": args.model,
        "model_id": model_id,
        "endpoint": endpoint_root,
        "endpoint_name": args.endpoint or "model-default",
        "judge_model": jcfg["model"],
        "judge_samples": jcfg["samples"],
        "suite": args.suite,
        "thinking": "on",
        "decoding": decoding,
        "samples": samples,
        "seed_base": seed_base,
        "conditions": conditions,
        "summary": agg,
        "headline": head,
        "judge_score_spread": score_spread,
        "thinking_failure_ids": think_fail,
        "judge_failure_ids": judge_fail,
        "leakage_suspect_ids": leak,
        "turn1_empty_excluded": turn1_empty,
        "n_rows": len(rows),
        "n_resumed": n_resumed,
        "n_failed": len(jobs) - len(new_rows),
        "per_row": rows,
        "wall_seconds": round(wall, 2),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[mt-judge] wrote {out_path}", flush=True)
    print(f"[mt-judge] verdict={head['verdict']} coupled_paired={head.get('coupled_paired_delta')} "
          f"control_paired={head.get('control_paired_delta')}", flush=True)
    if score_spread < 2:
        print(f"[mt-judge] WARNING: judge score spread is only {score_spread} (1-10 scale) — "
              f"judge may not be discriminating; check rubric/engagement", file=sys.stderr)
    if think_fail:
        print(f"[mt-judge] WARNING: {len(think_fail)} items had an empty-reasoning turn: {think_fail}", file=sys.stderr)
    if judge_fail:
        print(f"[mt-judge] WARNING: {len(judge_fail)} items had unparseable judge output: {judge_fail}", file=sys.stderr)
    if turn1_empty:
        print(f"[mt-judge] NOTE: {len(turn1_empty)} conversations had an empty turn-1 answer (excluded from scoring): {turn1_empty}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
