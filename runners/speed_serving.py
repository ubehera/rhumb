"""Speed runner: hits a running vLLM (OpenAI-compatible /v1/chat/completions)
endpoint with N streaming requests at varying concurrency, measures TTFT, ITL,
and overall throughput.

Output JSON schema:

    {
      "runner": "speed_serving",
      "model": "<name>",
      "endpoint": "<url>",
      "suite": "<suite-name>",
      "concurrency_results": {
        "1":  {"requests": 50, "wall_seconds": ..., "tokens_total": ...,
               "tok_per_sec_overall": ..., "ttft_p50_ms": ..., "ttft_p95_ms": ...,
               "itl_p50_ms": ..., "itl_p95_ms": ...},
        "4":  {...},
        ...
      },
      "started_at": "...",
      "finished_at": "..."
    }
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import pathlib
import statistics
import sys
import time

import httpx
import yaml


REPO = pathlib.Path(__file__).resolve().parent.parent


# A small fixed prompt set spanning short/medium/long inputs. We cycle through
# this list to fill `num_prompts`. Same prompts every run = deterministic shape.
PROMPTS = [
    "What is the capital of France?",
    "Write a one-sentence definition of recursion.",
    "Explain bubble sort in plain English. Keep it under 100 words.",
    "List five common Python data structures with one-line descriptions of each.",
    "In 2-3 sentences, what is the difference between a stack and a queue?",
    "Give me a haiku about computers.",
    "What does TCP stand for and what does it do, briefly?",
    "Summarize the plot of Hamlet in three sentences.",
    "Why is the sky blue? Keep it concise.",
    (
        "You are a senior systems engineer. A junior asks: 'When should I use a "
        "message queue versus calling a service synchronously?' Give a thoughtful, "
        "concrete answer with two examples of each, in 200 words or fewer."
    ),
]


async def one_request(client: httpx.AsyncClient, base_url: str, model: str, api_key: str,
                      prompt: str, max_tokens: int) -> dict:
    """Streams one chat completion, returns timing + token count."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    token_times: list[float] = []
    start = time.perf_counter()
    async with client.stream("POST", base_url + "/chat/completions",
                             headers=headers, json=body, timeout=300.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if "content" in delta and delta["content"]:
                token_times.append(time.perf_counter())
    end = time.perf_counter()

    if not token_times:
        return {"error": "no tokens received", "wall": end - start}

    ttft = (token_times[0] - start) * 1000.0
    itls = [(token_times[i] - token_times[i - 1]) * 1000.0 for i in range(1, len(token_times))]
    return {
        "wall": end - start,
        "tokens": len(token_times),
        "ttft_ms": ttft,
        "itl_ms_list": itls,
    }


async def run_concurrency(base_url: str, model: str, api_key: str,
                          num_prompts: int, max_tokens: int, concurrency: int) -> dict:
    """Issue num_prompts requests with the specified concurrency."""
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def bounded(p: str):
        async with sem:
            return await one_request(client, base_url, model, api_key, p, max_tokens)

    async with httpx.AsyncClient(http2=False) as client:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]
        t0 = time.perf_counter()
        results = await asyncio.gather(*[bounded(p) for p in prompts])
        wall = time.perf_counter() - t0

    successes = [r for r in results if "tokens" in r]
    if not successes:
        return {"error": "no successful requests", "wall_seconds": wall}

    ttfts = [r["ttft_ms"] for r in successes]
    itls = [v for r in successes for v in r["itl_ms_list"]]
    tokens_total = sum(r["tokens"] for r in successes)

    def pct(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        return float(statistics.quantiles(xs, n=100, method="inclusive")[max(0, min(99, int(q) - 1))])

    return {
        "requests": len(successes),
        "failures": len(results) - len(successes),
        "wall_seconds": round(wall, 3),
        "tokens_total": tokens_total,
        "tok_per_sec_overall": round(tokens_total / wall, 2) if wall > 0 else None,
        "ttft_p50_ms": round(pct(ttfts, 50), 2),
        "ttft_p95_ms": round(pct(ttfts, 95), 2),
        "ttft_p99_ms": round(pct(ttfts, 99), 2),
        "itl_p50_ms": round(pct(itls, 50), 2),
        "itl_p95_ms": round(pct(itls, 95), 2),
        "itl_p99_ms": round(pct(itls, 99), 2),
    }


async def main_async(args) -> int:
    models = yaml.safe_load((REPO / "models.yaml").read_text())["models"]
    if args.model not in models:
        sys.exit(f"unknown model '{args.model}' (known: {list(models)})")
    cfg = models[args.model]
    suite = yaml.safe_load((REPO / "suites" / f"{args.suite}.yaml").read_text())["speed"]

    api_key = os.environ.get(cfg.get("api_key_env", "VLLM_API_KEY"), "")
    if not api_key:
        print(f"[speed] WARNING: ${cfg.get('api_key_env', 'VLLM_API_KEY')} unset; "
              "requests may fail with 401", file=sys.stderr)

    # Endpoint resolution: --base-url wins; fall back to the model's `endpoint:`.
    endpoint_root = (args.base_url or cfg["endpoint"]).rstrip("/")
    endpoint_name = args.endpoint or "model-default"
    print(f"[speed] endpoint={endpoint_name} base={endpoint_root}", flush=True)
    base_url = endpoint_root

    started_at = datetime.datetime.now(datetime.timezone.utc)
    concurrency_results = {}
    for concurrency in suite["concurrency_levels"]:
        print(f"[speed] concurrency={concurrency} num_prompts={suite['num_prompts']} ...", flush=True)
        r = await run_concurrency(base_url, cfg["model_id"], api_key,
                                  suite["num_prompts"], suite["output_tokens"], concurrency)
        concurrency_results[str(concurrency)] = r
        print(f"  -> {json.dumps(r)}", flush=True)
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    out = {
        "runner": "speed_serving",
        "model": args.model,
        "model_id": cfg["model_id"],
        "endpoint": endpoint_root,
        "endpoint_name": endpoint_name,
        "suite": args.suite,
        "config": {"num_prompts": suite["num_prompts"],
                   "output_tokens": suite["output_tokens"],
                   "concurrency_levels": suite["concurrency_levels"]},
        "concurrency_results": concurrency_results,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }

    out_path = pathlib.Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[speed] wrote {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Key from models.yaml")
    p.add_argument("--suite", default="quick", help="Suite name from suites/")
    p.add_argument("--out", required=True, help="Path to write final JSON")
    p.add_argument("--endpoint", default=None,
                   help="Logical endpoint name (informational; stamped in output JSON). Typically passed by run.sh after endpoints.yaml resolution.")
    p.add_argument("--base-url", dest="base_url", default=None,
                   help="OpenAI-compatible base URL. Overrides the model's `endpoint:` from models.yaml.")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
