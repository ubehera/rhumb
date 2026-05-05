"""Quality runner (chat-completions variant): wraps lm-evaluation-harness for
**generative** tasks (GSM8K, MATH, BBH-generate, IFEval) against a vLLM
/v1/chat/completions endpoint.

Why this is separate from quality_lm_eval.py:
  - lm-eval's `local-completions` model only honors apply_chat_template for
    LOGLIKELIHOOD tasks, not generate_until tasks (this runner's domain).
    See notes/learnings/2026-05-05-lm-eval-chat-template-limitation.md.
  - The chat-completions endpoint applies the chat template server-side, so
    instruct/chat models receive properly-formatted prompts. The trade-off:
    we lose per-token logprobs (chat-completions doesn't return them).
  - This means generative tasks finally work correctly for instruct-tuned
    models — Qwen 3 32B AWQ stops emitting "Quora autocomplete" gibberish
    on math problems.

Outputs match the quality_lm_eval.py schema for diff.py compatibility:

    {
      "runner": "chat_lm_eval",
      "thinking": "on" | "off",
      "tasks": {"gsm8k": {"score": ..., "metric": "exact_match", ...}},
      ...
    }

Usage:
    python runners/quality_chat_lm_eval.py --model qwen3-32b-awq \\
        --suite reasoning --thinking on --out results/.../reasoning.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

import yaml


REPO = pathlib.Path(__file__).resolve().parent.parent


def load_yaml(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_thinking(cli_flag: str, suite_force: object, model_disable: object) -> bool:
    """Same priority resolution as quality_lm_eval.py — kept inline rather than
    factored to a shared module to keep each runner standalone."""
    if cli_flag == "on":
        return True
    if cli_flag == "off":
        return False
    if isinstance(suite_force, bool):
        return suite_force
    return not bool(model_disable)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Key from models.yaml")
    p.add_argument("--suite", default="reasoning", help="Suite name from suites/ (typically reasoning)")
    p.add_argument("--out", required=True, help="Path to write final JSON")
    p.add_argument("--thinking", choices=["on", "off", "auto"], default="auto",
                   help="Thinking mode override; 'auto' defers to suite.force_thinking then model.disable_thinking")
    p.add_argument("--endpoint", default=None,
                   help="Logical endpoint name (informational; stamped in output JSON). Typically passed by run.sh after endpoints.yaml resolution.")
    p.add_argument("--base-url", dest="base_url", default=None,
                   help="OpenAI-compatible base URL (e.g. http://box-b.local:8000/v1). Overrides the model's `endpoint:` from models.yaml.")
    args = p.parse_args()

    models = load_yaml(REPO / "models.yaml")["models"]
    if args.model not in models:
        sys.exit(f"unknown model '{args.model}' (known: {list(models)})")
    model_cfg = models[args.model]

    suite_path = REPO / "suites" / f"{args.suite}.yaml"
    if not suite_path.exists():
        sys.exit(f"unknown suite '{args.suite}' (file {suite_path} not found)")
    suite_yaml = load_yaml(suite_path)
    suite = suite_yaml["quality"]

    thinking_on = resolve_thinking(args.thinking, suite_yaml.get("force_thinking"), model_cfg.get("disable_thinking"))
    print(f"[quality-chat] resolved thinking={'on' if thinking_on else 'off'} "
          f"(cli={args.thinking}, suite.force_thinking={suite_yaml.get('force_thinking')}, "
          f"model.disable_thinking={model_cfg.get('disable_thinking')})", flush=True)

    api_key = os.environ.get(model_cfg.get("api_key_env", "VLLM_API_KEY"), "")

    # Endpoint resolution: --base-url (typically passed by run.sh after
    # endpoints.yaml lookup) wins; fall back to the model's own `endpoint:`.
    endpoint_root = (args.base_url or model_cfg["endpoint"]).rstrip("/")
    endpoint_name = args.endpoint or "model-default"
    print(f"[quality-chat] endpoint={endpoint_name} base={endpoint_root}", flush=True)

    # base_url for local-chat-completions points at the chat-completions endpoint.
    # vLLM applies the tokenizer's chat template server-side; we don't need
    # apply_chat_template/tokenizer args. We DO pass chat_template_kwargs in the
    # request body — vLLM forwards them to the tokenizer.
    base_url = endpoint_root + "/chat/completions"
    model_args_dict: dict = {
        "base_url": base_url,
        "model": model_cfg["model_id"],
        "tokenizer": model_cfg.get("tokenizer", model_cfg["model_id"]),
        "num_concurrent": suite.get("num_concurrent", 4),
        "tokenizer_backend": "huggingface",
    }
    if model_cfg.get("instruct"):
        # NOTE: apply_chat_template and fewshot_as_multiturn must be passed as
        # TOP-LEVEL lm-eval CLI flags for local-chat-completions, not as model_args
        # keys — local-chat-completions expects messages as list[dict] and only
        # builds them when the top-level flag is set. Putting them in model_args
        # is silently ignored and the runner crashes with:
        #   AssertionError: LocalChatCompletion expects messages as list[dict].
        # chat_template_kwargs DOES go via model_args (vLLM forwards it server-side).
        model_args_dict["chat_template_kwargs"] = {"enable_thinking": thinking_on}

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    model_args = json.dumps(model_args_dict)

    out_path = pathlib.Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_path.parent / "lm_eval_chat_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cli = [
        "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", ",".join(suite["tasks"]),
        "--num_fewshot", str(suite["num_fewshot"]),
        "--output_path", str(raw_dir),
        "--seed", "3407",
        "--verbosity", "INFO",
        # --log_samples writes samples_*.jsonl with per-question prompt+response+score.
        # Critical for diagnosing 0.0 scores on generative tasks (where the model
        # may be answering correctly but not following the answer-format expected
        # by the regex parser, especially with thinking ON).
        "--log_samples",
    ]
    if model_cfg.get("instruct"):
        cli.extend(["--apply_chat_template", "--fewshot_as_multiturn"])
    if suite.get("limit") is not None:
        cli.extend(["--limit", str(suite["limit"])])

    print(f"[quality-chat] running: {' '.join(cli)}", flush=True)

    sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    started_at = datetime.datetime.now(datetime.timezone.utc)
    t0 = time.perf_counter()
    rc = subprocess.run(cli, env=sub_env).returncode
    wall = time.perf_counter() - t0
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    if rc != 0:
        print(f"[quality-chat] lm_eval failed (exit {rc})", file=sys.stderr)
        return rc

    candidates = sorted(raw_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("[quality-chat] no results_*.json produced", file=sys.stderr)
        return 1
    raw = json.loads(candidates[0].read_text())

    baselines = model_cfg.get("baselines", {}) or {}
    drift_threshold = 0.05

    tasks_summary = {}
    drift_warnings: list[str] = []
    for task_name, results in raw.get("results", {}).items():
        # For generative tasks, exact_match is the most common principal metric;
        # acc and pass@1 also appear depending on the task. Same fallback logic.
        principal = None
        for key in ("exact_match,strict-match", "exact_match,flexible-extract",
                    "exact_match,none", "acc_norm,none", "acc,none", "pass@1,none"):
            if key in results and isinstance(results[key], (int, float)):
                principal = key
                break
        if principal is None:
            for k, v in results.items():
                if isinstance(v, (int, float)) and "stderr" not in k.lower():
                    principal = k
                    break
        if principal is None:
            tasks_summary[task_name] = {"raw": results}
            continue
        metric_name = principal.split(",")[0]
        stderr_key = principal.replace(metric_name, metric_name + "_stderr")
        score = float(results[principal])
        stderr = (float(results.get(stderr_key, 0.0))
                  if isinstance(results.get(stderr_key), (int, float)) else None)
        entry = {"score": score, "metric": metric_name, "stderr": stderr,
                 "filter": principal.split(",", 1)[1] if "," in principal else None}

        baseline = baselines.get(task_name)
        if baseline is not None:
            delta = score - baseline
            entry["baseline"] = baseline
            entry["delta"] = round(delta, 4)
            entry["drift"] = abs(delta) > drift_threshold
            if entry["drift"]:
                sign = "+" if delta > 0 else ""
                drift_warnings.append(
                    f"  {task_name}: {score:.4f} (baseline {baseline:.4f}, {sign}{delta:.4f})"
                )
        tasks_summary[task_name] = entry

    if drift_warnings:
        print(f"[quality-chat] WARNING: {len(drift_warnings)} task(s) drifted >"
              f"{drift_threshold:.0%} from baseline:", file=sys.stderr)
        for line in drift_warnings:
            print(line, file=sys.stderr)
    else:
        print(f"[quality-chat] all tasks within {drift_threshold:.0%} of baselines", flush=True)

    out = {
        "runner": "chat_lm_eval",
        "model": args.model,
        "model_id": model_cfg["model_id"],
        "endpoint": endpoint_root,
        "endpoint_name": endpoint_name,
        "suite": args.suite,
        "thinking": "on" if thinking_on else "off",
        "tasks": tasks_summary,
        "drift_threshold": drift_threshold,
        "drift_count": len(drift_warnings),
        "raw_output_dir": str(raw_dir),
        "wall_seconds": round(wall, 2),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[quality-chat] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
