"""Quality runner: wraps lm-evaluation-harness to run benchmarks against a
running vLLM (OpenAI-compatible) endpoint.

Targets the /v1/completions endpoint via lm-eval's `local-completions` model
type, which is the right path for **loglikelihood-scored** multi-choice tasks
(MMLU, HellaSwag, ARC). For generative tasks (GSM8K, MATH) use the sister
runner quality_chat_lm_eval.py — see notes/learnings/2026-05-05-lm-eval-chat-template-limitation.md
for why one runner can't do both.

Outputs a single JSON file with the standardized rhumb schema:

    {
      "runner": "lm_eval",
      "model": "<name>",
      "model_id": "<hf-id-or-served-name>",
      "endpoint": "<url>",
      "suite": "<suite-name>",
      "thinking": "on" | "off",
      "tasks": {
        "mmlu":      {"score": 0.xx, "metric": "acc", "stderr": 0.xx, ...},
        ...
      },
      "raw_output_dir": "<lm-eval results dir>",
      "wall_seconds": ...,
      "started_at": "<ISO8601>",
      "finished_at": "<ISO8601>"
    }

Usage:
    python runners/quality_lm_eval.py --model qwen-32b-awq --suite standard \\
        --out results/qwen-32b-awq/2026-05-05/quality.json
    python runners/quality_lm_eval.py --model qwen3-32b-awq --suite standard_q3 \\
        --thinking off --out ...
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
    """Resolve thinking-mode setting. Returns True if thinking should be ENABLED.

    Priority: CLI > suite.force_thinking > model.disable_thinking (inverted).
    - CLI: 'on' -> True; 'off' -> False; 'auto' -> defer.
    - suite.force_thinking: True/False/None.
    - model.disable_thinking: True (-> thinking off) / False/missing (-> thinking on).
    """
    if cli_flag == "on":
        return True
    if cli_flag == "off":
        return False
    # auto
    if isinstance(suite_force, bool):
        return suite_force
    return not bool(model_disable)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Key from models.yaml")
    p.add_argument("--suite", default="quick", help="Suite name from suites/")
    p.add_argument("--out", required=True, help="Path to write final JSON")
    p.add_argument("--thinking", choices=["on", "off", "auto"], default="auto",
                   help="Thinking mode override; 'auto' defers to suite.force_thinking then model.disable_thinking")
    p.add_argument("--endpoint", default=None,
                   help="Logical endpoint name (informational; stamped in output JSON). Typically passed by run.sh after endpoints.yaml resolution.")
    p.add_argument("--base-url", dest="base_url", default=None,
                   help="OpenAI-compatible base URL (e.g. http://node-2.local:8000/v1). Overrides the model's `endpoint:` from models.yaml.")
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
    print(f"[quality] resolved thinking={'on' if thinking_on else 'off'} "
          f"(cli={args.thinking}, suite.force_thinking={suite_yaml.get('force_thinking')}, "
          f"model.disable_thinking={model_cfg.get('disable_thinking')})", flush=True)

    api_key = os.environ.get(model_cfg.get("api_key_env", "VLLM_API_KEY"), "")

    # Endpoint resolution: --base-url (typically passed by run.sh after
    # endpoints.yaml lookup) wins; fall back to the model's own `endpoint:`.
    endpoint_root = (args.base_url or model_cfg["endpoint"]).rstrip("/")
    endpoint_name = args.endpoint or "model-default"
    print(f"[quality] endpoint={endpoint_name} base={endpoint_root}", flush=True)

    # JSON model_args — lm-eval-harness 0.4.5+ auto-detects JSON vs comma-separated;
    # JSON is the safer form because dict-valued kwargs (like chat_template_kwargs)
    # would otherwise collide with the comma separator.
    base_url = endpoint_root + "/completions"
    model_args_dict: dict = {
        "base_url": base_url,
        "model": model_cfg["model_id"],
        "tokenizer": model_cfg.get("tokenizer", model_cfg["model_id"]),
        "num_concurrent": suite.get("num_concurrent", 4),
        "tokenizer_backend": "huggingface",
    }
    # batch_size is plumbed via the CLI flag (--batch_size) below, NOT via
    # model_args, because lm-eval's CLI default is 1 and gets injected into
    # additional_config; passing it in both raises TypeError "got multiple
    # values for keyword argument 'batch_size'".
    # max_length override: lm-eval's local-completions defaults to 2048 (see
    # api_models.py:133), then subtracts 1 for the continuation slot, so any
    # 5-shot prompt over ~2K tokens gets silently left-truncated. Long-stem
    # benchmarks (MedXpertQA, etc.) blow past 2K easily. Models.yaml may set
    # `max_length:` to match the served vLLM max_model_len; we don't auto-pull
    # from /v1/models because the serving cap may be intentionally lower than
    # the model's full context (memory, throughput). Default leaves lm-eval's
    # 2048; set explicitly per model for medical/long-stem suites.
    if model_cfg.get("max_length") is not None:
        model_args_dict["max_length"] = int(model_cfg["max_length"])
    # Instruct/chat models: chat template wraps each request in <|im_start|>...
    # (only applied to loglikelihood tasks by lm-eval; generate_until ignores it,
    # see notes/learnings/2026-05-05-lm-eval-chat-template-limitation.md).
    if model_cfg.get("instruct"):
        model_args_dict["apply_chat_template"] = True
        model_args_dict["fewshot_as_multiturn"] = True
    # Thread thinking-mode through the tokenizer's chat-template kwargs.
    # For loglikelihood scoring on multi-choice tasks, thinking-off is generally
    # correct (see notes/learnings/2026-05-05-eval-metric-trap.md): forcing the
    # model into "I should reason" mode then scoring single-token answers is
    # off-distribution. We expose the knob anyway for fair-comparison runs.
    if model_cfg.get("instruct"):
        model_args_dict["chat_template_kwargs"] = {"enable_thinking": thinking_on}
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    model_args = json.dumps(model_args_dict)

    out_path = pathlib.Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_path.parent / "lm_eval_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cli = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", model_args,
        "--tasks", ",".join(suite["tasks"]),
        "--num_fewshot", str(suite["num_fewshot"]),
        "--output_path", str(raw_dir),
        "--seed", "3407",
        "--verbosity", "INFO",
        "--log_samples",
    ]
    if suite.get("limit") is not None:
        cli.extend(["--limit", str(suite["limit"])])

    # Server-side batched prompts: lm-eval packs `batch_size` requests into
    # one HTTP call (vLLM accepts prompt as list[list[int]]). Cuts HTTP
    # round-trip overhead on long-fewshot MCQ sweeps. Lossless — just a
    # payload-shape change. Default 1 (lm-eval's CLI default) preserves
    # current behavior for suites that don't set it.
    if suite.get("batch_size") is not None:
        cli.extend(["--batch_size", str(suite["batch_size"])])

    # Custom-task discovery: suites may declare `include_path: tasks` (relative
    # to repo root) to surface task YAMLs that aren't packaged in lm-eval. Used
    # by suites/medical_mcq.yaml for MedXpertQA — see tasks/medxpertqa/.
    # Read from suite_yaml (top-level), not suite (which is suite_yaml["quality"]).
    if suite_yaml.get("include_path"):
        cli.extend(["--include_path", str(REPO / suite_yaml["include_path"])])

    print(f"[quality] running: {' '.join(cli)}", flush=True)

    sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    started_at = datetime.datetime.now(datetime.timezone.utc)
    t0 = time.perf_counter()
    rc = subprocess.run(cli, env=sub_env).returncode
    wall = time.perf_counter() - t0
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    if rc != 0:
        print(f"[quality] lm_eval failed (exit {rc})", file=sys.stderr)
        return rc

    candidates = sorted(raw_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("[quality] no results_*.json produced", file=sys.stderr)
        return 1
    raw = json.loads(candidates[0].read_text())

    baselines = model_cfg.get("baselines", {}) or {}
    # Drift thresholds: suite-level default + optional per-task overrides.
    # Per-task lets the medical_mcq suite tighten to 3pp on the high-n tasks
    # (medxpertqa_text n=2455, medqa_4options n=1273, medmcqa n=4183) and
    # widen to 7pp on the small MMLU subjects (n=100-272) where 5pp is below
    # statistical resolution. See suites/medical_mcq.yaml drift_thresholds.
    default_drift_threshold = suite.get("drift_threshold", 0.05)
    per_task_drift = suite.get("drift_thresholds", {}) or {}

    tasks_summary = {}
    drift_warnings: list[str] = []
    for task_name, results in raw.get("results", {}).items():
        principal = None
        for key in ("acc_norm,none", "exact_match,none", "acc,none"):
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
        entry = {"score": score, "metric": metric_name, "stderr": stderr}

        baseline = baselines.get(task_name)
        if baseline is not None:
            threshold = per_task_drift.get(task_name, default_drift_threshold)
            delta = score - baseline
            entry["baseline"] = baseline
            entry["delta"] = round(delta, 4)
            entry["drift_threshold"] = threshold
            entry["drift"] = abs(delta) > threshold
            if entry["drift"]:
                sign = "+" if delta > 0 else ""
                drift_warnings.append(
                    f"  {task_name}: {score:.4f} (baseline {baseline:.4f}, {sign}{delta:.4f}, threshold {threshold:.0%})"
                )
        tasks_summary[task_name] = entry

    if drift_warnings:
        print(f"[quality] WARNING: {len(drift_warnings)} task(s) drifted from baseline:", file=sys.stderr)
        for line in drift_warnings:
            print(line, file=sys.stderr)
    else:
        print(f"[quality] all tasks within drift thresholds", flush=True)

    out = {
        "runner": "lm_eval",
        "model": args.model,
        "model_id": model_cfg["model_id"],
        "endpoint": endpoint_root,
        "endpoint_name": endpoint_name,
        "suite": args.suite,
        "thinking": "on" if thinking_on else "off",
        "tasks": tasks_summary,
        "drift_threshold_default": default_drift_threshold,
        "drift_thresholds_per_task": per_task_drift,
        "drift_count": len(drift_warnings),
        "raw_output_dir": str(raw_dir),
        "wall_seconds": round(wall, 2),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[quality] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
