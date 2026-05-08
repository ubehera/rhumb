# rhumb

Quality-benchmark harness for locally-served LLMs, with first-class drift-check against published baselines.

A *rhumb line* is the navigator's path of constant compass bearing — the line a sailor holds to know they haven't drifted off course. This project applies the same idea to LLM evals: every measured score is checked against a published baseline (model card / paper), and deltas above a configurable threshold are flagged as drift warnings rather than silently absorbed into the result.

## Why this exists

Most lm-evaluation-harness wrappers report numbers in isolation. Three silent-failure modes that this harness catches by design (each one has caused real misleading runs in the past two weeks):

1. **Metric trap** — reporting `acc` when the published baseline used `acc_norm` (or vice versa). HellaSwag's `acc_norm` is ~10pp higher than `acc` for typical Qwen models; mixing them turns a healthy result into a fake regression.
2. **Baseline drought** — taking aggregate "MMLU = 0.78" as a baseline when only per-subject numbers exist for some variants. rhumb stores per-subject baselines explicitly so subject-level drift is visible.
3. **Chat-template gap** — `lm-eval`'s `local-completions` model silently ignores `apply_chat_template` for `generate_until` tasks. rhumb routes those tasks through a separate `chat_lm_eval` runner that hits the chat-completions endpoint directly, where the chat template is applied server-side by vLLM.

## Quick start

```bash
# Install
uv sync

# Loglikelihood-scored standard suite (MMLU subjects + HellaSwag)
./run.sh --model qwen3.6-27b-awq --suite standard --thinking off --detach

# Generative reasoning suite (GSM8K via /v1/chat/completions, thinking on)
./run.sh --model qwen3.6-27b-awq --suite reasoning --thinking on --detach

# Multi-model sweep with automatic vLLM model swap between runs
./scripts/run_reasoning_sweep.sh --detach
```

Results land in `results/<model>/<date>/` (or `results/<model>/<date>/<endpoint>/` if `--endpoint` is set).

## Architecture

| Component | Role |
|---|---|
| `run.sh` | Orchestrator. Reads suite YAML, dispatches to the right runner. Supports `--detach` (setsid+nohup so runs survive parent exit), `--endpoint <name>` (multi-vLLM-server topologies), `--thinking on/off/auto`. |
| `runners/quality_lm_eval.py` | Wraps lm-eval-harness for loglikelihood-scored multi-choice tasks (MMLU, HellaSwag, ARC). |
| `runners/quality_chat_lm_eval.py` | Sister runner for generative tasks (GSM8K, MATH, IFEval) via `/v1/chat/completions`. Necessary because `local-completions` doesn't apply chat templates to generative tasks. |
| `runners/speed_serving.py` | TTFT / ITL / throughput sweep across concurrency levels. |
| `scripts/run_reasoning_sweep.sh` | Orchestrates multi-model runs with automatic vLLM model swap between models. |
| `models.yaml` | Model registry with per-task baselines for drift-check. |
| `endpoints.yaml` | Logical endpoint registry for multi-vLLM-server topologies. |
| `suites/` | Task collections (`quick`, `standard`, `standard_q3`, `reasoning`, `full`). |

## Drift check

For each `(model, task)` pair where `models.yaml.<model>.baselines.<task>` is set, the runner computes `delta = score - baseline` and stamps `drift: true` in the output JSON when `|delta| > 0.05`. The threshold is configurable per-runner. The point is to make a 5-percentage-point regression in HellaSwag scream, not whisper.

Baselines belong to one of two registers:

- **Published** (paper / model card numbers, methodology-adjusted where the public number used 8-shot CoT and the local run uses 5-shot non-CoT, etc.).
- **Measured-as-baseline** (the model's own first-good run on a task that has no public per-subject number — used as a regression anchor for *future* runs of the same model, not as a quality claim).

`models.yaml` documents which is which per task.

## Multi-endpoint support

`endpoints.yaml` registers logical names (e.g. `local`, `node-1`, `node-2`) → vLLM URLs + auth config. Use `--endpoint <name>` to target a specific server; results write to `results/<model>/<date>/<endpoint>/` so parallel runs against different boxes don't clobber.

With both nodes wired up (200G fabric operational since 2026-05-07), this enables data-parallel cross-model comparisons: e.g. Qwen 2.5 on `node-1` and Qwen 3 on `node-2` running simultaneously.

## What's not included

- Model serving — bring your own vLLM (or any OpenAI-compatible endpoint).
- Training / fine-tuning.
- Anything agentic — that's the scope of `localbench`, a planned sibling project currently in design. Not yet public. rhumb stays focused on standard quality benchmarks against published baselines.

## License

Apache 2.0 — see [LICENSE](LICENSE).
