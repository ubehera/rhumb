#!/usr/bin/env bash
# Top-level orchestrator. Runs the configured suite (quality + speed) for the
# named model, writes results under results/<model>/<date>/.
#
# Usage:
#   ./run.sh --model qwen-32b-awq                                       # default suite=quick
#   ./run.sh --model qwen-32b-awq --suite standard
#   ./run.sh --model qwen-32b-awq --suite standard --only quality
#   ./run.sh --model qwen3-32b-awq --suite reasoning --thinking on      # routes to chat runner
#   ./run.sh --model qwen3-32b-awq --suite standard_q3 --thinking off
#   ./run.sh --model qwen3.6-27b-awq --suite standard_q3 --detach       # survives parent exit
#   ./run.sh --model qwen3.6-27b-awq --suite quick --endpoint box-b     # target a specific vLLM
#
# Quality runner is auto-selected from the suite YAML's `runner:` field
# (default: lm_eval). Suites with `runner: chat_lm_eval` use the
# chat-completions endpoint and support generative tasks.
#
# --thinking on|off|auto threads through to the runner, overriding both the
# suite's force_thinking and the model's disable_thinking. 'auto' (default)
# lets those defaults win.
#
# --detach re-execs self under `setsid nohup` so the run survives parent exit
# (terminal close, Claude Code restart, SSH disconnect). Logs land at
# <out_dir>/run.log and the pid at <out_dir>/run.pid (paths below).
#
# --endpoint <name> picks a logical endpoint from endpoints.yaml. Without it,
# each model uses its own `endpoint:` field from models.yaml (current
# single-box default). With it:
#   - the endpoint's env_file is sourced before reading the API key
#   - the API key from endpoints.yaml.<name>.api_key_env is promoted into VLLM_API_KEY
#   - the runner is invoked with --base-url overriding the model's default
#   - results land under results/<model>/<date>/<endpoint>/ (NOT under the
#     bare date dir) so parallel runs against different boxes don't clobber.

set -euo pipefail

MODEL=""
SUITE="quick"
ONLY=""
THINKING="auto"
DETACH="0"
ENDPOINT=""

# Save the original argv before the while-loop consumes it via `shift`.
# We need it intact for the --detach re-exec.
ORIG_ARGS=("$@")

while [ $# -gt 0 ]; do
  case "$1" in
    --model)    MODEL="$2"; shift 2;;
    --suite)    SUITE="$2"; shift 2;;
    --only)     ONLY="$2"; shift 2;;
    --thinking) THINKING="$2"; shift 2;;
    --detach)   DETACH="1"; shift;;
    --endpoint) ENDPOINT="$2"; shift 2;;
    -h|--help)
      sed -n '1,/^set/p' "$0" | sed 's/^# \?//' | head -n -1
      exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [ -z "$MODEL" ]; then
  echo "--model required" >&2
  exit 1
fi

case "$THINKING" in
  on|off|auto) ;;
  *) echo "--thinking must be one of: on, off, auto" >&2; exit 1;;
esac

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE="$(date +%Y-%m-%d)"

# Endpoint resolution: when --endpoint is set, load endpoints.yaml,
# source env_file (if any), promote the named api_key_env into VLLM_API_KEY,
# and stash the resolved base_url for runner CLI passthrough. When unset,
# leave VLLM_API_KEY/base_url alone — the runner falls back to the model's
# own `endpoint:` from models.yaml (legacy single-box behavior).
EP_BASE_URL=""
if [ -n "$ENDPOINT" ]; then
  EP_YAML="$DIR/endpoints.yaml"
  if [ ! -f "$EP_YAML" ]; then
    echo "endpoints.yaml not found at $EP_YAML (required when --endpoint is set)" >&2
    exit 1
  fi
  # Three lines: base_url, api_key_env, env_file (env_file may be empty).
  EP_FIELDS="$(python3 - "$EP_YAML" "$ENDPOINT" <<'PY'
import sys, yaml
yaml_path, name = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(yaml_path))
endpoints = (d or {}).get("endpoints", {})
e = endpoints.get(name)
if not e:
    sys.exit(f"unknown endpoint '{name}' (known: {', '.join(endpoints) or '(none)'})")
print(e.get("base_url", ""))
print(e.get("api_key_env", "VLLM_API_KEY"))
print(e.get("env_file", ""))
PY
)" || exit 1
  EP_BASE_URL="$(printf '%s\n' "$EP_FIELDS" | sed -n 1p)"
  EP_API_KEY_ENV="$(printf '%s\n' "$EP_FIELDS" | sed -n 2p)"
  EP_ENV_FILE_RAW="$(printf '%s\n' "$EP_FIELDS" | sed -n 3p)"
  EP_ENV_FILE="${EP_ENV_FILE_RAW/#\~/$HOME}"

  if [ -z "$EP_BASE_URL" ]; then
    echo "endpoint '$ENDPOINT' is missing base_url in $EP_YAML" >&2
    exit 1
  fi

  if [ -n "$EP_ENV_FILE" ]; then
    if [ -f "$EP_ENV_FILE" ]; then
      echo "[run.sh] sourcing env_file: $EP_ENV_FILE"
      set -a
      # shellcheck disable=SC1090
      source "$EP_ENV_FILE"
      set +a
    else
      echo "[run.sh] WARNING: env_file '$EP_ENV_FILE' not found (continuing; \$$EP_API_KEY_ENV must already be in env)" >&2
    fi
  fi

  # Promote endpoint-specific API key into VLLM_API_KEY (where runners look).
  EP_API_KEY="$(printenv "$EP_API_KEY_ENV" || true)"
  if [ -n "$EP_API_KEY" ]; then
    export VLLM_API_KEY="$EP_API_KEY"
  else
    echo "[run.sh] WARNING: \$$EP_API_KEY_ENV is empty; requests may 401" >&2
  fi

  OUT_DIR="$DIR/results/$MODEL/$DATE/$ENDPOINT"
else
  OUT_DIR="$DIR/results/$MODEL/$DATE"
fi
mkdir -p "$OUT_DIR"

# Detach: re-exec self under setsid+nohup so the run survives parent exit.
# Sentinel env var prevents the inner invocation from forking again.
if [ "$DETACH" = "1" ] && [ -z "${LOCALBENCH_DETACHED:-}" ]; then
  LOG="$OUT_DIR/run.log"
  PID_FILE="$OUT_DIR/run.pid"
  echo "[run.sh] detaching; output -> $LOG"
  LOCALBENCH_DETACHED=1 setsid nohup "$0" "${ORIG_ARGS[@]}" >"$LOG" 2>&1 </dev/null &
  CHILD_PID=$!
  echo "$CHILD_PID" > "$PID_FILE"
  echo "[run.sh] pid:    $CHILD_PID"
  echo "[run.sh] tail:   tail -f $LOG"
  echo "[run.sh] status: ps -p $CHILD_PID -o pid,stat,etime,cmd"
  echo "[run.sh] kill:   kill \$(cat $PID_FILE)"
  exit 0
fi

# Fallback API-key sourcing only when --endpoint wasn't used. Endpoint mode
# already sourced its own env_file above.
if [ -z "$ENDPOINT" ] && [ -z "${VLLM_API_KEY:-}" ] && [ -f "$HOME/projects/vllm-server/vllm.env" ]; then
  export VLLM_API_KEY="$(grep -E '^VLLM_API_KEY=' "$HOME/projects/vllm-server/vllm.env" | cut -d= -f2-)"
fi

# Pick the quality runner from the suite YAML (default: lm_eval).
SUITE_YAML="$DIR/suites/${SUITE}.yaml"
if [ ! -f "$SUITE_YAML" ]; then
  echo "suite not found: $SUITE_YAML" >&2
  exit 1
fi
RUNNER="$(python3 -c "import yaml,sys; d=yaml.safe_load(open('$SUITE_YAML')); print(d.get('runner','lm_eval'))")"

case "$RUNNER" in
  lm_eval)      QUALITY_OUT="$OUT_DIR/quality.json";       QUALITY_SCRIPT="quality_lm_eval.py";;
  chat_lm_eval) QUALITY_OUT="$OUT_DIR/quality_chat.json";  QUALITY_SCRIPT="quality_chat_lm_eval.py";;
  *) echo "unknown runner '$RUNNER' in $SUITE_YAML (expected lm_eval or chat_lm_eval)" >&2; exit 1;;
esac

# Write a meta.json describing this run.
python3 - <<PY > "$OUT_DIR/meta.json"
import json, datetime, socket
print(json.dumps({
  "model": "$MODEL",
  "suite": "$SUITE",
  "runner": "$RUNNER",
  "thinking": "$THINKING",
  "endpoint_name": "$ENDPOINT" or "model-default",
  "endpoint_base_url": "$EP_BASE_URL" or "(from models.yaml)",
  "host": socket.gethostname(),
  "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "git_sha": "n/a",
}, indent=2))
PY

# Build the endpoint-passthrough args once. When --endpoint was unset, this
# array is empty and the runners fall back to model_cfg["endpoint"].
EP_ARGS=()
if [ -n "$ENDPOINT" ]; then
  EP_ARGS=(--endpoint "$ENDPOINT" --base-url "$EP_BASE_URL")
fi

run_quality() {
  echo "[run.sh] quality runner ($RUNNER) for $MODEL / $SUITE / thinking=$THINKING / endpoint=${ENDPOINT:-model-default}"
  uv run --project "$DIR" python "$DIR/runners/$QUALITY_SCRIPT" \
    --model "$MODEL" --suite "$SUITE" --thinking "$THINKING" \
    --out "$QUALITY_OUT" "${EP_ARGS[@]}"
}

run_speed() {
  echo "[run.sh] speed runner for $MODEL / $SUITE / endpoint=${ENDPOINT:-model-default}"
  uv run --project "$DIR" python "$DIR/runners/speed_serving.py" \
    --model "$MODEL" --suite "$SUITE" \
    --out "$OUT_DIR/speed.json" "${EP_ARGS[@]}"
}

case "$ONLY" in
  quality) run_quality;;
  speed)   run_speed;;
  "")      run_quality; run_speed;;
  *) echo "unknown --only: $ONLY (use 'quality' or 'speed')" >&2; exit 1;;
esac

echo "[run.sh] done. results in: $OUT_DIR"
ls -la "$OUT_DIR"
