#!/usr/bin/env bash
# Reasoning-suite sweep across multiple models with automatic vLLM model swap.
#
# For each model in the sweep:
#   1. If vLLM isn't already serving the model's HF id, edit vllm.env, restart
#      vllm.service, and wait for /v1/models to confirm the new model is up.
#   2. Run the reasoning suite (gsm8k, thinking on, chat-completions runner)
#      synchronously via run.sh.
#   3. Capture status + wall time.
#
# Continues past failures (one bad model doesn't block the others).
#
# Usage:
#   ./scripts/run_reasoning_sweep.sh                                # default: qwen3-32b-awq qwen-32b-awq
#   ./scripts/run_reasoning_sweep.sh --models "qwen3.6-27b-awq qwen3-32b-awq"
#   ./scripts/run_reasoning_sweep.sh --wait-for-pid 825993           # wait for that pid before starting
#   ./scripts/run_reasoning_sweep.sh --detach                        # survive parent exit (setsid+nohup)
#
# Combined "fire and forget after the in-flight Qwen 3.6 run":
#   ./scripts/run_reasoning_sweep.sh --wait-for-pid <pid> --detach
#
# Requires:
#   - NOPASSWD sudo for `systemctl restart vllm.service` (verified at start).
#   - models named below must exist in models.yaml.

set -euo pipefail

RHUMB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ENV="$HOME/projects/vllm-server/vllm.env"
VLLM_HEALTH_URL="http://localhost:8000/v1/models"
VLLM_HEALTH_TIMEOUT=480           # 8 min — covers cold model load + warmup
SUMMARY_DIR="$RHUMB_DIR/results/_sweep_logs"

# HF id mapping — keep in sync with models.yaml. Encoded here so the script
# doesn't need to parse YAML for what it's about to write to vllm.env anyway.
declare -A MODEL_HF_ID=(
  [qwen-32b-awq]="Qwen/Qwen2.5-32B-Instruct-AWQ"
  [qwen3-32b-awq]="Qwen/Qwen3-32B-AWQ"
  [qwen3.6-27b-awq]="cyankiwi/Qwen3.6-27B-AWQ-INT4"
)

DETACH="0"
WAIT_FOR_PID=""
MODELS_RAW=""

ORIG_ARGS=("$@")

while [ $# -gt 0 ]; do
  case "$1" in
    --models)        MODELS_RAW="$2"; shift 2;;
    --wait-for-pid)  WAIT_FOR_PID="$2"; shift 2;;
    --detach)        DETACH="1"; shift;;
    -h|--help)
      sed -n '1,/^set -euo/p' "$0" | sed 's/^# \?//' | head -n -1
      exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [ -z "$MODELS_RAW" ]; then
  MODELS=(qwen3-32b-awq qwen-32b-awq)        # current default: skip 3.6 (already running)
else
  read -r -a MODELS <<<"$MODELS_RAW"
fi

mkdir -p "$SUMMARY_DIR"
SWEEP_TS="$(date -u +%Y%m%dT%H%M%SZ)"
SWEEP_LOG="$SUMMARY_DIR/reasoning_${SWEEP_TS}.log"
SWEEP_PID_FILE="$SUMMARY_DIR/reasoning_${SWEEP_TS}.pid"

# ----- Detach: re-exec under setsid+nohup so the parent shell can exit -----
if [ "$DETACH" = "1" ] && [ -z "${LOCALBENCH_SWEEP_DETACHED:-}" ]; then
  echo "[sweep] detaching; log -> $SWEEP_LOG"
  LOCALBENCH_SWEEP_DETACHED=1 setsid nohup "$0" "${ORIG_ARGS[@]}" >"$SWEEP_LOG" 2>&1 </dev/null &
  CHILD_PID=$!
  echo "$CHILD_PID" > "$SWEEP_PID_FILE"
  echo "[sweep] pid:    $CHILD_PID"
  echo "[sweep] tail:   tail -f $SWEEP_LOG"
  echo "[sweep] status: ps -p $CHILD_PID -o pid,stat,etime,cmd"
  echo "[sweep] kill:   kill \$(cat $SWEEP_PID_FILE)"
  exit 0
fi

# ----- Precondition: NOPASSWD sudo for systemctl restart vllm -----
if ! sudo -n true 2>/dev/null; then
  echo "[sweep] ERROR: sudo -n requires a password. Either configure NOPASSWD" >&2
  echo "        for \`systemctl restart vllm.service\` or run with cached creds" >&2
  echo "        (sudo -v) before launching this sweep." >&2
  exit 2
fi

# ----- Source vllm.env so we can authenticate to /v1/models -----
if [ ! -f "$VLLM_ENV" ]; then
  echo "[sweep] ERROR: $VLLM_ENV not found" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "$VLLM_ENV"
set +a

# ----- Helpers -----
log() { echo "[sweep $(date -u +%H:%M:%S)] $*"; }

get_current_vllm_model() {
  curl -sS --max-time 5 -H "Authorization: Bearer ${VLLM_API_KEY:-}" "$VLLM_HEALTH_URL" 2>/dev/null \
    | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    print(d["data"][0]["id"])
except Exception:
    print("")' 2>/dev/null
}

archive_existing_meta() {
  # Before run.sh writes a fresh meta.json, rename any existing one out of
  # the way using its `suite` field as suffix. This preserves the context of
  # prior runs (which suite, when, on what host) that would otherwise be
  # silently overwritten.
  local out_dir="$1"
  local meta="$out_dir/meta.json"
  [ -f "$meta" ] || return 0
  local prior_suite
  prior_suite="$(python3 -c "import json,sys
try: print(json.load(open('$meta')).get('suite','unknown'))
except Exception: print('unknown')" 2>/dev/null)"
  local archive="$out_dir/meta.${prior_suite}.json"
  if [ -e "$archive" ]; then
    archive="$out_dir/meta.${prior_suite}.replaced-by-sweep-${SWEEP_TS}.json"
  fi
  mv "$meta" "$archive"
  log "  archived prior meta.json -> $(basename "$archive")"
}

swap_vllm_model() {
  local hf_id="$1"
  log "swapping vLLM model -> $hf_id"
  sed -i "s|^VLLM_MODEL=.*|VLLM_MODEL=$hf_id|" "$VLLM_ENV"
  sudo -n systemctl restart vllm.service
  log "  waiting for vLLM to become healthy with $hf_id..."
  local deadline=$(( $(date +%s) + VLLM_HEALTH_TIMEOUT ))
  while true; do
    local current
    current="$(get_current_vllm_model || true)"
    if [ "$current" = "$hf_id" ]; then
      log "  ready (served model: $current)"
      return 0
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      log "  ERROR: vLLM did not become healthy within ${VLLM_HEALTH_TIMEOUT}s (last seen: '${current:-(none)}')"
      return 1
    fi
    sleep 5
  done
}

# ----- Optional: wait for an in-flight pid (e.g. the Qwen 3.6 reasoning run) -----
if [ -n "$WAIT_FOR_PID" ]; then
  log "waiting for pid $WAIT_FOR_PID to finish before starting the sweep..."
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 30
    log "  pid $WAIT_FOR_PID still alive"
  done
  log "pid $WAIT_FOR_PID finished, proceeding"
fi

# ----- Sweep loop -----
log "sweep starting; models=(${MODELS[*]})"
declare -A RESULTS=()

for model_name in "${MODELS[@]}"; do
  hf_id="${MODEL_HF_ID[$model_name]:-}"
  if [ -z "$hf_id" ]; then
    log "unknown model_name '$model_name' — no MODEL_HF_ID mapping; skipping"
    RESULTS[$model_name]="skipped (unknown model)"
    continue
  fi

  current="$(get_current_vllm_model || true)"
  if [ "$current" = "$hf_id" ]; then
    log "vLLM already serving $hf_id; no swap needed"
  else
    if ! swap_vllm_model "$hf_id"; then
      RESULTS[$model_name]="FAILED (vllm swap)"
      continue
    fi
  fi

  log "running reasoning suite for $model_name (thinking on, chat_lm_eval)"
  DATE_DIR="$RHUMB_DIR/results/$model_name/$(date +%Y-%m-%d)"
  mkdir -p "$DATE_DIR"
  archive_existing_meta "$DATE_DIR"

  t0=$(date +%s)
  if "$RHUMB_DIR/run.sh" --model "$model_name" --suite reasoning --thinking on --only quality; then
    wall=$(( $(date +%s) - t0 ))
    RESULTS[$model_name]="ok (${wall}s)"
  else
    wall=$(( $(date +%s) - t0 ))
    RESULTS[$model_name]="FAILED run.sh (${wall}s)"
  fi
done

# ----- Summary -----
echo
log "=== reasoning sweep summary ==="
for model_name in "${MODELS[@]}"; do
  log "  $model_name: ${RESULTS[$model_name]:-(skipped)}"
done
log "log archived at: $SWEEP_LOG"
