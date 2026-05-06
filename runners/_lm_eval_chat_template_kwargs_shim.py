"""Wrapper around lm-eval that forwards `chat_template_kwargs` into the
chat-completions request payload.

WHY THIS EXISTS

  lm-eval (as of git main 243c5463 / 0.4.12.dev0) silently drops
  `chat_template_kwargs` from `--model_args` for `local-chat-completions`:

    1. Client-side: api_models.apply_chat_template() calls
       tokenizer.apply_chat_template(chat_history, ...) WITHOUT forwarding
       any extra kwargs (api_models.py:342).
    2. Server-side: openai_completions.LocalChatCompletion._create_payload()
       builds the request body with `messages/model/max_tokens/temperature/
       stop/seed/**gen_kwargs` and never includes chat_template_kwargs
       (openai_completions.py:200).

  Both paths drop it, so flags like `enable_thinking` never reach Qwen 3.6's
  Jinja chat template. Verified 2026-05-05 with curl probe + grep of the
  installed wheel.

WHAT THIS DOES

  Patches `LocalChatCompletion._create_payload` to inject
  chat_template_kwargs (read from $LM_EVAL_CHAT_TEMPLATE_KWARGS as JSON) into
  the payload before sending. Then dispatches to the normal lm-eval CLI.

  Drop the wrapper when lm-eval ships proper chat_template_kwargs forwarding
  (track upstream EleutherAI/lm-evaluation-harness for a fix).
"""
from __future__ import annotations

import json
import os
import sys

_CTK_ENV = "LM_EVAL_CHAT_TEMPLATE_KWARGS"
_STOP_ENV = "LM_EVAL_STOP_OVERRIDE"


def _load_json_env(name: str):
    raw = os.environ.get(name, "")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[shim] bad {name} JSON: {e}", file=sys.stderr)
        return None


def _apply_patch() -> None:
    chat_template_kwargs = _load_json_env(_CTK_ENV)
    stop_override = _load_json_env(_STOP_ENV)
    if not chat_template_kwargs and not stop_override:
        return

    import lm_eval.models.openai_completions as oai

    _original = oai.LocalChatCompletion._create_payload

    _first_dumped = [False]

    def _patched(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
        payload = _original(
            self,
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if stop_override is not None:
            payload["stop"] = list(stop_override)
        # Dump first payload so we can compare lm-eval's request shape vs hand-rolled probes.
        # Only first request — every subsequent is the same shape with a different question.
        if not _first_dumped[0]:
            try:
                dump_path = "/tmp/lm_eval_first_request.json"
                with open(dump_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"[shim] dumped first request payload to {dump_path}", file=sys.stderr)
                _first_dumped[0] = True
            except Exception as e:
                print(f"[shim] could not dump payload: {e}", file=sys.stderr)
        return payload

    oai.LocalChatCompletion._create_payload = _patched
    msg_parts = []
    if chat_template_kwargs:
        msg_parts.append(f"chat_template_kwargs={chat_template_kwargs}")
    if stop_override is not None:
        msg_parts.append(f"stop_override={stop_override}")
    print(
        f"[shim] patched LocalChatCompletion._create_payload: " + ", ".join(msg_parts),
        file=sys.stderr,
    )


def main() -> int:
    _apply_patch()
    from lm_eval.__main__ import cli_evaluate
    return cli_evaluate() or 0


if __name__ == "__main__":
    sys.exit(main())
