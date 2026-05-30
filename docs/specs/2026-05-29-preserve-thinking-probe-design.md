# preserve_thinking multi-turn probe — design

**Date:** 2026-05-29
**Status:** approved design, pre-implementation
**Owner:** umank
**Question it answers:** Does Qwen3.6's `preserve_thinking=true` (keep prior-turn `<think>` reasoning in multi-turn context) produce *better answers* than the default (strip prior reasoning) — and at what token cost — on our SGLang TP=2 stack?

## 1. Why this needs a new instrument

`preserve_thinking` only changes the prompt when the conversation history contains assistant turns the **model itself generated with `<think>` blocks**. Its sole effect (Qwen3.6 `chat_template.jinja` line 100) is whether those prior `<think>` blocks survive into the next turn's prompt or are stripped.

Every existing harness feeds **canned, single-generation items**, so the flag is inert:
- **rhumb / lm-eval:** each item is `[canned few-shot exemplars] + [one target question] -> one generation`. Few-shot "assistant" turns are gold answers with no `<think>`; there is no second model turn building on a first model turn's reasoning. (`suites/reasoning.yaml` confirms `fewshot_as_multiturn` with canned turns.)
- **healthbench:** given a conversation, produce one final response, grade vs rubric. Prior turns are supplied, not model-generated-with-`<think>`.

Running preserve on/off through either yields **identical scores** — a silent null ("looks like no effect," actually "never tested"). Measuring it requires a **live multi-turn loop** where the model generates real `<think>` in turn 1 and we decide whether it survives into turn 2.

## 2. Core mechanism (single-variable A/B)

For each item, run the conversation turn-by-turn against the endpoint. After each assistant turn, reconstruct its history entry as:

```
content = "<think>\n{reasoning_content}\n</think>\n\n{visible_content}"
```

(SGLang with `--reasoning-parser qwen3` returns `reasoning_content` and `content` separately; we re-embed.) On the next turn we send the **identical messages** under two conditions, flipping only one body field:

- **OFF:** `chat_template_kwargs = {"enable_thinking": true, "preserve_thinking": false}`
- **ON:**  `chat_template_kwargs = {"enable_thinking": true, "preserve_thinking": true}`

The Qwen3.6 template does the rest: with the flag false/absent it drops prior-turn `<think>` (verified: line 100's first clause false, prior assistant turn has index ≤ `last_query_index`, so the `else` branch strips); with it true it retains. **The flag is the only variable in the pipeline.** `enable_thinking` stays true throughout — preserve is meaningless with thinking off.

## 3. Coupling design (what makes the test discriminating)

Turn 1 instructs a **terse final answer**, forcing the method and intermediate values to live *only* inside `<think>`, not in the visible answer.

- **Coupled bucket (treatment):** turn 2 depends on those hidden intermediates ("using the discriminant you computed, now solve ..."). OFF -> model must re-derive (may diverge/err); ON -> it can reuse. Where preserve *should* help if it ever does.
- **Control bucket:** turn 2 is independent of turn 1 (needs nothing carried over). ON just adds tokens; expected delta ~0.

The coupled-vs-control contrast isolates "preserve helps *specifically when reasoning is carried*" from "preserve helps/hurts in general." Both buckets have unique checkable answers.

## 4. Dataset

- ~24 hand-authored items: **12 coupled + 12 control**, 2 turns each, with ~4 items at 3 turns to test accumulation.
- Domains: multi-step arithmetic / algebra / small combinatorics — each turn has a single unique, mechanically-checkable answer.
- Hand-built (small, fully controllable, no licensing). Stored as a JSON/YAML file in the suite dir.
- Item schema (per turn): `user` prompt, `expected` answer, optional `answer_regex`/`type` for extraction; item-level `bucket: coupled|control`.
- Authoring rule: a coupled item is only valid if turn-2 cannot be answered from turn-1's *visible* answer alone — verified by construction (turn-2 references a quantity that the terse turn-1 answer does not contain).

## 5. Driver (`runner: multiturn_chat` in rhumb)

New runner `runners/quality_multiturn.py`, selected via the suite YAML `runner:` field (run.sh gains a `multiturn_chat` case -> `quality_multiturn.json`). Flow per item × condition × sample:

1. Send turn 1; capture `reasoning_content`, `content`, `usage`.
2. Append reconstructed `<think>`-inclusive assistant message to history.
3. Send next user turn with `chat_template_kwargs` for the active condition; repeat.
4. Grade the **final** turn by exact-match (after extraction); record per-turn `prompt_tokens` / `completion_tokens`.

Reuses rhumb's existing plumbing: `endpoints.yaml` (target the SGLang TP=2 Qwen3.6 endpoint — add an entry if absent), `models.yaml` model entry, `results/<model>/<date>/` layout, `meta.json` run descriptor, and the per-request `chat_template_kwargs` pattern (the driver builds payloads directly, so it does not need the lm-eval monkeypatch shim).

## 6. Decoding & confounds

- Decoding: `temperature 0.6, top_p 0.95, top_k 20` (Qwen thinking-mode recommendation; reflects real serving).
- **Paired seeds:** for sample index `s`, the seed is shared across OFF and ON for the same item — a paired comparison so the only difference is the flag.
- **N = 5** samples per item per condition -> variance band; report mean and CI.
- Endpoint pinned to the SGLang TP=2 Qwen3.6 service; one model, both conditions, same session window.
- Token cost read from API `usage`, not estimated.

## 7. Metrics & outputs

Per `condition × bucket`:
- Exact-match accuracy over `items × samples`, with a bootstrap/normal CI.
- Mean `prompt_tokens` and total tokens (the cost side; ON should be higher in coupled multi-turn).

Headline:
- **Accuracy delta (ON − OFF)** in coupled vs control.
- **Token overhead** of ON.

Decision rule: preserve_thinking is "worth it" only if the **coupled** accuracy gain clearly exceeds the **control** gain (≈0) *and* justifies the token cost. The runner must also surface the case where ON **hurts** (off-distribution degradation) — a real candidate outcome, not an error.

Results JSON shape: `{condition: {bucket: {accuracy, ci, n, mean_prompt_tokens, mean_total_tokens}}}` plus per-item rows for `--log_samples`-style inspection.

## 8. Non-goals (YAGNI)

- No LLM judge (deterministic grading only — fits rhumb's epistemology, avoids self-preference bias).
- No new general multi-turn framework — just enough driver to run this probe.
- No cross-model sweep in v1 (single model: the Qwen3.6 daily driver). Generalize later only if the effect is real.
- Not wired into the default suites; opt-in `--suite preserve_thinking`.

## 9. Risks & mitigations

- **Terse-answer leakage:** if turn 1 still puts the method in visible content, OFF wouldn't lose it and the test goes inert. Mitigation: prompt for answer-only; assert `reasoning_content` non-empty and visible `content` matches the answer-only shape; drop/flag items that violate this.
- **Extraction errors:** a wrong grade from bad answer parsing looks like a real delta. Mitigation: strict per-item extractor + a quick manual spot-check of a few graded transcripts before trusting aggregates (cross-check-benchmarks discipline).
- **Too-easy items:** if both conditions hit ~100%, there's no headroom to show a delta. Mitigation: calibrate difficulty so OFF baseline sits mid-range; bump difficulty if turn-1 accuracy is near-ceiling.
- **Sampling noise:** addressed by paired seeds + N=5 + CIs.
