#!/usr/bin/env python3
"""Open-ended COUPLED + CONTROL items for the judge-graded preserve_thinking probe.

Design rationale: the exact-match probe ceilinged because deterministic
intermediates are recomputable from the visible turn-1 prompt. Here turn-1 is an
open-ended task answered with a TERSE conclusion, so the decision-relevant
content (assumptions made, alternatives weighed, the *why*) lives only in the
<think>. turn-2 then needs THAT specific reasoning. preserve=ON carries it;
preserve=OFF leaves only the terse conclusion, so the model must re-derive and
may diverge from its own turn-1 commitment -> the judge (rewarding coherence
with turn-1) can detect the gap. Controls pair a terse turn-1 with an
INDEPENDENT turn-2, so preserve should NOT help (null baseline).

Each item: {id, bucket, turns:[{role,content}...], judge_rubric}. turn-1 prompts
force a terse visible answer ("Reply with ONLY ...") so the reasoning stays in
the think (and the leakage guard stays satisfied).
"""
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parent / "items_judge.jsonl"

def item(id_, bucket, t1, t2, rubric):
    return {"id": id_, "bucket": bucket,
            "turns": [{"role": "user", "content": t1}, {"role": "user", "content": t2}],
            "judge_rubric": rubric}

items = []

# ---------------- COUPLED (turn-2 needs turn-1's reasoning/choices) ----------------

items.append(item("est-tuners", "coupled",
    "Estimate how many piano tuners work in Chicago. Reply with ONLY the final integer.",
    "Which single assumption in your estimate contributes the MOST uncertainty, and if that "
    "assumption were off by a factor of 2, what would your revised integer estimate be? "
    "Give the assumption and the revised integer.",
    "A strong answer names a specific assumption that plausibly underlies the turn-1 estimate "
    "(pianos per household/capita, tunings per piano per year, jobs per tuner per year), explains "
    "why it dominates uncertainty, and gives a revised integer that is numerically consistent with "
    "scaling THAT assumption by 2x relative to the turn-1 number. Reward tight coherence with the "
    "turn-1 estimate; penalize generic answers or a revision inconsistent with the original number."))

items.append(item("design-datastore", "coupled",
    "A startup needs a datastore for a write-heavy event-logging system: ~1,000,000 events/sec, "
    "mostly append-only, occasional analytical queries, run by a 3-person team. Choose ONE: "
    "PostgreSQL, Cassandra, or Kafka+S3. Reply with ONLY the chosen option.",
    "What is the single biggest OPERATIONAL risk of your choice for THIS system specifically "
    "(this workload, this team size), and how would you mitigate it within a small team's capacity? "
    "Be concrete to the workload described.",
    "Reward a risk specific to BOTH the chosen system AND this workload (1M ev/s append-heavy, "
    "3-person team) - e.g. Cassandra: compaction/tombstone and repair burden on a tiny team; "
    "Kafka+S3: analytical-query latency / no ad-hoc SQL; PostgreSQL: write throughput ceiling / "
    "vacuum pressure. The mitigation must be realistic for 3 people. Penalize generic risks that "
    "ignore the workload or contradict the turn-1 choice."))

items.append(item("debug-cache", "coupled",
    "This function intermittently returns stale data under concurrency:\n```python\n"
    "cache = {}\ndef get_user(uid):\n    if uid in cache:\n        return cache[uid]\n"
    "    u = db.fetch(uid)\n    cache[uid] = u\n    return u\n```\n"
    "Identify the SINGLE most important root cause of the staleness. Reply with ONLY a "
    "one-sentence root cause.",
    "Rewrite the function to fix the root cause you identified, and name the two specific failure "
    "modes your rewrite now prevents. Keep it Python.",
    "Reward a rewrite that addresses the SAME root cause stated in turn-1 (the cache has no "
    "invalidation/TTL so entries never reflect later DB writes, and/or concurrent population races) "
    "by adding TTL/invalidation and/or locking; the two named failure modes must match the rewrite "
    "and the turn-1 diagnosis. Penalize fixes inconsistent with the turn-1 root cause or that "
    "introduce new races."))

items.append(item("incident-bottleneck", "coupled",
    "Incident summary: 'Checkout latency spiked 10x for 40 minutes during a flash sale. DB CPU was "
    "40%, app-server CPU 30%, but the database connection pool was saturated the entire time. "
    "Autoscaling added app servers, which did not help.' Identify the single most likely bottleneck. "
    "Reply with ONLY a short phrase naming it.",
    "Given that bottleneck, explain WHY adding app servers failed to help, and state the one specific "
    "change that would actually resolve it. Be concrete.",
    "Reward identifying connection-pool / DB-connection saturation as the bottleneck; turn-2 must "
    "explain that adding app servers worsens it (each server opens more connections, increasing "
    "contention on a fixed pool) and propose a concrete fix (right-size the pool, add pgbouncer / a "
    "connection proxy, cap per-server connections). Reward coherence with the turn-1 bottleneck; "
    "penalize answers blaming DB CPU or app CPU (contradicted by the data) or generic advice."))

items.append(item("est-datacenter", "coupled",
    "Estimate the total annual electricity consumption, in kWh, of all data centers in the United "
    "States. Reply with ONLY the final number (kWh).",
    "Rank the THREE biggest drivers of your estimate from most to least impactful on the final number, "
    "and state which ONE you are least confident about. Be specific to how you got your number.",
    "Reward three drivers that actually compose the turn-1 estimate (total IT load in GW or number of "
    "data centers, PUE/overhead factor, hours per year, utilization), ranked sensibly by leverage on "
    "the result, with a coherent least-confident pick. Penalize drivers that don't connect to the "
    "stated number or 'it depends' non-answers."))

items.append(item("plan-migration", "coupled",
    "Outline a minimal 4-step plan to migrate a monolith to microservices with no downtime. "
    "Reply with ONLY the four step titles, numbered 1-4.",
    "For STEP 3 of your plan, list its hard prerequisites from the earlier steps and the single "
    "biggest way step 3 can fail. Be specific to YOUR plan, not generic migration advice.",
    "Reward prerequisites for step 3 that correctly depend on the specific content the assistant gave "
    "steps 1-2, and a failure mode specific to what step 3 actually is in this plan. Reward coherence "
    "with the turn-1 plan; penalize generic migration advice not tied to the assistant's own steps."))

items.append(item("design-loyalty", "coupled",
    "Design a loyalty program for a small independent coffee shop: one location, ~200 regulars, thin "
    "margins, NO budget for an app or new tech. Describe the core mechanic in ONE sentence. Reply with "
    "ONLY that one sentence.",
    "What is the main way your loyalty mechanic could LOSE the shop money, and what single rule would "
    "you add to prevent it - without any app or new tech? Be specific to your mechanic.",
    "Reward a money-loss analysis specific to the turn-1 mechanic (e.g. punch-card breakage math, "
    "discount stacking, reward cost vs thin margin) and a concrete no-tech rule that plugs exactly "
    "that hole. Reward coherence with the stated mechanic; penalize generic loyalty advice or any rule "
    "requiring an app/tech the constraints forbid."))

items.append(item("spec-search", "coupled",
    "A product manager asks you to build 'a fast search feature' with no other detail. Identify the "
    "single most important ambiguity you must resolve first, and state the specific assumption you will "
    "proceed with. Reply with ONLY your chosen assumption, in one sentence.",
    "Given that assumption, what is the simplest architecture that satisfies it, and what would you "
    "have to change if the OPPOSITE assumption were true? Be concrete.",
    "Reward an architecture that follows directly from the SPECIFIC assumption chosen in turn-1 (e.g. "
    "if the assumption was 'small static corpus' -> in-memory index; if 'huge corpus, typo-tolerant' "
    "-> a search engine like Elasticsearch) and a sensible contrast under the opposite assumption. "
    "Reward coherence with the turn-1 assumption; penalize architectures unrelated to that assumption "
    "or generic search-system descriptions."))

# ---------------- CONTROL (turn-2 independent of turn-1: null baseline) ----------------

items.append(item("ctrl-coffee-tcp", "control",
    "Estimate how many cups of coffee are sold daily in Seattle. Reply with ONLY the final integer.",
    "Separate, unrelated question: explain the difference between TCP and UDP in two sentences.",
    "Reward a correct, clear two-sentence explanation of TCP vs UDP (TCP: connection-oriented, "
    "reliable, ordered; UDP: connectionless, no delivery/order guarantee, lower latency). This is "
    "INDEPENDENT of turn-1; judge only the explanation's correctness and clarity, ignore turn-1."))

items.append(item("ctrl-sort-closure", "control",
    "Which sorting algorithm is best for nearly-sorted data: insertion sort, quicksort, or heapsort? "
    "Reply with ONLY the choice.",
    "Separate, unrelated question: what is a closure in programming? Answer in two sentences.",
    "Reward a correct, concise definition of a closure (a function plus the captured variables from "
    "its enclosing lexical scope, which it retains after that scope returns). INDEPENDENT of turn-1; "
    "judge only the definition, ignore turn-1."))

items.append(item("ctrl-load-index", "control",
    "Name the most likely cause when a Linux server shows a high load average but low CPU utilization. "
    "Reply with ONLY a short phrase.",
    "Separate, unrelated question: state one upside and one downside of adding a database index.",
    "Reward correctly stating an index speeds up reads/lookups (upside) at the cost of slower writes "
    "and extra storage (downside). INDEPENDENT of turn-1; judge only the index trade-off, ignore "
    "turn-1."))

items.append(item("ctrl-novel-float", "control",
    "Estimate the number of words in a typical 300-page novel. Reply with ONLY the integer.",
    "Separate, unrelated question: in one short paragraph, explain why 0.1 + 0.2 does not exactly "
    "equal 0.3 in most programming languages.",
    "Reward a correct explanation (0.1, 0.2, 0.3 are not exactly representable in binary floating "
    "point; the sum carries rounding error, so it differs from the stored value of 0.3). INDEPENDENT "
    "of turn-1; judge only the floating-point explanation, ignore turn-1."))

with open(OUT, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")

nc = sum(1 for it in items if it["bucket"] == "coupled")
nk = sum(1 for it in items if it["bucket"] == "control")
print(f"wrote {len(items)} items ({nc} coupled, {nk} control) -> {OUT}")
for it in items:
    t1 = it["turns"][0]["content"].replace("\n", " ")
    print(f"  [{it['id']:20} {it['bucket']:8}] t1.len={len(it['turns'][0]['content'])}  {t1[:70]}")
