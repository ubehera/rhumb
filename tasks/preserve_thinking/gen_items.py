#!/usr/bin/env python3
"""Generate HARDER coupled + matched control items for the preserve_thinking probe.

Design (see why the 2026-05-30 run ceilinged at 100%): preserve_thinking only
matters when, on the final turn, reconstructing the needed intermediate from
turn-1's SHORT final answer (reasoning stripped, OFF arm) is error-prone. So
turn-1 here is a multi-step STATEFUL computation whose final integer does NOT
reveal a specific buried intermediate, and turn-2 asks for that intermediate:
  - chain      : a long running-total; turn-2 = the running value after step k
  - recur      : a custom recurrence f(n); turn-2 = an earlier f(m)
  - dualcount   : two interleaved counters A,B; turn-1=final B, turn-2=final A
Controls are two INDEPENDENT multi-step problems (turn-2 needs nothing from
turn-1) — same surface difficulty, no coupling.

All answer keys are computed in code (never hand-derived). Turn-1 answers are a
single integer (<=40 chars) so the intermediate lives only in the <think>.
Difficulty is varied (chain length, recurrence depth) so calibration can pick
items whose OFF accuracy lands in the discriminating band (~30-80%).
"""
import json, random, pathlib

rng = random.Random(20260530)
OUT = pathlib.Path(__file__).resolve().parent / "items_v2_candidates.jsonl"
items = []

def chain(cid, start, ops, k):
    v = start; run = [start]; desc = []
    for sym, n in ops:
        if sym == "+": v += n; desc.append(f"add {n}")
        elif sym == "-": v -= n; desc.append(f"subtract {n}")
        elif sym == "*": v *= n; desc.append(f"multiply by {n}")
        run.append(v)
    steps = "; ".join(f"({i+1}) {d}" for i, d in enumerate(desc))
    t1 = (f"Start with the number {start}. Apply these operations strictly in order: "
          f"{steps}. Reply with ONLY the final integer.")
    t2 = (f"Reply with ONLY the running value immediately AFTER step {k} "
          f"(the '{desc[k-1]}' step), i.e. before step {k+1} is applied.")
    return {"id": cid, "bucket": "coupled",
            "turns": [{"role": "user", "content": t1, "expected": str(run[-1])},
                      {"role": "user", "content": t2, "expected": str(run[k])}]}

def recur(cid, a, p, q, c, N, M):
    f = {1: a}
    for n in range(2, N + 1):
        f[n] = p * f[n - 1] + q * n + c
    qd = f" + {q}*n" if q else ""
    cd = f" + {c}" if c else ""
    t1 = (f"Define a sequence: f(1) = {a}, and for n > 1, f(n) = {p}*f(n-1){qd}{cd}. "
          f"Reply with ONLY the integer value of f({N}).")
    t2 = f"Reply with ONLY the integer value of f({M})."
    return {"id": cid, "bucket": "coupled",
            "turns": [{"role": "user", "content": t1, "expected": str(f[N])},
                      {"role": "user", "content": t2, "expected": str(f[M])}]}

def dualcount(cid, instrs):
    A = B = 0; lines = []
    for kind, n in instrs:
        if kind == "A+": A += n; lines.append(f"add {n} to A")
        elif kind == "A-": A -= n; lines.append(f"subtract {n} from A")
        elif kind == "B+A": B += A; lines.append("add the current value of A to B")
        elif kind == "B*": B *= n; lines.append(f"multiply B by {n}")
        elif kind == "B+": B += n; lines.append(f"add {n} to B")
    steps = "; ".join(f"({i+1}) {d}" for i, d in enumerate(lines))
    t1 = (f"Two counters start at A=0 and B=0. Apply strictly in order: {steps}. "
          f"Reply with ONLY the final integer value of B.")
    t2 = "Reply with ONLY the final integer value of A."
    return {"id": cid, "bucket": "coupled",
            "turns": [{"role": "user", "content": t1, "expected": str(B)},
                      {"role": "user", "content": t2, "expected": str(A)}]}

def control(cid, prob1, ans1, prob2, ans2):
    return {"id": cid, "bucket": "control",
            "turns": [{"role": "user", "content": prob1 + " Reply with ONLY the integer.", "expected": str(ans1)},
                      {"role": "user", "content": "Unrelated new problem: " + prob2 + " Reply with ONLY the integer.", "expected": str(ans2)}]}

# --- chains: vary length 6..11, k buried in the middle ---
SYMS = ["+", "-", "*"]
for i, L in enumerate([6, 7, 8, 8, 9, 9, 10, 11]):
    start = rng.randint(11, 40)
    ops = []
    for _ in range(L):
        s = rng.choice(SYMS)
        n = rng.randint(2, 5) if s == "*" else rng.randint(40, 600)
        ops.append((s, n))
    k = rng.randint(3, L - 2)  # buried, not first/last
    items.append(chain(f"chain-{i+1:02d}", start, ops, k))

# --- recurrences: vary depth N 5..9, ask an earlier M ---
for i, (N, M) in enumerate([(5, 3), (6, 3), (6, 4), (7, 4), (7, 5), (8, 4), (8, 5), (9, 5)]):
    a = rng.randint(2, 7); p = rng.choice([2, 3]); q = rng.choice([0, 1, 2, 3]); c = rng.choice([0, 0, 5, 7])
    items.append(recur(f"recur-{i+1:02d}", a, p, q, c, N, M))

# --- dual-counter: vary instruction count ---
for i, L in enumerate([6, 7, 8, 9, 10, 11]):
    instrs = []
    for _ in range(L):
        kind = rng.choice(["A+", "A+", "A-", "B+A", "B*", "B+"])
        n = rng.randint(2, 3) if kind == "B*" else rng.randint(3, 40)
        instrs.append((kind, n))
    if not any(k == "B+A" for k, _ in instrs):  # ensure A feeds B (real coupling)
        instrs[rng.randint(0, L - 1)] = ("B+A", 0)
    items.append(dualcount(f"dual-{i+1:02d}", instrs))

# --- controls: two independent multi-step problems ---
ctrl_specs = [
    ("Compute 6 * 7 + 5 * 8.", 82, "Compute the sum of the first 12 positive integers.", 78),
    ("Compute 144 / 12 * 3.", 36, "Find the LCM of 8 and 14.", 56),
    ("Compute 17 * 13.", 221, "Compute 9 factorial divided by 7 factorial.", 72),
    ("Compute (45 - 18) * 4.", 108, "How many ways to choose 3 from 7 (combinations)?", 35),
    ("Compute 2^8 + 2^5.", 288, "Compute the area of a triangle with base 22 and height 9.", 99),
    ("Compute 1000 - 7 * 49.", 657, "Compute 53 squared minus 47 squared.", 600),
    ("Compute 360 / 8 + 360 / 9.", 85, "Sum of interior angles of a hexagon in degrees.", 720),
    ("Compute 19 * 21.", 399, "Number of primes strictly below 30.", 10),
    ("Compute 7 * 8 * 9.", 504, "Compute 15% of 240.", 36),
    ("Compute 88 + 76 + 59 + 41.", 264, "Compute the GCD of 84 and 126.", 42),
    ("Compute 3^5 - 100.", 143, "Compute 256 / 16 + 19.", 35),
    ("Compute 14 * 15 / 2.", 105, "Number of edges of a cube plus faces of a cube.", 18),
]
for i, (p1, a1, p2, a2) in enumerate(ctrl_specs):
    items.append(control(f"control-{i+1:02d}", p1, a1, p2, a2))

with open(OUT, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")

nc = sum(1 for it in items if it["bucket"] == "coupled")
nk = sum(1 for it in items if it["bucket"] == "control")
print(f"wrote {len(items)} items ({nc} coupled, {nk} control) -> {OUT}")
print("\nsamples:")
for it in [items[0], items[8], items[14], items[20]]:
    print(f"\n[{it['id']} / {it['bucket']}]")
    for t in it["turns"]:
        print(f"  Q: {t['content'][:140]}")
        print(f"  A(key): {t['expected']}")
