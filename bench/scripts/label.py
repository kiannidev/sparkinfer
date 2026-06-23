#!/usr/bin/env python3
"""Eval-loop label = deterministic function of measurements (so validators converge).

  label.py <tps> <frontier_tps> <ceiling_tps> <top1> <kl> <commit>
           [<baseline_agree> <baseline_selfkl> <baseline_commit>]

Emits one line:  RESULT_JSON {...}

- Correctness gate first: top-1 token agreement >= 0.90 and KL <= 0.5 vs llama.cpp,
  else REJECT (score 0).
- Optional baseline gate (when baseline_agree provided): argmax agreement >= 0.99 and
  self-KL <= 0.01 vs the previous frontier build — catches silent output drift that
  still passes the llama.cpp reference.
- Significance gate: improvement must exceed ~2% of the frontier (a CI proxy), else label "none".
- Label = bucket of the **fraction of remaining headroom closed** (maturity-adaptive: a small
  absolute gain near the ceiling still maps to a high label). Thresholds are governance-tunable.
"""
import sys, json

tps      = float(sys.argv[1])   # measured median tok/s of the submission
frontier = float(sys.argv[2])   # current best verified tok/s (0 = none yet)
ceiling  = float(sys.argv[3])   # roofline / strong-reference cap (0 = scale by frontier)
top1     = float(sys.argv[4])   # token-match vs llama.cpp, 0..1
kl       = float(sys.argv[5])   # mean KL vs llama.cpp (nats)
commit   = sys.argv[6]

baseline_agree   = float(sys.argv[7]) if len(sys.argv) > 7 else None
baseline_selfkl  = float(sys.argv[8]) if len(sys.argv) > 8 else None
baseline_commit  = sys.argv[9] if len(sys.argv) > 9 else None

TOP1_BAR, KL_BAR = 0.90, 0.50
BASELINE_AGREE_BAR, BASELINE_SELFKL_BAR = 0.99, 0.01
SIG = 0.02                                              # significance: >2% over frontier
BUCKETS = [(0.25, "XL"), (0.10, "L"), (0.03, "M"), (0.01, "S"), (0.0, "XS")]

res = {"commit": commit, "tps": round(tps, 2), "top1": round(top1, 4),
       "kl": round(kl, 4), "frontier_tps": round(frontier, 2)}

if baseline_agree is not None:
    res["baseline_commit"] = baseline_commit
    res["baseline_top1"] = round(baseline_agree, 4)
    res["baseline_selfkl"] = round(baseline_selfkl, 6)

reasons = []
if top1 < TOP1_BAR or kl > KL_BAR:
    reasons.append(f"llama.cpp correctness (top1={top1}, kl={kl})")
if baseline_agree is not None and (baseline_agree < BASELINE_AGREE_BAR
                                   or baseline_selfkl > BASELINE_SELFKL_BAR):
    reasons.append(f"baseline drift (agree={baseline_agree}, selfkl={baseline_selfkl})")

if reasons:
    res.update(pass_=False, label="REJECT", reason="; ".join(reasons))
elif frontier <= 0:
    res.update(pass_=True, label="BASELINE", note="no frontier set; this submission becomes it")
else:
    delta = tps - frontier
    if delta <= SIG * frontier:
        res.update(pass_=True, label="none", delta_tps=round(delta, 2),
                   note="within significance gate — not a verified improvement")
    else:
        head = (ceiling - frontier) if ceiling > frontier else frontier   # remaining headroom
        f = delta / head
        label = next(l for thr, l in BUCKETS if f >= thr)
        res.update(pass_=True, label=label, delta_tps=round(delta, 2),
                   pct_over_frontier=round(100 * delta / frontier, 1), frac_headroom=round(f, 4))

# JSON keys can't be "pass" via kwarg; normalize
res["pass"] = res.pop("pass_", True)
print("RESULT_JSON " + json.dumps(res))
