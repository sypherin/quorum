#!/usr/bin/env python3
"""Job 2 data: distillation items for the quorum prob-head.

cloud-Jev probability is the regression target; gold is its binarisation.
Sources: bench/teacher_run.jsonl (240 items with cloud probs) +
corpus/auto-labeled.jsonl (2,680 cloud-labeled pairs). Deduped, capped.
"""
import json
import os

BENCH = os.path.dirname(os.path.abspath(__file__))
CORP = os.path.expanduser("~/judgment-model/corpus")
out, seen = [], set()

for line in open(f"{BENCH}/teacher_run.jsonl"):
    r = json.loads(line)
    if r.get("cloud") is None:
        continue
    key = (r["text"] + r["yes"])[:200]
    if key in seen:
        continue
    seen.add(key)
    out.append({"state": r["text"], "question": f"yes: {r['yes']} / no: {r['no']}",
                "cloud": float(r["cloud"]), "gold": 1 if r["cloud"] >= 0.5 else 0,
                "src": "teacher"})

for line in open(f"{CORP}/auto-labeled.jsonl"):
    r = json.loads(line)
    key = (r["state"] + r.get("question", ""))[:200]
    if key in seen:
        continue
    seen.add(key)
    out.append({"state": r["state"], "question": r.get("question", r["qid"]),
                "cloud": float(r["cloud"]), "gold": int(r["gold"]), "src": "auto"})

with open(f"{BENCH}/distill_items.jsonl", "w") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
pos = sum(1 for r in out if r["gold"] == 1)
print(f"{len(out)} distill items (pos {pos} / neg {len(out)-pos}) -> distill_items.jsonl")
