#!/usr/bin/env python3
"""Triage the accumulated judgment corpus with cloud Jev — no manual review.

Pool (deduped by state):
  - ~/dev/quorum/log/judgments.jsonl   (full state + questions + local probs)
  - ~/judgment-model/gate-log.jsonl    (probs records: state_head + probs)
  - ~/judgment-model/g3-shadow.jsonl   (shadow verdicts, if richer)

For each unique state we ask cloud Jev the SAME questions the local gate asked
(falling back to the standard DECOMP set), then score usefulness:

  disagreement  |local P(yes) - cloud P(yes)|   -> local model is wrong here;
                                                  highest training value
  informativeness  distance of cloud prob from 0.5 -> confident teacher signal
  redundancy     near-duplicate states (token jaccard) -> keep one per cluster

Rank = disagreement * 0.6 + informativeness * 0.4, one representative per
near-duplicate cluster. Emits a review file sorted by value; a top slice is
auto-accepted (cloud label = gold) into corpus/auto-labeled.jsonl, the rest
waits in corpus/auto-candidates.jsonl only if you want to skim.

Cost: one cloud call per unique state (~1-2s each, batched 4 questions).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, os.path.expanduser("~/dev/quorum"))

API = "https://api.typesafe.ai/v1/systemone"
KEYFILE = os.path.expanduser("~/.config/typesafe/env")
_lock = threading.Lock()


def _key():
    for line in open(KEYFILE):
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no key")


STOP = set("the a an is are was were be been being this that these those it its to of and in on for with as at by from has have had not you your i we they he she".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cloud_ask(state, questions, retries=3):
    body = json.dumps({"model": "jev-latest", "state": state[:6000],
                       "questions": questions}).encode()
    for att in range(retries):
        try:
            req = urllib.request.Request(API, data=body, headers={
                "Authorization": "Bearer " + _key(), "content-type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=60).read())
            probs = {}
            for qid, a in out.get("answers", {}).items():
                if a.get("type") == "noul" and isinstance(a.get("noul"), (int, float)):
                    probs[qid] = float(a["noul"])
            return probs, out.get("usage", {})
        except Exception as e:
            if att == retries - 1:
                return None, str(e)[:100]
    return None, "retries"


def load_pool():
    """unique states -> {state, local_probs per qid, questions, callers, n_seen}"""
    pool = {}

    def add(state, probs, questions, caller):
        state = (state or "").strip()
        if len(state) < 40:
            return
        key = state[:400]
        e = pool.setdefault(key, {"state": state, "local": {}, "questions": questions or {},
                                  "callers": set(), "n": 0})
        e["n"] += 1
        e["callers"].add(caller or "?")
        for qid, p in (probs or {}).items():
            if isinstance(p, (int, float)) and qid not in e["local"]:
                e["local"][qid] = float(p)

    p1 = os.path.expanduser("~/dev/quorum/log/judgments.jsonl")
    for line in open(p1):
        try:
            r = json.loads(line)
        except Exception:
            continue
        probs = {}
        for qid, a in (r.get("answers") or {}).items():
            if isinstance(a, dict):
                pp = a.get("probabilities")
                if isinstance(pp, dict) and isinstance(pp.get("yes"), (int, float)):
                    probs[qid] = float(pp["yes"])
                elif isinstance(a.get("noul"), (int, float)):
                    probs[qid] = float(a["noul"])
        add(r.get("state"), probs, r.get("questions"), "quorum-log")

    p2 = os.path.expanduser("~/judgment-model/gate-log.jsonl")
    for line in open(p2):
        try:
            r = json.loads(line)
        except Exception:
            continue
        probs = {k: v for k, v in (r.get("probs") or {}).items()
                 if isinstance(v, (int, float))}
        add(r.get("state_head") or r.get("situation_head"), probs, None, r.get("caller"))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/judgment-model/corpus/auto-triage.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="cap cloud calls (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    pool = load_pool()
    print(f"{len(pool)} unique states after dedupe")

    # near-duplicate clustering (greedy, cheap): representative = most-seen
    items = sorted(pool.values(), key=lambda e: -e["n"])
    reps = []  # (tokset, rep_item)
    for e in items:
        t = toks(e["state"])
        merged = False
        for i, (rt, rep) in enumerate(reps):
            if jaccard(t, rt) > 0.7:
                rep["n"] += e["n"]
                rep["callers"] |= e["callers"]
                merged = True
                break
        if not merged:
            reps.append((t, e))
    uniq = [e for _, e in reps]
    print(f"{len(uniq)} after near-duplicate clustering (jaccard>0.7)")
    if a.limit:
        uniq = uniq[: a.limit]

    out_fh = open(a.out, "w")

    def one(e):
        qs = e["questions"] or {}
        noul_qs = {qid: q for qid, q in qs.items()
                   if isinstance(q, dict) and q.get("type") == "noul"}
        if not noul_qs:
            from quorum.gate_client import DECOMP_QUESTIONS
            noul_qs = DECOMP_QUESTIONS
        cp, usage = cloud_ask(e["state"], noul_qs)
        rec = {"state": e["state"][:1500], "callers": sorted(e["callers"]),
               "n_seen": e["n"], "cloud": cp, "local": e["local"],
               "questions": {k: v.get("instructions", "")[:100] for k, v in noul_qs.items()}}
        scores = []
        if cp:
            for qid, c in cp.items():
                lp = e["local"].get(qid)
                inf = abs(c - 0.5) * 2
                dis = abs(lp - c) if lp is not None else 0.35  # unseen local = medium
                scores.append(0.6 * dis + 0.4 * inf)
        rec["value"] = round(max(scores), 4) if scores else None
        with _lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
        return rec

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(one, uniq))
    recs = [r for r in recs if r.get("value") is not None]
    recs.sort(key=lambda r: -r["value"])
    with open(a.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    hi = sum(1 for r in recs if r["value"] >= 0.35)
    print(f"wrote {len(recs)} scored records -> {a.out} ({hi} with value>=0.35)")


if __name__ == "__main__":
    main()
