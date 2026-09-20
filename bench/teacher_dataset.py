#!/usr/bin/env python3
"""Build the noul calibration corpus from public labeled datasets (HF caches).

Emits JSONL: {id, task, text, yes, no, label(1/0), gold}.
These are the items cloud Jev scores as a *teacher*; its probabilities become
calibration targets for quorum's local raw-logprob distribution.

Deterministic sampling (seeded) so teacher runs and local re-runs stay aligned.
"""
from __future__ import annotations

import argparse
import json
import random

from datasets import load_dataset

SEED = 20260919
PER_TASK = 60  # balanced: sampled per-class


def _bal(rows, n, key=lambda r: r["label"]):
    pos = [r for r in rows if key(r) == 1]
    neg = [r for r in rows if key(r) == 0]
    rng = random.Random(SEED)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    return pos[:k] + neg[:k]


def sst2():
    ds = load_dataset("stanfordnlp/sst2", "default")["validation"]
    out = []
    for i, r in enumerate(ds):
        out.append({"id": f"sst2-{i}", "task": "sst2-sentiment", "text": r["sentence"],
                    "yes": "the sentence expresses a positive sentiment",
                    "no": "the sentence does not express a positive sentiment",
                    "label": int(r["label"])})
    return _bal(out, PER_TASK)


def offensive():
    ds = load_dataset("cardiffnlp/tweet_eval", "offensive")["test"]
    out = []
    for i, r in enumerate(ds):
        out.append({"id": f"off-{i}", "task": "tweet-offensive", "text": r["text"],
                    "yes": "the tweet contains offensive or abusive language",
                    "no": "the tweet is not offensive",
                    "label": int(r["label"])})
    return _bal(out, PER_TASK)


def irony():
    ds = load_dataset("cardiffnlp/tweet_eval", "irony")["test"]
    out = []
    for i, r in enumerate(ds):
        out.append({"id": f"iron-{i}", "task": "tweet-irony", "text": r["text"],
                    "yes": "the tweet is ironic or sarcastic",
                    "no": "the tweet is literal, not ironic",
                    "label": int(r["label"])})
    return _bal(out, PER_TASK)


def spam():
    # sms_spam cache is gone; fall back to a public mirror if present, else skip.
    try:
        ds = load_dataset("giskard/sms-spam-en")["test"]
    except Exception:
        return []
    out = []
    for i, r in enumerate(ds):
        lab = r.get("label", r.get("v1"))
        out.append({"id": f"sms-{i}", "task": "sms-spam", "text": r.get("text", r.get("v2", "")),
                    "yes": "this SMS is spam / a marketing or scam message",
                    "no": "this SMS is a legitimate personal message",
                    "label": int(lab) if str(lab).lower() in ("spam", "1", "true") else 0})
    return _bal(out, PER_TASK)


def intent_pos_neg():
    """banking77 -> binary 'matches this intent' noul (positive/negative pairs)."""
    ds = load_dataset("mteb/banking77")["test"]
    rng = random.Random(SEED)
    by_lab = {}
    for r in ds:
        by_lab.setdefault(r["label"], []).append(r)
    labs = sorted(by_lab)
    out = []
    per = max(2, PER_TASK // (2 * 10))  # 10 intents, pos+neg each
    for lab in rng.sample(labs, 10):
        pos_rows = by_lab[lab]
        neg_labs = [l for l in labs if l != lab]
        for j in range(per):
            p = rng.choice(pos_rows)
            out.append({"id": f"int-{lab}-p{j}", "task": "intent-match", "text": p["text"],
                        "yes": f"the customer's issue is exactly about {p['label_text']}",
                        "no": "the issue is about something else",
                        "label": 1})
            nl = rng.choice(neg_labs)
            n = rng.choice(by_lab[nl])
            out.append({"id": f"int-{lab}-n{j}", "task": "intent-match", "text": n["text"],
                        "yes": f"the customer's issue is exactly about {p['label_text']}",
                        "no": "the issue is about something else",
                        "label": 0})
    return out


BUILDERS = {"sst2": sst2, "offensive": offensive, "irony": irony,
            "spam": spam, "intent": intent_pos_neg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="teacher_items.jsonl")
    ap.add_argument("--tasks", default=",".join(BUILDERS))
    a = ap.parse_args()
    n = 0
    with open(a.out, "w") as fh:
        for t in a.tasks.split(","):
            rows = BUILDERS[t]()
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += len(rows)
            print(f"{t}: {len(rows)}")
    print(f"total {n} -> {a.out}")


if __name__ == "__main__":
    main()
