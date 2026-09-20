#!/usr/bin/env python3
"""Build the judgment-gate training sets from all three corpus sources.

Outputs (in bench/):
  kto_train.jsonl   — preference data for KTO: {prompt, completion, label}
                      label=True  -> completion is the desired behaviour
                      label=False -> completion is the undesired behaviour
  sft_train.jsonl   — messages format, same recipe as split.py to_messages()
  sft_eval.jsonl    — the FROZEN held-out set (corpus/heldout.json), untouched

Sources:
  corpus/approved.jsonl            647 Zach-reviewed pairs  -> SFT + KTO(pos/neg)
  corpus/auto-labeled.jsonl        2,680 cloud-Jev-labeled binary probes
      gold=1 -> KTO positive  ("VERDICT: OK")
      gold=0 -> KTO negative  (local said OK, cloud says not)
  data/train.jsonl                 current SFT set (570) — kept identical for
                                   the SFT arm so v11 stays comparable to v10.

v12 dataset fixes (v11 KTO scored 17/63 under the old broken scorer; the true
baseline is 35/63, so part of that was a scoring artifact — but these dataset
problems are real and were the planned v12 changes):
  1. auto rows had a DOUBLE phase prefix — state already begins with
     "[phase: gate] ..." and the builder prepended another "[phase: gate] ".
  2. gold=0 positives used a random CANON tag (the model's own tag, not the
     cloud's judgment — cloud Jev only gives a yes/no probability, so the tag
     was invented) with a truncated state sentence as the "correction".
     Now: tag = the model's own tag (so the pair isolates OK-vs-flag as the
     only variable) and the correction is generic rather than state mush.
  3. gold=1 negatives used the question text as the correction hint, which
     reads as a question, not an imperative. Now a generic imperative.
Balance is deliberately ~50/50 pos/neg (per-class weights can't fix a badly
skewed pair; we subsample the majority side with a fixed seed).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.expanduser("~/judgment-model"))
from split import SYS, CANON, user_key, verdict_line, split  # noqa: E402

ROOT = os.path.expanduser("~/judgment-model")
BENCH = os.path.dirname(os.path.abspath(__file__))
TAGS = [t for t in CANON.split(", ")]
rng = random.Random(20260919)

WRONG_TPL = "VERDICT: {tag} | {hint}"

# Generic imperatives for the auto-labeled pairs: cloud Jev gives no correction
# text, so keep the hint plausible-but-neutral instead of pasting state mush.
GENERIC_HINTS = [
    "verify before you call it done",
    "check the flagged aspect first",
    "don't ship unverified",
    "confirm against the source",
    "retest the actual path",
]
MODEL_TAG_RE = re.compile(r"VERDICT:\s*([A-Z][A-Z-]*)")


def approved_records():
    return [json.loads(l) for l in open(f"{ROOT}/corpus/approved.jsonl") if l.strip()]


def sft_sets():
    corpus = approved_records()
    frozen = json.load(open(f"{ROOT}/corpus/heldout.json"))
    train, held, missing = split(corpus, frozen["keys"])
    if missing:
        raise SystemExit(f"frozen heldout keys missing from corpus: {missing[:3]}")
    return [to_messages(r) for r in train], [to_messages(r) for r in held]


def to_messages(rec):
    return {"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user_key(rec)},
        {"role": "assistant", "content": verdict_line(rec)},
    ]}


def prompt_of(rec):
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": user_key(rec)}]


def kto_from_approved():
    rows = []
    for rec in approved_records():
        p = prompt_of(rec)
        good = verdict_line(rec)
        rows.append({"prompt": p, "completion": good, "label": True})
        if rec["principle"] != "OK":
            bad_tag = rng.choice([t for t in TAGS if t != rec["principle"]])
            bad = WRONG_TPL.format(tag=bad_tag, hint=rec["zach_verdict"][:60])
            rows.append({"prompt": p, "completion": bad, "label": False})
    return rows


def auto_user(rec):
    """One phase prefix only — 1,794/2,680 auto states already carry
    '[phase: ...]' and the v11 builder prepended '[phase: gate]' unconditionally."""
    state = rec["state"]
    q = rec.get("question") or rec["qid"]
    if not state.lstrip().startswith("[phase:"):
        state = f"[phase: gate] {state}"
    return f"{state}\nQuestion: {q}"


def model_tag_of(rec):
    """The local gate's own verdict tag, if the record stores one; fall back to
    a random CANON tag."""
    m = MODEL_TAG_RE.search(str(rec.get("local_pred", "")))
    if m and m.group(1) != "OK":
        return m.group(1)
    return rng.choice(TAGS)


def kto_from_auto():
    rows = []
    for line in open(os.path.expanduser("~/judgment-model/corpus/auto-labeled.jsonl")):
        r = json.loads(line)
        p = [{"role": "system", "content": SYS}, {"role": "user", "content": auto_user(r)}]
        if r["gold"] == 1:
            rows.append({"prompt": p, "completion": "VERDICT: OK", "label": True})
            tag = model_tag_of(r)
            hint = rng.choice(GENERIC_HINTS)
            rows.append({"prompt": p, "completion": WRONG_TPL.format(tag=tag, hint=hint),
                         "label": False})
        else:
            rows.append({"prompt": p, "completion": "VERDICT: OK", "label": False})
            tag = model_tag_of(r)
            hint = rng.choice(GENERIC_HINTS)
            rows.append({"prompt": p, "completion": WRONG_TPL.format(tag=tag, hint=hint),
                         "label": True})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", type=int, default=3000, help="cap per KTO class")
    a = ap.parse_args()

    sft_tr, sft_ev = sft_sets()
    with open(f"{BENCH}/sft_train.jsonl", "w") as fh:
        for m in sft_tr:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(f"{BENCH}/sft_eval.jsonl", "w") as fh:
        for m in sft_ev:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    kto = kto_from_approved() + kto_from_auto()
    pos = [r for r in kto if r["label"]]
    neg = [r for r in kto if not r["label"]]
    n = min(len(pos), len(neg), a.balance)
    rng.shuffle(pos)
    rng.shuffle(neg)
    balanced = pos[:n] + neg[:n]
    rng.shuffle(balanced)
    with open(f"{BENCH}/kto_train.jsonl", "w") as fh:
        for r in balanced:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"sft_train {len(sft_tr)} | sft_eval {len(sft_ev)} (frozen)")
    print(f"kto raw pos {len(pos)} neg {len(neg)} -> balanced {2*n} -> kto_train.jsonl")


if __name__ == "__main__":
    main()
