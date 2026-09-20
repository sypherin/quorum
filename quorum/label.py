"""quorum.label — interactive labeling pass over the judgment log.

The corpus loop is log -> label -> calibrate; this is the "label" step.
Walks the log newest-first, shows one judgment (state + what quorum decided)
per screen, and records the human verdict as a "labels": {qid: value} dict
merged into that record in place. Keys are the question ids, values are the
answer values as quorum states them ("yes"/"no", option keys, level digits;
for score you may also enter a 0..1 graded correctness).

Keys at the prompt: Enter/`y` = agree with quorum's choice for every
question; `n` or a value = override (prompts per question); `s` = skip;
`q` = quit (writes what was labeled so far).

Usage:
    python3 -m quorum.label [log/judgments.jsonl] [--limit 30] [--qid q]
                            [--random] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .report import prob_of

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "log" / "judgments.jsonl"


def candidates(records: list[dict], qid: str | None) -> list[int]:
    idx = []
    for i, r in enumerate(records):
        labels = r.get("labels")
        if isinstance(labels, dict) and labels:
            continue  # already labeled
        if r.get("label") is not None:
            continue
        if not (r.get("answers") or {}):
            continue
        if qid and qid not in r["answers"]:
            continue
        idx.append(i)
    return idx


def prompt_value(qid: str, ans: dict, default: str, inp) -> str | None:
    t = ans.get("type")
    if t == "noul":
        tip = f"yes/no [Enter={default}]"
    elif t == "choice":
        tip = f"one of {sorted((ans.get('probabilities') or {}).keys())} [Enter={default}]"
    else:
        tip = f"level digit, or 0..1 graded [Enter={default}]"
    v = inp(f"  {qid} — correct? {tip}: ").strip()
    if not v:
        return default
    return v


def label_record(r: dict, inp) -> dict | None:
    """Returns the labels dict, or None to skip this record."""
    state = r.get("state")
    if isinstance(state, (dict, list)):
        state = json.dumps(state, ensure_ascii=False)
    state = str(state)
    print("\n" + "-" * 72)
    print("STATE:", state[:600] + ("…" if len(state) > 600 else ""))
    defaults: dict[str, str] = {}
    for qid, ans in r["answers"].items():
        p, chosen = prob_of(ans)
        if p is None:
            print(f"  {qid}: (no probability — answer blind)")
            defaults[qid] = ""
        else:
            print(f"  {qid}: {chosen!r}  p={p:.3f}")
            defaults[qid] = chosen
    while True:
        cmd = inp("agree? [Enter/y]=yes  n=override  s=skip  q=quit: ").strip().lower()
        if cmd in ("", "y"):
            if all(defaults[q] for q in defaults):
                return dict(defaults)
            print("  some questions had no probability — answer them:")
            cmd = "n"
        if cmd == "s":
            return None
        if cmd == "q":
            return "QUIT"
        if cmd == "n":
            labels = {}
            bad = False
            for qid, ans in r["answers"].items():
                v = prompt_value(qid, ans, defaults[qid], inp)
                if v == "q":
                    return "QUIT"
                if v is None or v == "":
                    bad = True
                    break
                labels[qid] = v
            if not bad:
                return labels
            print("  incomplete — try again")
            continue
        # unknown input, loop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", nargs="?", default=str(DEFAULT_LOG))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--qid", default=None)
    ap.add_argument("--random", action="store_true", help="shuffle instead of newest-first")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    path = Path(args.log)
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]
    pool = candidates(records, args.qid)
    if args.random:
        random.Random(args.seed).shuffle(pool)
    else:
        pool = list(reversed(pool))
    pool = pool[: args.limit]
    print(f"{len(pool)} unlabeled records selected from {len(records)}")

    inp = input
    done = 0
    for i in pool:
        labels = label_record(records[i], inp)
        if labels == "QUIT":
            break
        if labels is None:
            continue
        rec = records[i]
        rec["labels"] = labels
        rec["labeled_ts"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()
        lines[i] = json.dumps(rec, ensure_ascii=False)
        done += 1
        if labels:
            print(f"  saved {labels}")
        if done >= args.limit:
            break
    path.write_text("\n".join(lines) + "\n")
    print(f"\nlabeled {done} records, wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
