#!/usr/bin/env python3
"""Shared embedding shortlist for high-cardinality choice questions.

  shortlist.py banking77 [--k 20] [--model BAAI/bge-small-en-v1.5]

Of the arena tasks only banking77 (77 intents) is high-cardinality: massive has
18 scenarios and gate 16 verdicts, both under k. The ranking is laya's own
coarse-to-fine step (laya.shortlist.shortlist_choice: cosine between the
instructions+state and each "label: description") with one shared bi-encoder.
It writes _data/<task>.sl.json = {"k", "model", "lists": {item id: {qid: [labels]}}}.

Every system run as "<system>+sl" asks over those SAME candidates, so the
comparison isolates the judge, not the retriever. k is fixed in advance
(laya's default, 20) and never tuned on these items; recall at other k is
printed for information only. A gold label the shortlist drops is a miss the
judge cannot recover, so recall@k is the ceiling on "+sl" accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "_data"
DEFAULT_K = 20


def shortlist_path(task: str) -> Path:
    return DATA / f"{task}.sl.json"


def recall_at_k(ranked: list[list[str]], golds: list[str], k: int) -> float:
    """Fraction of items whose gold label is among the first k ranked labels."""
    if not golds:
        return float("nan")
    return sum(1 for r, g in zip(ranked, golds) if g in r[:k]) / len(golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    from laya.shortlist import shortlist_choice
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(args.model, device="cpu")

    def embed_fn(texts):
        return enc.encode(list(texts), normalize_embeddings=True, batch_size=64)

    items = [json.loads(l) for l in open(DATA / f"{args.task}.jsonl")]
    lists: dict[str, dict[str, list[str]]] = {}
    by_q: dict[str, tuple[list, list]] = {}
    for it in items:
        for qid, q in it["questions"].items():
            if q["type"] != "choice" or len(q["criteria"]) <= args.k:
                continue
            # rank the FULL list once; the file keeps the top k, recall reads deeper
            ranked = shortlist_choice(it["state"], q["criteria"], embed_fn, k=len(q["criteria"]) - 1,
                                      instructions=q.get("instructions"))
            lists.setdefault(it["id"], {})[qid] = list(ranked[: args.k])
            r, g = by_q.setdefault(qid, ([], []))
            r.append(list(ranked))
            g.append(it["gold"][qid]["label"])
    if not lists:
        raise SystemExit(f"{args.task}: no choice question has more than {args.k} options; nothing to shortlist")
    shortlist_path(args.task).write_text(json.dumps({"k": args.k, "model": args.model, "lists": lists}))
    for qid, (r, g) in by_q.items():
        rec = {k: round(recall_at_k(r, g, k), 3) for k in (1, 5, 10, 20, 30)}
        print(f"{args.task}.{qid}: n={len(g)} recall@k {rec}  (shipped k={args.k})")
    print(f"wrote {shortlist_path(args.task)}")


if __name__ == "__main__":
    main()
