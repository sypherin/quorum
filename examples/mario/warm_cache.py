#!/usr/bin/env python3
"""Fill the judgment cache: ask the local model every question about every distinct paragraph
seen in any logged run (plus the four air forms). Live play then runs on lookups; the answers
are the model's own, stored once (greedy decoding makes them repeatable)."""
import glob, itertools, json
from pathlib import Path
import policy4b as P
from play import JudgmentCache
from quorum_client import Quorum

HERE = Path(__file__).resolve().parent
cache = JudgmentCache(Quorum(reasoning=True), HERE / "runs" / "_judgment_cache.jsonl")
ground, n = set(), 0
for f in glob.glob(str(HERE / "runs" / "*" / "attempt_*" / "decisions.jsonl")):
    for l in open(f):
        d = json.loads(l)
        if d.get("kind") == "ground" and "if_hop_right" in d["state"]:
            ground.add(P.situation(d["state"]))
print("distinct ground paragraphs:", len(ground), flush=True)
for sit in sorted(ground):
    for key, q in (("jump", P.Q_JUMP_W), ("big", P.Q_BIG_W), ("group", P.Q_GROUP_W)):
        cache.ask({"situation": sit}, {key: q}); n += 1
for k, pl in itertools.product(("floor", "pit", None), repeat=2):
    st = {"if_keep_right": {"lands_on": k} if k else None, "if_pull_left": {"lands_on": pl} if pl else None}
    cache.ask({"situation": P.air_situation(st)}, {"pull": P.Q_PULL_W}); n += 1
print(f"asked {n} | new model calls {cache.misses} | already cached {cache.hits} | entries {len(cache.mem)}")
