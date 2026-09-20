#!/usr/bin/env python3
"""Offline bench: how well does a local quorum model reproduce cloud Jev's Mario decisions?

Cloud Jev's logged runs (incl. the 1-1 clear) are free teacher labels: state -> action.
Each DESIGN turns a logged ground state into (payload, questions) for the local shim and
composes the answers back into an action. No emulator, cached responses, compact metrics.

  python3 bench4b.py atomic-json atomic-prose jumpnow-json [--cot] [--limit N]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

from quorum_client import Quorum
import policy4b as P

HERE = Path(__file__).resolve().parent
TEACHER_PREFIX = "jev"        # decisions logged by a cloud Jev run are the teacher labels
CACHE = HERE / "runs" / "_bench_cache.jsonl"


def load_ground():
    """ground decisions from every logged cloud-Jev run that carries landing predictions.
    Make them with your own TypeSafe key:  python3 play.py --backend cloud --attempts 3"""
    out = []
    for f in sorted((HERE / "runs").glob("*/attempt_*/decisions.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if not str(d.get("model", "")).startswith(TEACHER_PREFIX) or "if_hop_right" not in d["state"]:
                continue
            if d["kind"] == "ground" and d["choice"] in ("run_right", "hop_right", "jump_right"):
                out.append((d["state"], d["choice"]))
    if not out:
        raise SystemExit("no teacher logs under runs/: record some with  python3 play.py --backend cloud")
    return out


class Cached:
    def __init__(self, reasoning):
        self.cl = Quorum(reasoning=reasoning)
        self.mem = {}
        if CACHE.exists():
            for l in open(CACHE):
                r = json.loads(l)
                self.mem[r["k"]] = r["a"]
        self.reasoning = reasoning
        self.calls = 0
        self.secs = 0.0

    def ask(self, payload, questions):
        k = hashlib.sha1(json.dumps([payload, questions, self.reasoning], sort_keys=True).encode()).hexdigest()
        if k not in self.mem:
            t0 = time.time()
            a = self.cl.ask(payload, questions)["answers"]
            self.secs += time.time() - t0
            self.calls += 1
            self.mem[k] = a
            with open(CACHE, "a") as f:
                f.write(json.dumps({"k": k, "a": a}) + "\n")
        return self.mem[k]


def pyes(ans):
    v = (ans or {}).get("noul")
    if v is None:                       # no distribution: fall back to the committed answer
        return 1.0 if (ans or {}).get("answer") == "yes" else 0.0
    return float(v)


def best_threshold(scores, labels):
    best = (0, 0.5)
    for t in sorted(set(scores)):
        acc = sum((s >= t) == y for s, y in zip(scores, labels)) / len(labels)
        if acc > best[0]:
            best = (acc, t)
    return best


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("designs", nargs="+")
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--enemy-set", action="store_true", help="unique states with a hostile ahead + a few without")
    ap.add_argument("--dev", action="store_true", help="all teacher jumps + evenly sampled hard negatives")
    args = ap.parse_args()
    data = load_ground()
    if args.limit:
        data = data[:: max(1, len(data) // args.limit)]
    if args.enemy_set:
        seen, keep, none = set(), [], []
        for d in data:
            k = json.dumps(P.hostile_ahead(d[0]), sort_keys=True)
            if k in seen:
                continue
            seen.add(k)
            keep.append(d)
        data = keep
    if args.dev:
        pos = [d for d in data if d[1] != "run_right"]
        hard = [d for d in data if d[1] == "run_right" and
                (P.hostile_ahead(d[0]) or d[0].get("nearest_wall") or d[0].get("nearest_pit"))]
        data = pos + hard
        seen, uniq = set(), []
        for d in data:                       # the runs share early-level states: keep one of each
            k = json.dumps(P.pruned(d[0]), sort_keys=True)
            if k not in seen:
                seen.add(k)
                uniq.append(d)
        data = uniq
    n_jump = sum(c != "run_right" for _, c in data)
    print(f"ground states {len(data)} | teacher jumps {n_jump} (hop {sum(c=='hop_right' for _,c in data)}, "
          f"full {sum(c=='jump_right' for _,c in data)})", flush=True)
    for name in args.designs:
        design = P.DESIGNS[name]
        ca = Cached(args.cot)
        rows = []
        for state, teacher in data:
            payload, questions = design["build"](state)
            ans = ca.ask(payload, questions)
            rows.append((state, teacher, {q: pyes(a) for q, a in ans.items()}))
        ylab = [t != "run_right" for _, t, _ in rows]
        jscore = [design["jump_score"](p) for _, _, p in rows]
        acc, thr = best_threshold(jscore, ylab)
        print(f"\n== {name} ({'cot' if args.cot else 'direct'})  new calls {ca.calls} "
              f"avg {ca.secs / ca.calls if ca.calls else 0:.1f}s", flush=True)
        print(f"  jump-vs-run   AUC {auc(jscore, ylab):.3f}  best acc {acc:.3f} @thr {thr:.3f}  "
              f"(floor {max(sum(ylab), len(ylab) - sum(ylab)) / len(ylab):.3f})")
        for q in rows[0][2]:
            s = [p[q] for _, _, p in rows]
            print(f"  noul {q:12} AUC vs jump {auc(s, ylab):.3f}  mean|jump {sum(v for v, y in zip(s, ylab) if y) / max(1, sum(ylab)):.2f}"
                  f"  mean|run {sum(v for v, y in zip(s, ylab) if not y) / max(1, len(ylab) - sum(ylab)):.2f}")
        if "full_score" in design:
            jr = [(design["full_score"](p), t == "jump_right") for _, t, p in rows if t != "run_right"]
            fa, ft = best_threshold([s for s, _ in jr], [y for _, y in jr])
            print(f"  full-vs-hop   AUC {auc([s for s, _ in jr], [y for _, y in jr]):.3f}  best acc {fa:.3f} @thr {ft:.3f}")
        truths = dict(P.TRUTH, jump=lambda s: any(P.TRUTH[k](s) for k in ("enemy", "pit", "wall", "ledge")) or bool(s.get("stuck")))
        for q, truth in truths.items():        # reading accuracy: does the answer match what the state says?
            if q in rows[0][2]:
                tv = [truth(st) for st, _, _ in rows]
                sv = [p[q] for _, _, p in rows]
                ra, rt = best_threshold(sv, tv)
                print(f"  reads {q:6} truth-yes {sum(tv):3}  AUC {auc(sv, tv):.3f}  acc@0.5 "
                      f"{sum((v >= 0.5) == t for v, t in zip(sv, tv)) / len(tv):.3f}  best acc {ra:.3f} @thr {rt:.2f}")
        miss = [(st, t, p) for (st, t, p), s, y in zip(rows, jscore, ylab) if y and s < thr]
        print(f"  missed jumps at best thr: {len(miss)}/{sum(ylab)}")
        for st, t, p in miss[:4]:
            print(f"    teacher={t} wall={st.get('nearest_wall')} pit={st.get('nearest_pit')} "
                  f"en={[(e['distance'], e['height']) for e in st.get('enemies', [])[:2]]} p={ {k: round(v, 2) for k, v in p.items()} }")


if __name__ == "__main__":
    main()
