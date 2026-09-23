#!/usr/bin/env python3
"""Run one system over arena tasks. Resumable: an item already answered in
_runs/<system>/<task>.jsonl is skipped; an item that errored is retried.

  run.py <system> [task ...] [--limit N] [--workers K] [--min-avail-gib G]

Every line: {"id", "ok", "answers": {qid: normalized}, "raw": {qid: raw}, "ms", "meta"}
or {"id", "ok": false, "error"}. Failures are written, counted and printed, never skipped.
The memory watchdog stops the run (exit 3) when MemAvailable falls below
--min-avail-gib, before the host's own OOM guard has to.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA, RUNS = HERE / "_data", HERE / "_runs"
sys.path.insert(0, str(HERE))

import adapters  # noqa: E402


def mem_avail_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return float("nan")


def load_items(task: str) -> list[dict]:
    return [json.loads(l) for l in open(DATA / f"{task}.jsonl")]


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("ok"):
            out.add(r["id"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-avail-gib", type=float, default=None,
                    help="memory floor; default 6.5 for systems that load a local model (laya*), "
                         "off for quorum (model already resident) and jev (cloud)")
    args = ap.parse_args()
    if args.min_avail_gib is None:
        args.min_avail_gib = 6.5 if args.system.startswith("laya") else 0.0

    tasks = args.tasks or sorted(p.stem for p in DATA.glob("*.jsonl"))
    if mem_avail_gib() < args.min_avail_gib:
        raise SystemExit(f"refusing to start: MemAvailable {mem_avail_gib():.1f} GiB < {args.min_avail_gib}")
    sysobj = adapters.make(args.system)
    outdir = RUNS / args.system
    outdir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    grand = {"ok": 0, "err": 0, "skipped_private": 0}

    for task in tasks:
        items = load_items(task)
        if args.limit:
            items = items[: args.limit]
        path = outdir / f"{task}.jsonl"
        have = done_ids(path)
        todo = [it for it in items if it["id"] not in have]
        if args.system == "jev":
            priv = [it for it in todo if it.get("private")]
            grand["skipped_private"] += len(priv)
            todo = [it for it in todo if not it.get("private")]
            if priv:
                print(f"[{task}] {len(priv)} private items NOT sent to cloud", flush=True)
        if not todo:
            print(f"[{task}] {len(items)} items, all done", flush=True)
            continue
        stats = {"ok": 0, "err": 0, "ms": 0}
        t0 = time.time()
        f = open(path, "a")
        stop = threading.Event()

        def one(it):
            if stop.is_set():
                return
            if mem_avail_gib() < args.min_avail_gib:
                stop.set()
                return
            t1 = time.time()
            try:
                raw, meta = sysobj.ask(it)
                ms = round((time.time() - t1) * 1000)
                norm = {qid: adapters.normalize_answer(raw.get(qid), q) for qid, q in it["questions"].items()}
                rec = {"id": it["id"], "ok": True, "answers": norm, "raw": raw, "ms": ms, "meta": meta}
                key = "ok"
            except Exception as e:  # noqa: BLE001 - written and counted
                rec = {"id": it["id"], "ok": False, "error": f"{type(e).__name__}: {e}"[:500],
                       "ms": round((time.time() - t1) * 1000)}
                key = "err"
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                stats[key] += 1
                stats["ms"] += rec["ms"]
                n = stats["ok"] + stats["err"]
                if key == "err" and stats["err"] <= 5:
                    print(f"[{task}] ERROR {it['id']}: {rec['error'][:200]}", flush=True)
                if n % 50 == 0:
                    print(f"[{task}] {n}/{len(todo)} ok={stats['ok']} err={stats['err']} "
                          f"avg={stats['ms'] / n:.0f}ms avail={mem_avail_gib():.1f}GiB", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, todo))
        f.close()
        n = stats["ok"] + stats["err"]
        print(f"[{task}] DONE {stats['ok']}/{len(todo)} ok, {stats['err']} errors, "
              f"{(time.time() - t0):.0f}s wall, avg {stats['ms'] / max(1, n):.0f} ms/item", flush=True)
        grand["ok"] += stats["ok"]
        grand["err"] += stats["err"]
        if stop.is_set():
            print(f"STOPPED by memory watchdog: MemAvailable {mem_avail_gib():.1f} GiB "
                  f"< {args.min_avail_gib}", flush=True)
            sys.exit(3)
    print(f"TOTAL {args.system}: ok={grand['ok']} err={grand['err']} private_skipped={grand['skipped_private']}",
          flush=True)


if __name__ == "__main__":
    main()
