#!/usr/bin/env python3
"""Teacher run: cloud Jev probabilities + quorum local probs over the same items.

For each labeled noul item, records:
  - cloud: P(yes) from api.typesafe.ai (jev-latest)  -> the calibration target
  - local: raw quorum P(yes) direct, and with CoT    -> the thing to be calibrated
  - gold label
Batches questions (B per request) to quorum; cloud calls one-per-request with a
small parallel pool. Resumable: skips ids already present in --out.

Cost note: ~240 cloud calls x ~300 input tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/dev/quorum"))
from quorum.gate_client import ask  # noqa: E402

API = "https://api.typesafe.ai/v1/systemone"
KEYFILE = os.path.expanduser("~/.config/typesafe/env")
_lock = threading.Lock()


def _key():
    for line in open(KEYFILE):
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no TYPESAFE_API_KEY")


def cloud_noul(text, yes, no, retries=3):
    body = json.dumps({
        "model": "jev-latest", "state": text,
        "questions": {"q": {"type": "noul",
                            "instructions": "How true is the 'yes' criterion of the state?",
                            "criteria": {"yes": yes, "no": no}}},
    }).encode()
    for att in range(retries):
        try:
            req = urllib.request.Request(API, data=body, headers={
                "Authorization": "Bearer " + _key(), "content-type": "application/json"})
            t0 = time.time()
            out = json.loads(urllib.request.urlopen(req, timeout=45).read())
            return out["answers"]["q"]["noul"], int((time.time() - t0) * 1000)
        except Exception as e:
            if att == retries - 1:
                return None, str(e)[:120]
            time.sleep(1.5 * (att + 1))


def local_noul(text, yes, no, cot):
    q = {"q": {"type": "noul", "instructions": "How true is the 'yes' criterion of the state?",
               "criteria": {"yes": yes, "no": no}}}
    try:
        r = ask(text, q, caller="teacher-run", reasoning=cot)
        if "error" in r:
            return None, r["error"]
        return r["probs"].get("q"), r.get("latency_ms")
    except Exception as e:
        return None, str(e)[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="teacher_items.jsonl")
    ap.add_argument("--out", default="teacher_run.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    items = [json.loads(l) for l in open(a.items) if json.loads(l)["id"] not in done]
    print(f"{len(done)} done, {len(items)} to run", flush=True)

    out_fh = open(a.out, "a")

    def one(it):
        cp, cms = cloud_noul(it["text"], it["yes"], it["no"])
        dp, dms = local_noul(it["text"], it["yes"], it["no"], cot=False)
        tp, tms = local_noul(it["text"], it["yes"], it["no"], cot=True)
        rec = {**it, "cloud": cp, "cloud_ms": cms,
               "local_direct": dp, "local_cot": tp, "local_cot_ms": tms}
        with _lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
        return it["id"]

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rid in enumerate(ex.map(one, items), 1):
            if i % 20 == 0:
                print(f"{i}/{len(items)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
