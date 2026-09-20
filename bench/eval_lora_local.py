#!/usr/bin/env python3
"""Eval a judgment-gate LoRA adapter on the frozen heldout, locally.

Runs the same prompt/greedy/scoring path as the Lambda eval.py, but against
a local llama.cpp-style server (default :8005, the judgment-gate
llama-server). Pass --base-model so the served base and the adapter's base
match, otherwise the result is meaningless.

  eval_lora_local.py --base-url http://127.0.0.1:8005/v1 --model judgment
  EVAL_ADAPTER=/path/to/lora eval_lora_local.py ...
"""
import json
import os
import re
import sys
import urllib.request

BASE = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8005/v1")
MODEL = os.environ.get("EVAL_MODEL", "judgment")
ADAPTER = os.environ.get("EVAL_ADAPTER", "")
EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_eval.jsonl")


def chat(messages):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": 0, "max_tokens": 48}).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


TAGRE = re.compile(r"VERDICT:\s*([A-Z][A-Z-]*)")


def score_tag(text):
    """Extract the verdict tag from gold/pred for comparison.

    FIX (v12): the gate model sometimes emits a stray think-token artifact
    before the verdict ("VERDICT: X"); the old string comparison scored a
    correct tag as wrong. Take the LAST tag occurrence, not the first.
    """
    m = TAGRE.findall(text)
    return m[-1] if m else ""


def main():
    ok = tot = okn = totn = 0
    misses = []
    for line in open(EVAL):
        m = json.loads(line)["messages"]
        gold = m[-1]["content"]
        pred = chat(m[:2])
        gt, pt = score_tag(gold), score_tag(pred)
        tot += 1
        ok += gt == pt
        if gt != "OK":
            totn += 1
            okn += gt == pt
            if gt != pt:
                misses.append((gt, pt[:40]))
    print(f"adapter={ADAPTER or '(none — base model)'}")
    print(f"eval heldout: all {ok}/{tot}  non-OK {okn}/{totn}")
    for gt, pt in misses[:10]:
        print(f"  gold {gt:<28} pred {pt}")


if __name__ == "__main__":
    main()
