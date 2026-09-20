#!/usr/bin/env python3
"""Audit whether the judgment gate's flags correlate with actual rework.

Method: claude-stop-hook fires at every assistant turn-end in a Claude Code
session and logs the verdict + the tail of the assistant reply. Pair each gate
call with the NEXT user message in the same session (from the session jsonl).
If the gate flagged, did the user rework-scold ("no i didnt"/"why did u"/"stop")
or continue ("ok"/"next"/"go")? Compare flag vs OK rates between the two pools.
"""
import json, os, re, sys, glob, collections
from datetime import datetime

LOG = os.path.expanduser("~/judgment-model/gate-log.jsonl")
SESS_GLOB = os.path.expanduser("~/.claude/projects/*/[0-9a-f]*.jsonl")

REWORK = re.compile(r"\b(no,|no i|nope|didn'?t|didnt|why (did|are|is|u|you)|stop |don'?t|dont|again|wrong|instead|i said|that'?s not|isn'?t|wasn'?t|still (broken|not|the)|u (broke|deleted|overwrote|changed)|revert|undo|not what i|never mind|nah)\b", re.I)
CONTINUE = re.compile(r"^(ok(ay|ie|)?\b|cool|nice|good|great|thanks|thx|ty|nice|alright|sure|yep|yes|yup|go|next|then|now|do it|do the|pls |please |can u|could u|what about|how about|let'?s|ship|deploy|commit|push|run |check |show |add |make |build |fix |create |write |update |look |see |tell |give |send |start |try |test |review |compare|research|find|search|open |install|set |config|remember|also |and |btw)\b", re.I)

def parse_ts(s):
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")

def load_gate():
    calls = []
    for line in open(LOG):
        try: d = json.loads(line)
        except: continue
        if d.get("caller") != "claude-stop-hook": continue
        sid = d.get("session")
        if not sid: continue
        v = (d.get("verdict") or "").strip()
        vu = v.upper()
        if vu == "OK" or vu.startswith("OK ") or vu.startswith("VERDICT: OK"):
            kind = "OK"; tag = "OK"
        elif vu.startswith("VERDICT:"):
            kind = "FLAG"; tag = v[8:].split("|")[0].strip()
        else:
            kind = "ERR"; tag = ""
        calls.append({"ts": parse_ts(d["ts"]), "sid": sid, "kind": kind,
                      "tag": locals().get("tag", ""), "v": v[:120],
                      "sit": (d.get("situation_head") or "")[:160]})
    return calls

def load_sessions(sids):
    """sid -> list of (ts, role, text) user/assistant messages."""
    out = collections.defaultdict(list)
    for f in glob.glob(SESS_GLOB):
        base = os.path.basename(f)[:-6]
        if base not in sids: continue
        try:
            for line in open(f):
                try: d = json.loads(line)
                except: continue
                t = d.get("type")
                if t not in ("user", "assistant"): continue
                msg = d.get("message") or {}
                c = msg.get("content")
                if isinstance(c, list):
                    txt = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                else:
                    txt = c if isinstance(c, str) else ""
                txt = txt.strip()
                if not txt or txt.startswith("<"): continue
                if txt.startswith("Stop hook feedback"): continue
                if txt.startswith("This session is being continued"): continue
                if "qwen:user-prompt-submit-context" in txt: continue
                tsr = d.get("timestamp", "")
                try: ts = datetime.strptime(tsr[:19], "%Y-%m-%dT%H:%M:%S")
                except: continue
                out[base].append((ts, t, txt))
        except Exception:
            pass
    return out

def main():
    calls = load_gate()
    by_sess = collections.defaultdict(list)
    for c in calls:
        if c["kind"] == "ERR": continue
        by_sess[c["sid"]].append(c)
    sess = load_sessions(set(by_sess.keys()))
    print(f"gate calls (OK/FLAG): {sum(1 for c in calls if c['kind']!='ERR')}")
    print(f"sessions with gate calls: {len(by_sess)}, sessions found on disk: {len(sess)}")

    pairs = []  # (call, next_user_text)
    for sid, cs in by_sess.items():
        msgs = sess.get(sid)
        if not msgs: continue
        msgs.sort(key=lambda m: m[0])
        users = [(ts, tx) for ts, r, tx in msgs if r == "user"]
        for c in cs:
            nxt = next(((ts, tx) for ts, tx in users if ts >= c["ts"]), None)
            if nxt is None: continue
            gap = (nxt[0] - c["ts"]).total_seconds()
            if gap > 3600: continue
            pairs.append((c, nxt[1]))

    def classify(tx):
        return "rework" if REWORK.search(tx) else ("continue" if CONTINUE.search(tx[:60]) else "other")

    stats = collections.Counter()
    examples = collections.defaultdict(list)
    for c, tx in pairs:
        lab = classify(tx)
        stats[(c["kind"], lab)] += 1
        if lab == "rework" and len(examples[c["kind"]]) < 6:
            examples[c["kind"]].append((c["tag"], c["v"][:70], tx[:110].replace("\n", " ")))

    print(f"\npaired gate calls: {len(pairs)}")
    for kind in ("OK", "FLAG"):
        tot = sum(stats[(kind, l)] for l in ("rework", "continue", "other"))
        if not tot: continue
        rw = stats[(kind, "rework")]; co = stats[(kind, "continue")]
        print(f"{kind:5s} n={tot:5d}  rework-after={rw} ({100*rw/tot:.1f}%)  continue-after={co} ({100*co/tot:.1f}%)")
    print("\n-- rework examples after FLAG --")
    for tag, v, tx in examples["FLAG"]: print(f"  [{tag}] {v}  =>  USER: {tx}")
    print("\n-- rework examples after OK --")
    for tag, v, tx in examples["OK"]: print(f"  {v}  =>  USER: {tx}")

    # per-tag precision proxy: flag -> rework rate
    tagstats = collections.defaultdict(collections.Counter)
    for c, tx in pairs:
        if c["kind"] == "FLAG":
            tagstats[c["tag"]][classify(tx)] += 1
    print("\n-- per-tag: rework% among flags (n>=30) --")
    for tag, st in sorted(tagstats.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(st.values())
        if n >= 30:
            print(f"  {tag:15s} n={n:4d} rework={100*st['rework']/n:4.1f}% continue={100*st['continue']/n:4.1f}%")

if __name__ == "__main__":
    main()
