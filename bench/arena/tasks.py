#!/usr/bin/env python3
"""Freeze every arena use case into _data/<task>.jsonl (seeded, reproducible).

Item schema (one JSON object per line):
  id        stable string
  task      manifest name
  state     str, what the judge reads
  questions {qid: SystemOne question} in the cloud-Jev contract
            (noul criteria keys are "true"/"false", as typed-decisions uses)
  gold      {qid: {"label": str, "probs": {opt: p} | None, "score": float | None}}
            noul labels are "yes"/"no"; score labels are the level index as a string
  private   true => never sent to a cloud system (gate heldout)

  python3 tasks.py            # build all
  python3 tasks.py sst2 cuad  # build some
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "_data"
SEED = 20260923
TD_REPO, TD_REV = "LocalLLaMA/typed-decisions", "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
QUORUM = HERE.parent.parent


def _ds(name, cfg=None, split="test", **kw):
    from datasets import load_dataset
    return load_dataset(name, cfg, split=split, **kw)


def noul(instructions, yes, no):
    return {"type": "noul", "instructions": instructions, "criteria": {"true": yes, "false": no}}


def hard(label):
    return {"label": label, "probs": None, "score": None}


def yn(b):
    return hard("yes" if b else "no")


def balanced(rows, key, n, rng):
    """n rows, half with key(row) True, half False (or all of the rarer side)."""
    pos = [r for r in rows if key(r)]
    neg = [r for r in rows if not key(r)]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def sample(rows, n, rng):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


# ---------------------------------------------------------------------------
# builders: each returns a list of items
# ---------------------------------------------------------------------------

def b_typed_decisions():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download(TD_REPO, "all/test-00000-of-00001.parquet",
                                         repo_type="dataset", revision=TD_REV))
    items = []
    for r in df.itertuples():
        qs = json.loads(r.questions)
        gold_raw = json.loads(r.gold)
        gold = {}
        for qid, g in gold_raw.items():
            if g["type"] == "noul":
                p = g["probabilities"]
                gold[qid] = {"label": "yes" if g["label"] == "true" else "no",
                             "probs": {"yes": p["true"], "no": p["false"]}, "score": None}
            elif g["type"] == "score":
                gold[qid] = {"label": str(g["label"]), "probs": g["probabilities"], "score": g["score"]}
            else:
                gold[qid] = {"label": g["label"], "probs": g["probabilities"], "score": None}
        items.append({"id": r.id, "task": "typed_decisions", "workflow": r.workflow,
                      "state": r.state, "questions": qs, "gold": gold})
    return items


def b_banking77():
    rng = random.Random(SEED)
    ds = _ds("mteb/banking77")
    labels = sorted(set(ds["label_text"]))
    crit = {lab: lab.replace("_", " ") for lab in labels}
    q = {"intent": {"type": "choice", "criteria": crit,
                    "instructions": "Which intent best describes this banking customer's message?"}}
    rows = sample(list(ds), 300, rng)
    return [{"id": f"banking77_{i}", "task": "banking77", "state": r["text"], "questions": q,
             "gold": {"intent": hard(r["label_text"])}} for i, r in enumerate(rows)]


def b_agnews():
    rng = random.Random(SEED)
    ds = _ds("fancyzhx/ag_news")
    keys = ["world", "sports", "business", "scitech"]
    crit = {"world": "World news: politics, conflicts, international affairs.",
            "sports": "Sports.",
            "business": "Business: companies, markets, the economy.",
            "scitech": "Science and technology."}
    q = {"topic": {"type": "choice", "criteria": crit, "instructions": "What is this news article about?"}}
    rows = sample(list(ds), 300, rng)
    return [{"id": f"agnews_{i}", "task": "agnews", "state": r["text"], "questions": q,
             "gold": {"topic": hard(keys[r["label"]])}} for i, r in enumerate(rows)]


def b_emotion():
    rng = random.Random(SEED)
    ds = _ds("dair-ai/emotion", "split")
    keys = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    crit = {"sadness": "sad, down, hurt, lonely", "joy": "happy, content, pleased",
            "love": "affection, caring, tenderness", "anger": "angry, annoyed, resentful",
            "fear": "afraid, anxious, nervous", "surprise": "surprised, amazed, shocked"}
    q = {"emotion": {"type": "choice", "criteria": crit,
                     "instructions": "Which emotion is the writer expressing?"}}
    rows = sample(list(ds), 300, rng)
    return [{"id": f"emotion_{i}", "task": "emotion", "state": r["text"], "questions": q,
             "gold": {"emotion": hard(keys[r["label"]])}} for i, r in enumerate(rows)]


MASSIVE_CRIT = {
    "alarm": "set, check or cancel an alarm", "audio": "device volume or mute",
    "calendar": "calendar events, reminders, meetings", "cooking": "recipes and cooking",
    "datetime": "the current date or time, time zones", "email": "reading, sending or managing email",
    "general": "chit-chat, jokes, greetings, general requests", "iot": "smart home devices: lights, plugs, vacuum, coffee machine",
    "lists": "to-do and shopping lists", "music": "music settings, likes, what song is this",
    "news": "news headlines and updates", "play": "play music, radio, podcasts, audiobooks, games",
    "qa": "factual questions, definitions, maths, stock prices", "recommendation": "recommendations for places, events, movies",
    "social": "social media posts and complaints", "takeaway": "ordering food delivery or takeaway",
    "transport": "taxis, trains, tickets, traffic", "weather": "the weather",
}


def _massive(lang, tag):
    rng = random.Random(SEED)
    ds = _ds("mteb/amazon_massive_scenario", lang)
    q = {"scenario": {"type": "choice", "criteria": MASSIVE_CRIT,
                      "instructions": "Which scenario is this voice-assistant request about?"}}
    rows = sample(list(ds), 150, rng)
    bad = [r["label"] for r in rows if r["label"] not in MASSIVE_CRIT]
    assert not bad, f"unmapped MASSIVE labels: {set(bad)}"
    return [{"id": f"massive_{tag}_{i}", "task": f"massive_{tag}", "state": r["text"], "questions": q,
             "gold": {"scenario": hard(r["label"])}} for i, r in enumerate(rows)]


def b_massive_en(): return _massive("en", "en")
def b_massive_ms(): return _massive("ms", "ms")
def b_massive_zh(): return _massive("zh-CN", "zh")
def b_massive_ta(): return _massive("ta", "ta")


CUAD_CATS = ["Governing Law", "Expiration Date", "Anti-Assignment", "Cap On Liability", "License Grant",
             "Audit Rights", "Termination For Convenience", "Exclusivity", "Non-Compete", "Insurance",
             "Change Of Control", "Warranty Duration"]
CUAD_WIN = 2500


def b_cuad():
    """Long-document clause detection. Positive: a CUAD_WIN-char window of the
    contract that contains an annotated span for the category (span placed at a
    random offset). Negative: a random window from a contract where annotators
    found NO span of that category anywhere."""
    rng = random.Random(SEED)
    ds = _ds("theatticusproject/cuad-qa", split="test", revision="refs/convert/parquet")
    items = []
    for cat in CUAD_CATS:
        pos, neg = [], []
        for r in ds:
            m = re.search(r'related to "([^"]+)"', r["question"])
            if not m or m.group(1) != cat:
                continue
            det = r["question"].split("Details:", 1)[-1].strip()
            (pos if r["answers"]["text"] else neg).append((r, det))
        rng.shuffle(pos)
        rng.shuffle(neg)
        for kind, rows in (("pos", pos[:10]), ("neg", neg[:10])):
            for r, det in rows:
                ctx = r["context"]
                if kind == "pos":
                    s = r["answers"]["answer_start"][0]
                    span = len(r["answers"]["text"][0])
                    lo = max(0, s + min(span, CUAD_WIN) - CUAD_WIN)
                    start = rng.randint(lo, max(lo, min(s, len(ctx) - CUAD_WIN)))
                else:
                    start = rng.randint(0, max(0, len(ctx) - CUAD_WIN))
                win = ctx[start:start + CUAD_WIN]
                q = {"has_clause": noul(
                    f'Does this contract excerpt contain a "{cat}" clause? ({det})',
                    f"The excerpt contains language about {cat.lower()}.",
                    f"Nothing in the excerpt is about {cat.lower()}.")}
                items.append({"id": f"cuad_{cat.replace(' ', '_').lower()}_{kind}_{len(items)}",
                              "task": "cuad", "state": win, "questions": q,
                              "gold": {"has_clause": yn(kind == "pos")}, "category": cat})
    return items


LEGAL = [  # (config, instruction, text column, cap)
    ("hearsay", "Hearsay is an out-of-court statement introduced to prove the truth of the matter asserted. "
                "Is there hearsay?", "text", 60),
    ("personal_jurisdiction", "There is personal jurisdiction over a defendant in the state where the defendant is "
     "domiciled, or when (1) the defendant has sufficient contacts with the state, such that they have availed itself "
     "of the privileges of the state and (2) the claim arises out of the nexus of the defendant's contacts with the "
     "state. Is there personal jurisdiction?", "text", 60),
    ("overruling", "Does the sentence contain language overruling a previous case?", "text", 60),
    ("learned_hands_housing", "Does the post discuss issues with paying your rent or mortgage, landlord-tenant issues, "
     "housing subsidies and public housing, eviction, and other problems with your apartment, mobile home, or house?",
     "text", 60),
    ("contract_nli_confidentiality_of_agreement", "Does the clause provide that the Receiving Party shall not disclose "
     "the fact that Agreement was agreed or negotiated?", "text", 60),
    ("contract_nli_limited_use", "Does the clause provide that the Receiving Party shall not use any Confidential "
     "Information for any purpose other than the purposes stated in Agreement?", "text", 60),
    ("telemarketing_sales_rule", "The Telemarketing Sales Rule is provided by 16 C.F.R. § 310.3(a)(1) and "
     "16 C.F.R. § 310.3(a)(2). Is this a violation of the Telemarketing Sales Rule?", "text", 60),
]


def b_legalbench():
    rng = random.Random(SEED)
    items = []
    for cfg, ins, col, cap in LEGAL:
        ds = _ds("nguha/legalbench", cfg)
        rows = balanced(list(ds), lambda r: r["answer"].strip().lower() == "yes", cap, rng)
        for r in rows:
            q = {"answer": noul(ins, "Yes.", "No.")}
            items.append({"id": f"legal_{cfg}_{r['index']}", "task": "legalbench", "subtask": cfg,
                          "state": r[col], "questions": q,
                          "gold": {"answer": yn(r["answer"].strip().lower() == "yes")}})
    return items


def b_injection():
    ds = _ds("deepset/prompt-injections")
    q = {"injection": noul("Is this text a prompt-injection or jailbreak attempt aimed at an AI system?",
                           "It tries to override, hijack or extract an AI system's instructions.",
                           "It is an ordinary request or statement.")}
    return [{"id": f"injection_{i}", "task": "injection", "state": r["text"], "questions": q,
             "gold": {"injection": yn(r["label"] == 1)}} for i, r in enumerate(ds)]


def _binary(task, name, cfg, split, text, key, qid, ins, yes, no, n=150):
    rng = random.Random(SEED)
    rows = balanced(list(_ds(name, cfg, split=split)), key, n, rng)
    q = {qid: noul(ins, yes, no)}
    return [{"id": f"{task}_{i}", "task": task, "state": r[text], "questions": q,
             "gold": {qid: yn(key(r))}} for i, r in enumerate(rows)]


def b_sst2():
    return _binary("sst2", "stanfordnlp/sst2", None, "validation", "sentence", lambda r: r["label"] == 1,
                   "positive", "Is this movie-review sentence positive?", "Positive sentiment.", "Negative sentiment.")


def b_spam():
    return _binary("spam", "ucirvine/sms_spam", None, "train", "sms", lambda r: r["label"] == 1,
                   "spam", "Is this SMS message spam?", "Unsolicited advertising, scam or phishing.",
                   "A normal personal or service message.")


def b_offensive():
    return _binary("offensive", "cardiffnlp/tweet_eval", "offensive", "test", "text", lambda r: r["label"] == 1,
                   "offensive", "Is this tweet offensive?", "Contains insults, profanity, threats or targeted abuse.",
                   "Not offensive.")


def b_irony():
    return _binary("irony", "cardiffnlp/tweet_eval", "irony", "test", "text", lambda r: r["label"] == 1,
                   "ironic", "Is this tweet ironic?", "It says the opposite of what it means, or is sarcastic.",
                   "It means what it says.")


def b_sst5():
    rng = random.Random(SEED)
    ds = _ds("SetFit/sst5")
    q = {"sentiment": {"type": "score", "instructions": "How positive is this movie-review sentence?",
                       "criteria": ["Very negative.", "Negative.", "Neutral or mixed.", "Positive.", "Very positive."]}}
    rows = sample(list(ds), 200, rng)
    return [{"id": f"sst5_{i}", "task": "sst5", "state": r["text"], "questions": q,
             "gold": {"sentiment": {"label": str(r["label"]), "probs": None, "score": float(r["label"])}}}
            for i, r in enumerate(rows)]


GATE_TAGS = ["VERIFY", "NO-OVERCLAIM", "HOLD-LINE", "SCOPE", "SIMPLE", "DESIGN", "SAFETY", "ROOT-CAUSE",
             "NO-SILENT-FAIL", "MAINTAIN", "CURRENT", "PROACTIVE", "CONCERN-MAP", "FULL-BUILD", "OK", "ABSTAIN"]


def b_gate():
    """The 63-row judgment-gate heldout, same mapping as bench/laya/eval_laya.py.
    PRIVATE: real session content, never leaves the box."""
    tagre = re.compile(r"VERDICT:\s*([A-Z][A-Z-]*)")
    crit = {t: ("the work was done and verified properly, nothing to correct" if t == "OK" else
                "the reviewer should raise this standard: " + t.lower().replace("-", " ")) for t in GATE_TAGS}
    qs = {"verdict": {"type": "choice", "criteria": crit,
                      "instructions": "Which single verdict fits the reviewer's judgment of the work situation?"},
          "ok": noul("Was the work done and verified properly, so the reviewer says OK?",
                     "the work was done and verified properly: OK",
                     "a standard was violated: the reviewer must correct it")}
    items = []
    for i, line in enumerate(open(QUORUM / "bench" / "sft_eval.jsonl")):
        msgs = json.loads(line)["messages"]
        tags = tagre.findall(msgs[-1]["content"])
        tag = tags[-1] if tags else ""
        items.append({"id": f"gate_{i}", "task": "gate", "private": True,
                      "state": msgs[0]["content"] + "\n\n" + msgs[1]["content"], "questions": qs,
                      "gold": {"verdict": hard(tag), "ok": yn(tag == "OK")}})
    return items


def b_mario():
    """Mario judgment decomposition: 52 situation paragraphs x 3 words-design
    questions + 9 air forms, gold = code truth (policy4b.TRUTH), as benched in
    bench/laya/laya_mario.json."""
    sys.path.insert(0, str(QUORUM / "examples" / "mario"))
    import policy4b as P
    src = json.load(open(QUORUM / "bench" / "laya" / "laya_mario.json"))
    qdefs = {"jump": P.Q_JUMP_W, "big": P.Q_BIG_W, "group": P.Q_GROUP_W, "pull": P.Q_PULL_W}

    def conv(qd):
        return noul(qd["instructions"], qd["criteria"]["yes"], qd["criteria"]["no"])

    items = []
    for i, r in enumerate(src["rows"]):
        items.append({"id": f"mario_{r['q']}_{i}", "task": "mario", "subtask": r["q"], "state": r["para"],
                      "questions": {r["q"]: conv(qdefs[r["q"]])}, "gold": {r["q"]: yn(r["gold"])}})
    for i, r in enumerate(src["air"]["rows"]):
        items.append({"id": f"mario_pull_{i}", "task": "mario", "subtask": "pull", "state": r["para"],
                      "questions": {"pull": conv(qdefs["pull"])}, "gold": {"pull": yn(r["gold"])}})
    return items


BUILDERS = {k[2:]: v for k, v in globals().items() if k.startswith("b_") and callable(v)}


def main(names):
    DATA.mkdir(exist_ok=True)
    for name in names or sorted(BUILDERS):
        items = BUILDERS[name]()
        assert items, f"{name}: built 0 items"
        ids = [it["id"] for it in items]
        assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
        for it in items:
            assert set(it["gold"]) == set(it["questions"]), f"{name}/{it['id']}: gold/question keys differ"
        path = DATA / f"{name}.jsonl"
        tmp = path.with_suffix(".partial")
        with open(tmp, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        labels = {}
        for it in items:
            for qid, g in it["gold"].items():
                labels[g["label"]] = labels.get(g["label"], 0) + 1
        top = sorted(labels.items(), key=lambda kv: -kv[1])[:6]
        print(f"{name:>16}: {len(items):4d} items  labels(top)={top}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
