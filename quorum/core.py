"""quorum core — pure logic for the local quorum shim.

Maps a typesafe-style request {state, questions{noul|choice|score}} onto a
constrained llama-server call and extracts typed answers + probabilities
from logprobs. No I/O here; tests cover everything in this module.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

PRIMITIVES = ("noul", "choice", "score")


class BadRequest(ValueError):
    pass


def validate_request(body: dict) -> tuple[Any, dict]:
    if not isinstance(body, dict):
        raise BadRequest("body must be an object")
    state = body.get("state")
    if state is None or state == "":
        raise BadRequest("state is required")
    if not isinstance(state, (str, dict, list)):
        raise BadRequest("state must be string or JSON object/array")
    if isinstance(state, (dict, list)):
        state = json.dumps(state, ensure_ascii=False)
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise BadRequest("questions must be a non-empty object")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in PRIMITIVES:
            raise BadRequest(f"question {qid!r}: type must be one of {PRIMITIVES}")
        if not q.get("instructions"):
            raise BadRequest(f"question {qid!r}: instructions required")
        if q["type"] in ("choice", "score"):
            crit = q.get("criteria")
            if q["type"] == "choice":
                if not isinstance(crit, dict) or not crit:
                    raise BadRequest(f"question {qid!r}: choice criteria must be an object of options")
            else:
                # score criteria: list of level descriptions (typesafe rejects objects)
                if not isinstance(crit, list) or len(crit) < 2:
                    raise BadRequest(f"question {qid!r}: score criteria must be a list of >=2 levels")
    return state, questions


# Chain-of-thought (opt-in): a bounded reasoning field emitted BEFORE each answer.
# The 4B judges far better with a rationale, but the field MUST be length-capped —
# an unbounded string overruns max_tokens and truncates the JSON before the answer
# token is ever emitted (measured 2026-09-18: bounded CoT lifts noul 57.7% -> 94.2%).
REASON_SUFFIX = "__why"
REASON_MAXLEN = 200  # chars (~one 15-word sentence); backstop to the prompt instruction


def build_schema(questions: dict, cot: bool = False) -> dict:
    """JSON schema forcing exactly one answer field per question.

    cot=True interleaves a bounded '<id>__why' string before each answer, so the
    model reasons then commits. Answer keys/positions are unchanged, so logprob
    extraction (which reads at the answer token) works identically."""
    props: dict[str, Any] = {}
    required: list[str] = []
    for qid, q in questions.items():
        if cot:
            wkey = qid + REASON_SUFFIX
            props[wkey] = {"type": "string", "maxLength": REASON_MAXLEN}
            required.append(wkey)
        t = q["type"]
        if t == "noul":
            props[qid] = {"type": "string", "enum": ["yes", "no"]}
        elif t == "choice":
            props[qid] = {"type": "string", "enum": list(q["criteria"].keys())}
        else:
            props[qid] = {"type": "integer", "minimum": 0, "maximum": len(q["criteria"]) - 1}
        required.append(qid)
    return {
        "name": "quorum_answers",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        },
    }


def build_messages(state: str, questions: dict, cot: bool = False) -> list[dict]:
    if cot:
        sys = (
            "You are a careful judgment engine. For each question you are given a reasoning "
            "field named '<id>__why' and an answer field named '<id>'. In each '<id>__why' put "
            "ONE short sentence (max 15 words) weighing the evidence, then put the required value "
            "in '<id>': yes/no for noul, the option key for choice, the integer level for score. "
            "Reason ONLY inside the '__why' fields, and never add any other field."
        )
    else:
        sys = (
            "You are a judgment engine. You do not explain, do not reason in the answer, "
            "and never add fields. For each question, judge the state and output the single "
            "required answer value: yes/no for noul questions, the option key for choice "
            "questions, the integer level for score questions."
        )
    qdef = {}
    for qid, q in questions.items():
        t = q["type"]
        d = {"type": t, "instructions": q["instructions"]}
        if t == "choice":
            d["options"] = {k: v for k, v in q["criteria"].items()}
        elif t == "score":
            d["levels"] = {str(i): v for i, v in enumerate(q["criteria"])}
        else:
            d["answer"] = "yes or no"
            if q.get("criteria"):
                # noul criteria (yes/no definitions) carry real meaning — pass
                # them through instead of dropping them silently.
                d["criteria"] = q["criteria"]
        qdef[qid] = d
    user = "STATE:\n" + state + "\n\nQUESTIONS (answer every key, keys are verbatim):\n" + json.dumps(qdef, ensure_ascii=False)
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Calibration (temperature scaling)
# ---------------------------------------------------------------------------

def load_temperatures() -> dict[str, float]:
    """Per-question-type temperatures from calibration.json.

    Path: $QUORUM_CALIBRATION, else calibration.json next to this module.
    Missing file => {} => T=1 passthrough (uncalibrated)."""
    path = os.environ.get("QUORUM_CALIBRATION")
    path = Path(path) if path else Path(__file__).with_name("calibration.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: float(v) for k, v in data.get("temperatures", {}).items()}


def temperature_for(kind: str, temps: dict[str, float]) -> float:
    return temps.get(kind, temps.get("default", 1.0))


def _candidate_matches(tok_text: str, candidates: list[str]) -> list[str]:
    """Candidates a token could be the start of (token may carry quotes/punct).

    Exact match wins outright. Prefix matches are only usable when exactly one
    candidate matches — a token like "not" is the start of both "not_now" and
    "not_a_fit", and attributing its logprob to either would be wrong."""
    t = tok_text.strip().strip('"').lstrip()
    if not t:
        return []
    exact = [c for c in candidates if c == t]
    if exact:
        return exact
    return [c for c in candidates if c.startswith(t[: len(c)])]


def probs_at_position(logprobs: list, value_token_index: int, candidates: list[str],
                      temperature: float = 1.0) -> dict[str, float] | None:
    """Distribution over candidates from top_logprobs at the value's token position.

    `temperature` scales logprobs before normalization (softmax(lp / T));
    T=1 is a passthrough."""
    if value_token_index is None or value_token_index >= len(logprobs):
        return None
    entry = logprobs[value_token_index]
    tops = entry.get("top_logprobs") or []
    if not tops:
        return None
    scores: dict[str, float] = {}
    for tp in tops:
        tok = tp.get("token", "")
        lp = tp.get("logprob")
        if lp is None:
            continue
        matches = _candidate_matches(tok, candidates)
        if len(matches) != 1:
            continue  # no match, or ambiguous prefix — never misattribute
        cand = matches[0]
        if cand not in scores:
            scores[cand] = math.exp(lp / temperature)
    if len(scores) < 2:  # need at least a competing signal to be meaningful
        return None
    total = sum(scores.values())
    if total <= 0:
        return None
    return {c: scores.get(c, 0.0) / total for c in candidates}


def probs_for_key(logprobs: list, key: str, candidates: list[str], max_walk: int = 4,
                  temperature: float = 1.0) -> dict[str, float] | None:
    """Distribution over candidates at the value position for `key`.

    The JSON emitter may put spacer tokens (' \"', ' ') between the colon and
    the value, and the value may be quoted/merged — walk forward from the key
    and accept the first position that distinguishes >=2 candidates."""
    start = find_value_token(logprobs, key)
    if start is None:
        return None
    for i in range(start, min(start + max_walk, len(logprobs))):
        p = probs_at_position(logprobs, i, candidates, temperature)
        if p is not None:
            return p
    return None


def find_value_token(logprobs: list, key: str) -> int | None:
    """Index of the token that starts the value for `key`, scanning the raw
    decoded stream (think-blocks precede the JSON, so rfind the key)."""
    acc = ""
    positions = []  # (end_index_in_acc, token_index)
    for i, e in enumerate(logprobs):
        tok = e.get("token", "")
        positions.append((len(acc), i))
        acc += tok
    needle = json.dumps(key) + ":"  # '"team":'
    pos = acc.rfind(needle)
    if pos == -1:
        return None
    after = pos + len(needle)
    for start, i in positions:
        if start >= after:
            return i
    return None


def extract_answers(raw_json: dict, questions: dict, logprobs: list | None,
                    temperatures: dict[str, float] | None = None) -> dict:
    """`temperatures` maps question type -> T (from load_temperatures());
    each question's logprobs are divided by its type's T before softmax."""
    temps = temperatures or {}
    answers: dict[str, Any] = {}
    for qid, q in questions.items():
        t = q["type"]
        temp = temperature_for(t, temps)
        val = raw_json.get(qid)
        if val is None:
            answers[qid] = {"type": t, "error": "missing answer"}
            continue
        lp = logprobs or []
        if t == "noul":
            p = probs_for_key(lp, qid, ["yes", "no"], temperature=temp)
            # Cloud Jev contract: the noul field is always P(yes), no matter
            # which answer the model picked. (Was P(chosen) — wrong for "no".)
            ans: dict[str, Any] = {"type": "noul", "noul": p["yes"] if p is not None else None}
            # Keep the model's committed answer too — P(yes) alone cannot
            # reconstruct which side the schema forced (needed by quorum.report).
            ans["answer"] = str(val).strip().lower()
            if p is not None:
                ans["probabilities"] = p
                ans["confidence"] = max(p.values())
            else:
                ans["probabilities"] = None
            answers[qid] = ans
        elif t == "choice":
            opts = list(q["criteria"].keys())
            p = probs_for_key(lp, qid, opts, temperature=temp)
            ans = {"type": "choice", "choice": val}
            if p is not None:
                ans["probabilities"] = p
                ans["confidence"] = p.get(val, 0.0)
            else:
                ans["probabilities"] = None
            answers[qid] = ans
        else:  # score
            levels = q["criteria"]
            idx = int(val)
            cands = [str(i) for i in range(len(levels))]
            p = probs_for_key(lp, qid, cands, temperature=temp)
            ans = {
                "type": "score",
                "score": float(idx),
                "legend": {str(i): v for i, v in enumerate(levels)},
            }
            if p is not None:
                ans["probabilities"] = p
                ans["score"] = sum(i * p[str(i)] for i in range(len(levels)))
                ans["confidence"] = p[str(idx)]
            else:
                ans["probabilities"] = None
            answers[qid] = ans
    return answers
