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


STATE_FORMATS = ("json", "prose")


def validate_request(body: dict, state_format: str = "json") -> tuple[Any, dict]:
    """state_format decides how an object/array state reaches the model:
    "json" (compact JSON, the default) or "prose" (render_state). A string
    state is always passed through untouched."""
    if not isinstance(body, dict):
        raise BadRequest("body must be an object")
    if state_format not in STATE_FORMATS:
        raise BadRequest(f"state_format must be one of {STATE_FORMATS}")
    state = body.get("state")
    if state is None or state == "":
        raise BadRequest("state is required")
    if not isinstance(state, (str, dict, list)):
        raise BadRequest("state must be string or JSON object/array")
    if isinstance(state, (dict, list)):
        state = render_state(state) if state_format == "prose" else json.dumps(state, ensure_ascii=False)
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


def _scalar(v: Any) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return "none"
    return json.dumps(v, ensure_ascii=False)  # true/false, numbers


def render_state(obj: Any, indent: int = 0) -> str:
    """Structured state as indented `key: value` lines, the way a person would
    write it down. Nothing is dropped or renamed: every key and value in the
    JSON appears once, so the only change is the syntax the model reads.
    Multi-line strings keep their line breaks, indented under their key."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        if not obj:
            return pad + "(empty)"
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(render_state(v, indent + 1))
            else:
                text = "(empty)" if isinstance(v, (dict, list)) else _scalar(v)
                first, *rest = text.split("\n")
                lines.append(f"{pad}{k}: {first}")
                lines.extend(f"{pad}  {r}" for r in rest)
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return pad + "(empty)"
        for v in obj:
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}-")
                lines.append(render_state(v, indent + 1))
            else:
                text = "(empty)" if isinstance(v, (dict, list)) else _scalar(v)
                first, *rest = text.split("\n")
                lines.append(f"{pad}- {first}")
                lines.extend(f"{pad}  {r}" for r in rest)
        return "\n".join(lines)
    return pad + _scalar(obj)


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
# Calibration: temperature scaling (any type) and Platt scaling (noul)
# ---------------------------------------------------------------------------

def _calibration_path() -> Path:
    path = os.environ.get("QUORUM_CALIBRATION")
    return Path(path) if path else Path(__file__).with_name("calibration.json")


def load_calibration() -> dict:
    """{"temperatures": {type: T}, "platt": {type: {"a": .., "b": ..}}} from
    calibration.json ($QUORUM_CALIBRATION, else next to this module).
    Missing file => both empty => answers pass through uncalibrated."""
    path = _calibration_path()
    if not path.exists():
        return {"temperatures": {}, "platt": {}}
    data = json.loads(path.read_text())
    return {
        "temperatures": {k: float(v) for k, v in data.get("temperatures", {}).items()},
        "platt": {k: {"a": float(v["a"]), "b": float(v["b"])} for k, v in data.get("platt", {}).items()},
    }


def load_temperatures() -> dict[str, float]:
    """Per-question-type temperatures only (kept for older callers)."""
    return load_calibration()["temperatures"]


def temperature_for(kind: str, temps: dict[str, float]) -> float:
    return temps.get(kind, temps.get("default", 1.0))


PLATT_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, PLATT_EPS), 1 - PLATT_EPS)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _platt_loss(xs: list[float], y: list[bool], a: float, b: float, l2: float) -> float:
    loss = 0.5 * l2 * ((a - 1) ** 2 + b * b)
    for x, t in zip(xs, y):
        z = a * x + b
        # log(1 + e^z) - t*z, computed stably
        loss += (z if z > 0 else 0.0) + math.log1p(math.exp(-abs(z))) - (z if t else 0.0)
    return loss


def platt_fit(p_yes: list[float], y: list[bool], l2: float = 1e-3, iters: int = 100) -> tuple[float, float]:
    """Fit P(y) = sigmoid(a * logit(p) + b) by damped Newton (backtracking line
    search: a plain Newton step overshoots when the starting predictions are
    saturated). L2 on (a-1, b) keeps a degenerate fit near identity.
    Returns (a, b); identity is (1, 0).

    Temperature scaling is the special case b = 0, a = 1/T: Platt can also move
    the midpoint, which a yes/no judge biased toward one answer needs."""
    xs = [_logit(p) for p in p_yes]
    a, b = 1.0, 0.0
    cur = _platt_loss(xs, y, a, b, l2)
    for _ in range(iters):
        ga, gb = l2 * (a - 1), l2 * b
        haa, hab, hbb = l2, 0.0, l2
        for x, t in zip(xs, y):
            s = _sigmoid(a * x + b)
            r = s - (1.0 if t else 0.0)
            w = s * (1 - s)
            ga += r * x
            gb += r
            haa += w * x * x
            hab += w * x
            hbb += w
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = 1.0
        while step > 1e-6:
            na, nb = a - step * da, b - step * db
            new = _platt_loss(xs, y, na, nb, l2)
            if new <= cur:
                break
            step *= 0.5
        else:
            break
        moved = abs(na - a) + abs(nb - b)
        a, b, cur = na, nb, new
        if moved < 1e-10:
            break
    return a, b


def platt_apply(p_yes: float, a: float, b: float) -> float:
    return _sigmoid(a * _logit(p_yes) + b)


def temper(probs: dict[str, float], t: float) -> dict[str, float]:
    """softmax(log p / T) over the same keys; zeros stay zero. Identical to
    dividing the logprobs by T before the softmax (the constant cancels)."""
    if t == 1.0:
        return dict(probs)
    logs = {k: math.log(v) / t for k, v in probs.items() if v > 0}
    if not logs:
        return dict(probs)
    m = max(logs.values())  # log-space: v ** (1/T) underflows to all-zero at small T
    w = {k: math.exp(logs[k] - m) if k in logs else 0.0 for k in probs}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def apply_calibration(raw_answers: dict, cal: dict | None) -> dict:
    """Calibrated copy of extract_answers() output. The committed answer never
    changes, only the probabilities and the fields derived from them
    (noul/confidence/score). Per type: Platt when calibration.json has one
    (noul only: it supersedes T there), else temperature T, else passthrough.
    `raw_answers` is not modified, so the caller can log both."""
    cal = cal or {}
    temps, platt = cal.get("temperatures") or {}, cal.get("platt") or {}
    out: dict[str, Any] = {}
    for qid, ans in raw_answers.items():
        ans = dict(ans)
        out[qid] = ans
        probs = ans.get("probabilities")
        kind = ans.get("type")
        if not probs or "error" in ans:
            continue
        if kind == "noul" and kind in platt:
            py = platt_apply(probs["yes"], platt[kind]["a"], platt[kind]["b"])
            probs = {"yes": py, "no": 1.0 - py}
        else:
            probs = temper(probs, temperature_for(kind, temps))
        ans["probabilities"] = probs
        if kind == "noul":
            ans["noul"] = probs["yes"]
            ans["confidence"] = max(probs.values())
        elif kind == "choice":
            ans["confidence"] = probs.get(ans["choice"], 0.0)
        elif kind == "score":
            ans["score"] = sum(int(k) * v for k, v in probs.items())
            ans["confidence"] = probs.get(str(ans["level"]), 0.0)
    return out


def _candidate_matches(tok_text: str, candidates: list[str], prefer: str | None = None) -> list[str]:
    """Candidates a token could be the start of (token may carry quotes/punct).

    Exact match wins outright. A prefix that fits exactly one candidate is used.
    A prefix that fits SEVERAL candidates (e.g. "run" starts both "run_right" and
    "run_left") is ambiguous: on its own it is dropped, because attributing its
    logprob to any one of them would be wrong. But when `prefer` (the answer the
    model actually committed to) is among those candidates, the ambiguity is
    resolved — the token's mass went to the continuation the model emitted, so it
    is credited to `prefer`. This is exact when P(suffix | prefix) ~= 1, which is
    the norm for an enum decoded at temperature 0; for shared-prefix options it is
    the difference between the chosen option reading its true probability and
    reading 0.0."""
    t = tok_text.strip().strip('"').lstrip()
    if not t:
        return []
    exact = [c for c in candidates if c == t]
    if exact:
        return exact
    pref = [c for c in candidates if c.startswith(t[: len(c)])]
    if len(pref) > 1 and prefer in pref:
        return [prefer]
    return pref


def probs_at_position(logprobs: list, value_token_index: int, candidates: list[str],
                      temperature: float = 1.0, prefer: str | None = None) -> dict[str, float] | None:
    """Distribution over candidates from top_logprobs at the value's token position.

    `temperature` scales logprobs before normalization (softmax(lp / T));
    T=1 is a passthrough. `prefer` (the committed answer) resolves an otherwise
    ambiguous shared-prefix token in favour of the option the model emitted."""
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
        matches = _candidate_matches(tok, candidates, prefer=prefer)
        if len(matches) != 1:
            continue  # no match, or ambiguous prefix we cannot resolve — never misattribute
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
                  temperature: float = 1.0, prefer: str | None = None) -> dict[str, float] | None:
    """Distribution over candidates at the value position for `key`.

    The JSON emitter may put spacer tokens (' \"', ' ') between the colon and
    the value, and the value may be quoted/merged — walk forward from the key
    and accept the first position that distinguishes >=2 candidates. `prefer`
    (the committed answer) resolves shared-prefix ambiguity toward it."""
    start = find_value_token(logprobs, key)
    if start is None:
        return None
    for i in range(start, min(start + max_walk, len(logprobs))):
        p = probs_at_position(logprobs, i, candidates, temperature, prefer=prefer)
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
            p = probs_for_key(lp, qid, ["yes", "no"], temperature=temp,
                              prefer=str(val).strip().lower())
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
            p = probs_for_key(lp, qid, opts, temperature=temp, prefer=str(val))
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
            p = probs_for_key(lp, qid, cands, temperature=temp, prefer=str(idx))
            ans = {
                "type": "score",
                "score": float(idx),
                "level": idx,  # the committed level; "score" becomes the expectation below
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
