"""quorum serve — bare-ASGI shim exposing POST /v1/systemone backed by a
local llama-server (default: judgment-gate 4B on 127.0.0.1:8005).

No FastAPI dependency. Run:
  QUORUM_PORT=8017 uvicorn quorum.serve:app --host 127.0.0.1 --port 8017

Calibration: calibration.json next to the package (or $QUORUM_CALIBRATION)
holds per-type temperatures and/or a noul Platt fit, applied to the raw
distribution after extraction (core.apply_calibration). No file => raw
passthrough. Fit with `python3 -m quorum.calibrate`.

Per-request options (each also has an env default):
  "reasoning"/"cot": bool   bounded rationale before each answer (QUORUM_COT)
  "fanout": bool            one upstream call per question, all sharing the
                            system+STATE prefix so llama-server reuses its
                            prompt cache; answers cannot bleed into each other
                            (QUORUM_FANOUT; QUORUM_FANOUT_CONCURRENCY>1 primes
                            the cache with the first question, then runs the
                            rest concurrently for a --parallel N upstream)
  "state_format": "json"|"prose"   how an object state is shown (QUORUM_STATE_FORMAT)

Cache: identical requests (same state, questions and options) are answered
from an in-process LRU of raw answers (QUORUM_CACHE_SIZE, default 256, 0 = off;
25.6% of the 8,983 logged calls on 2026-09-23 repeated an earlier one).
Calibration is applied on every hit, so a new calibration.json takes effect
immediately.

Logging: every successful judgment appends one JSONL record (state,
questions, calibrated answers, raw_answers, calibration used, label: null) to
$QUORUM_LOG (default: log/judgments.jsonl under the repo root; QUORUM_LOG=off
disables). Best-effort — never fails a request. calibrate.py fits on
raw_answers, never on the calibrated ones.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import core

UPSTREAM = os.environ.get("QUORUM_UPSTREAM", "http://127.0.0.1:8005")
MODEL_ALIAS = os.environ.get("QUORUM_MODEL_ALIAS", "quorum-local-4b")
# Chain-of-thought default: QUORUM_COT=1 turns it on globally; a request's
# "reasoning": true|false always overrides. Off by default => zero change for
# existing callers (direct single-pass, ~1s). CoT ~3x slower but ~cloud-grade noul.
COT_DEFAULT = os.environ.get("QUORUM_COT", "").strip().lower() in ("1", "true", "yes", "on")
MAX_STATE_CHARS = int(os.environ.get("QUORUM_MAX_STATE_CHARS", "60000"))
# top_logprobs count requested from the upstream. Some engines (e.g. halogen)
# return only the chosen token's logprob and reject top_logprobs — set
# QUORUM_TOP_LOGPROBS=0 for those. Answers still come from the constrained JSON;
# only the probability distribution (and thus confidence) goes unavailable.
TOP_LOGPROBS = int(os.environ.get("QUORUM_TOP_LOGPROBS", "12"))
DEFAULT_LOG = Path(__file__).resolve().parent.parent / "log" / "judgments.jsonl"
_TRUE = ("1", "true", "yes", "on")
FANOUT_DEFAULT = os.environ.get("QUORUM_FANOUT", "").strip().lower() in _TRUE
FANOUT_CONCURRENCY = max(1, int(os.environ.get("QUORUM_FANOUT_CONCURRENCY", "1")))
STATE_FORMAT_DEFAULT = os.environ.get("QUORUM_STATE_FORMAT", "json").strip().lower() or "json"
CACHE_SIZE = int(os.environ.get("QUORUM_CACHE_SIZE", "256"))
_cache: OrderedDict[str, dict] = OrderedDict()


async def _read_body(receive) -> bytes:
    buf = b""
    while True:
        msg = await receive()
        if msg["type"] == "http.request":
            buf += msg.get("body", b"")
            if not msg.get("more_body"):
                return buf
        elif msg["type"] == "http.disconnect":
            return b""


async def _send_response(send, status: int, body: bytes):
    await send({"type": "http.response.start", "status": status, "headers": [[b"content-type", b"application/json"]]})
    await send({"type": "http.response.body", "body": body})


async def call_upstream(state: str, questions: dict, cot: bool = False) -> dict:
    messages = core.build_messages(state, questions, cot=cot)
    schema = core.build_schema(questions, cot=cot)
    # CoT needs headroom for the bounded rationale per question; direct mode stays lean.
    # worst case a 200-char rationale is ~200 tokens (one per char for junk/escapes), so budget for it:
    # a truncated rationale means the answer token is never emitted and the JSON cannot be parsed.
    max_tokens = min(2048, 128 + 256 * len(questions)) if cot else 256
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    if TOP_LOGPROBS > 0:
        # logprobs (and the distribution built from top_logprobs) are only
        # requested when wanted. Some greedy-decoding engines reject logprobs at
        # temperature 0 outright, so in no-distribution mode we omit it entirely.
        payload["logprobs"] = True
        payload["top_logprobs"] = TOP_LOGPROBS
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(UPSTREAM + "/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()


def _log_path() -> Path | None:
    raw = os.environ.get("QUORUM_LOG")
    if raw is None:
        return DEFAULT_LOG
    if raw.lower() in ("off", "0", "none", ""):
        return None
    return Path(raw)


def log_record(state: str, questions: dict, answers: dict, raw_answers: dict | None = None,
               calibration: dict | None = None, extra: dict | None = None) -> None:
    """Best-effort append of one judgment record to the JSONL log.

    Never raises: logging must not break the judgment path. `answers` is what
    the caller was served (calibrated); `raw_answers` is the uncalibrated
    extraction, which is what quorum.calibrate must fit on — fitting on the
    served answers would calibrate an already-calibrated distribution.
    """
    path = _log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "endpoint": UPSTREAM,
            "state": state,
            "questions": questions,
            "answers": answers,
            "raw_answers": raw_answers,
            "calibration": calibration,
            **(extra or {}),
            "label": None,  # fill in later, per record or via a labeling pass
        }
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


class UpstreamFailure(Exception):
    def __init__(self, detail: dict):
        super().__init__(detail.get("detail"))
        self.detail = detail


def _usage(resp: dict) -> dict:
    u = resp.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = (resp.get("timings") or {}).get("cache_n")  # llama-server
    return {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
            "cached_tokens": cached}


async def _one_pass(state: str, questions: dict, cot: bool) -> tuple[dict, dict]:
    """One upstream call for `questions` -> (raw answers, pass meta).
    A CoT pass whose JSON is unparseable degrades to one direct pass and says so."""
    try:
        resp = await call_upstream(state, questions, cot=cot)
    except Exception as e:  # noqa: BLE001 — surface every upstream failure
        raise UpstreamFailure({"detail": f"upstream error: {e}"}) from e
    fallback = None
    content = None
    try:
        ch = resp["choices"][0]
        content = ch["message"]["content"]
        raw = json.loads(content)
    except Exception as e:  # noqa: BLE001
        if not cot:
            raise UpstreamFailure({"detail": f"unparseable upstream content: {e}", "raw": str(content)[:400]}) from e
        # A runaway rationale truncated the JSON before the answer. Decoding is greedy, so a retry
        # would fail identically: degrade to a direct (no-rationale) pass and SAY so in the response.
        fallback = f"cot output unparseable ({e}); answered in direct mode"
        print(f"WARN quorum: {fallback} [{', '.join(questions)}]", flush=True)
        try:
            resp = await call_upstream(state, questions, cot=False)
            ch = resp["choices"][0]
            content = ch["message"]["content"]
            raw = json.loads(content)
        except Exception as e2:  # noqa: BLE001
            raise UpstreamFailure({"detail": f"unparseable upstream content (cot and direct): {e2}",
                                   "raw": str(content)[:400]}) from e2
    try:
        logprobs = ch.get("logprobs", {}).get("content") or None
    except AttributeError:
        logprobs = None
    answers = core.extract_answers(raw, questions, logprobs)  # raw: calibration comes after
    return answers, {"fallback": fallback, "usage": _usage(resp), "logprobs": bool(logprobs), "calls": 2 if fallback else 1}


async def _judge(state: str, questions: dict, cot: bool, fanout: bool) -> tuple[dict, dict]:
    """Raw answers for every question, one call per question when fanout."""
    if not fanout or len(questions) == 1:
        answers, meta = await _one_pass(state, questions, cot)
        passes = [(list(questions), meta)]
    else:
        items = list(questions.items())

        async def one(qid, q):
            return [qid], await _one_pass(state, {qid: q}, cot)

        first = await one(*items[0])  # alone: it leaves the shared prefix in the upstream cache
        sem = asyncio.Semaphore(FANOUT_CONCURRENCY)

        async def bounded(qid, q):
            async with sem:
                return await one(qid, q)

        rest = await asyncio.gather(*(bounded(qid, q) for qid, q in items[1:]))
        answers = {}
        passes = []
        for qids, (ans, meta) in [first, *rest]:
            answers.update(ans)
            passes.append((qids, meta))

    def total(key):
        vals = [m["usage"][key] for _, m in passes]
        return sum(vals) if all(v is not None for v in vals) else None

    fell_back = [q for qids, m in passes if m["fallback"] for q in qids]
    with_lp = sum(m["logprobs"] for _, m in passes)
    meta = {
        "usage": {k: total(k) for k in ("input_tokens", "output_tokens", "cached_tokens")},
        "calls": sum(m["calls"] for _, m in passes),
        "prob_method": "logprobs" if with_lp == len(passes) else ("partial" if with_lp else "unavailable"),
        "mode": ("direct-fallback" if fell_back else "cot") if cot else "direct",
    }
    if fell_back:
        meta["degraded"] = (passes[0][1]["fallback"] if len(passes) == 1 else
                            f"cot output unparseable for {', '.join(fell_back)}; those answered in direct mode")
    return answers, meta


def _cache_key(state: str, questions: dict, cot: bool, fanout: bool) -> str:
    blob = json.dumps([UPSTREAM, TOP_LOGPROBS, cot, fanout, state, questions], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _flag(body: dict, *keys: str, default: bool) -> bool:
    for k in keys:
        if body.get(k) is not None:
            return bool(body[k])
    return default


async def handle_systemone(body: dict) -> tuple[int, dict]:
    fmt = body.get("state_format", STATE_FORMAT_DEFAULT) if isinstance(body, dict) else STATE_FORMAT_DEFAULT
    try:
        state, questions = core.validate_request(body, state_format=fmt)
    except core.BadRequest as e:
        return 400, {"detail": str(e)}
    if len(state) > MAX_STATE_CHARS:
        return 400, {"detail": f"state exceeds {MAX_STATE_CHARS} chars"}
    # opt-in reasoning / fan-out: the request wins over the env default
    cot = _flag(body, "reasoning", "cot", default=COT_DEFAULT)
    fanout = _flag(body, "fanout", default=FANOUT_DEFAULT)
    t0 = time.time()
    key = _cache_key(state, questions, cot, fanout) if CACHE_SIZE > 0 else None
    hit = _cache.get(key) if key else None
    if hit is not None:
        _cache.move_to_end(key)
        raw_answers, meta = hit["answers"], dict(hit["meta"])
        meta["usage"] = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
        meta["calls"] = 0
    else:
        try:
            raw_answers, meta = await _judge(state, questions, cot, fanout)
        except UpstreamFailure as e:
            return 502, e.detail
        if key:
            _cache[key] = {"answers": raw_answers, "meta": meta}
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
    cal = core.load_calibration()
    answers = core.apply_calibration(raw_answers, cal)
    out = {
        "model": MODEL_ALIAS,
        "answers": answers,
        "usage": meta["usage"],
        "latency_ms": int((time.time() - t0) * 1000),
        "prob_method": meta["prob_method"],
        "mode": meta["mode"],
        "fanout": fanout,
        "calls": meta["calls"],
        "cached": hit is not None,
    }
    if meta.get("degraded"):
        out["degraded"] = meta["degraded"]
    log_record(state, questions, answers, raw_answers=raw_answers, calibration=cal,
               extra={"mode": meta["mode"], "fanout": fanout, "cached": hit is not None})
    return 200, out


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    path = scope["path"]
    if scope["method"] == "GET" and path in ("/healthz", "/"):
        await _send_response(send, 200, json.dumps({"ok": True, "upstream": UPSTREAM}).encode())
        return
    if scope["method"] == "POST" and path == "/v1/systemone":
        body = await _read_body(receive)
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            await _send_response(send, 400, b'{"detail":"invalid json"}')
            return
        status, payload = await handle_systemone(parsed)
        await _send_response(send, status, json.dumps(payload).encode())
        return
    await _send_response(send, 404, b'{"detail":"not found"}')
