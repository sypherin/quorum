"""quorum serve — bare-ASGI shim exposing POST /v1/systemone backed by a
local llama-server (default: judgment-gate 4B on 127.0.0.1:8005).

No FastAPI dependency. Run:
  QUORUM_PORT=8017 uvicorn quorum.serve:app --host 127.0.0.1 --port 8017

Calibration: per-question-type temperatures from calibration.json next to
the package (or $QUORUM_CALIBRATION) are applied to logprobs before softmax.
No file => T=1 passthrough. Fit with `python3 -m quorum.calibrate`.

Logging: every successful judgment appends one JSONL record (state,
questions, answers, label: null) to $QUORUM_LOG (default: log/judgments.jsonl
under the repo root; QUORUM_LOG=off disables). Best-effort — never fails a
request. Records are the raw material for quorum.calibrate once labeled.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
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


def log_record(state: str, questions: dict, answers: dict) -> None:
    """Best-effort append of one judgment record to the JSONL log.

    Never raises: logging must not break the judgment path. Records carry
    state + questions + raw answers so outcomes can be labeled later and fed
    to quorum.calibrate.
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
            "label": None,  # fill in later, per record or via a labeling pass
        }
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


async def handle_systemone(body: dict) -> tuple[int, dict]:
    try:
        state, questions = core.validate_request(body)
    except core.BadRequest as e:
        return 400, {"detail": str(e)}
    if len(state) > MAX_STATE_CHARS:
        return 400, {"detail": f"state exceeds {MAX_STATE_CHARS} chars"}
    # opt-in reasoning: request "reasoning" (or "cot") wins over the env default
    flag = body.get("reasoning", body.get("cot"))
    cot = bool(flag) if flag is not None else COT_DEFAULT
    t0 = time.time()
    try:
        resp = await call_upstream(state, questions, cot=cot)
    except Exception as e:  # noqa: BLE001 — surface every upstream failure
        return 502, {"detail": f"upstream error: {e}"}
    fallback = None
    try:
        ch = resp["choices"][0]
        content = ch["message"]["content"]
        raw = json.loads(content)
    except Exception as e:  # noqa: BLE001
        if not cot:
            return 502, {"detail": f"unparseable upstream content: {e}", "raw": str(content)[:400]}
        # A runaway rationale truncated the JSON before the answer. Decoding is greedy, so a retry
        # would fail identically: degrade to a direct (no-rationale) pass and SAY so in the response.
        fallback = f"cot output unparseable ({e}); answered in direct mode"
        print(f"WARN quorum: {fallback}", flush=True)
        try:
            resp = await call_upstream(state, questions, cot=False)
            ch = resp["choices"][0]
            content = ch["message"]["content"]
            raw = json.loads(content)
            cot = False
        except Exception as e2:  # noqa: BLE001
            return 502, {"detail": f"unparseable upstream content (cot and direct): {e2}", "raw": str(content)[:400]}
    logprobs = None
    try:
        logprobs = ch.get("logprobs", {}).get("content") or None
    except AttributeError:
        logprobs = None
    answers = core.extract_answers(raw, questions, logprobs, core.load_temperatures())
    usage = resp.get("usage", {})
    out = {
        "model": MODEL_ALIAS,
        "answers": answers,
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        },
        "latency_ms": int((time.time() - t0) * 1000),
        "prob_method": "logprobs" if logprobs else "unavailable",
        "mode": "cot" if cot else ("direct-fallback" if fallback else "direct"),
    }
    if fallback:
        out["degraded"] = fallback
    log_record(state, questions, answers)
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
