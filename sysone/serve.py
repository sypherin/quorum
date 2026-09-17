"""sysone serve — bare-ASGI shim exposing POST /v1/systemone backed by a
local llama-server (default: judgment-gate 4B on 127.0.0.1:8005).

No FastAPI dependency. Run:
  SYSONE_PORT=8017 uvicorn sysone.serve:app --host 127.0.0.1 --port 8017

Calibration: per-question-type temperatures from calibration.json next to
the package (or $SYSONE_CALIBRATION) are applied to logprobs before softmax.
No file => T=1 passthrough. Fit with `python3 -m sysone.calibrate`.

Logging: every successful judgment appends one JSONL record (state,
questions, answers, label: null) to $SYSONE_LOG (default: log/judgments.jsonl
under the repo root; SYSONE_LOG=off disables). Best-effort — never fails a
request. Records are the raw material for sysone.calibrate once labeled.
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

UPSTREAM = os.environ.get("SYSONE_UPSTREAM", "http://127.0.0.1:8005")
MODEL_ALIAS = os.environ.get("SYSONE_MODEL_ALIAS", "sysone-local-4b")
MAX_STATE_CHARS = int(os.environ.get("SYSONE_MAX_STATE_CHARS", "60000"))
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


async def call_upstream(state: str, questions: dict) -> dict:
    messages = core.build_messages(state, questions)
    schema = core.build_schema(questions)
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_schema", "json_schema": schema},
        "logprobs": True,
        "top_logprobs": 12,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(UPSTREAM + "/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()


def _log_path() -> Path | None:
    raw = os.environ.get("SYSONE_LOG")
    if raw is None:
        return DEFAULT_LOG
    if raw.lower() in ("off", "0", "none", ""):
        return None
    return Path(raw)


def log_record(state: str, questions: dict, answers: dict) -> None:
    """Best-effort append of one judgment record to the JSONL log.

    Never raises: logging must not break the judgment path. Records carry
    state + questions + raw answers so outcomes can be labeled later and fed
    to sysone.calibrate.
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
    t0 = time.time()
    try:
        resp = await call_upstream(state, questions)
    except Exception as e:  # noqa: BLE001 — surface every upstream failure
        return 502, {"detail": f"upstream error: {e}"}
    try:
        ch = resp["choices"][0]
        content = ch["message"]["content"]
        raw = json.loads(content)
    except Exception as e:  # noqa: BLE001
        return 502, {"detail": f"unparseable upstream content: {e}", "raw": str(content)[:400]}
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
    }
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
