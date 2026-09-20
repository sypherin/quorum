"""Local quorum client: same interface as jev_client.Jev, but talks to the
local quorum shim (POST /v1/systemone on 127.0.0.1:8017) instead of the hosted
TypeSafe API.

Drop-in for Jev: same `.ask(state, questions)` return shape and the same
`.totals()`, so play.py only has to pick which client to build. Nothing leaves
the machine. quorum speaks the identical System One contract, so the state and
questions are byte-for-byte what cloud Jev received.

reasoning=True turns on quorum's bounded chain-of-thought (a short rationale
before each answer): ~3x slower per call, materially better noul/choice quality.
Since the emulator is paused while the model thinks, that extra latency costs
wall-clock only, never a game frame -- so it is on by default here.
"""
import http.client
import json
import os
import time
from urllib.parse import urlparse


class QuorumError(RuntimeError):
    pass


class Quorum:
    def __init__(self, url=None, reasoning=True, retries=3, timeout=120):
        url = url or os.environ.get("QUORUM_URL", "http://127.0.0.1:8017/v1/systemone")
        u = urlparse(url)
        self.host, self.port, self.path = u.hostname, u.port or 80, u.path or "/v1/systemone"
        self.reasoning = reasoning
        self.retries, self.timeout = retries, timeout
        self._conn = None
        self.calls = self.failures = self.input_tokens = self.output_tokens = 0
        self.latency_ms_total = 0

    def _post(self, body):
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        self._conn.request("POST", self.path, body=body,
                           headers={"content-type": "application/json"})
        r = self._conn.getresponse()
        raw = r.read()
        if r.status != 200:
            raise QuorumError(f"HTTP {r.status}: {raw[:200]!r}")
        return json.loads(raw)

    def ask(self, state, questions):
        """returns {"answers", "model", "usage", "request_id", "latency_ms"} -- the
        same shape jev_client.Jev.ask returns, so callers cannot tell them apart."""
        payload = {"state": json.dumps(state), "questions": questions}
        if self.reasoning:
            payload["reasoning"] = True
        body = json.dumps(payload)
        last = None
        for attempt in range(1, self.retries + 1):
            t0 = time.time()
            try:
                out = self._post(body)
                lat = round((time.time() - t0) * 1000)
                usage = out.get("usage") or {}
                # quorum reports input_tokens/output_tokens directly (may be null)
                self.calls += 1
                self.latency_ms_total += lat
                self.input_tokens += usage.get("input_tokens") or 0
                self.output_tokens += usage.get("output_tokens") or 0
                return {
                    "answers": out["answers"],
                    "model": out.get("model", "quorum-local"),
                    "usage": {"input_tokens": usage.get("input_tokens"),
                              "output_tokens": usage.get("output_tokens")},
                    "request_id": f'local:{out.get("mode", "?")}',
                    "latency_ms": out.get("latency_ms", lat),
                }
            except Exception as e:  # noqa: BLE001 - counted, then re-raised after retries
                last = e
                self.failures += 1
                print(f"[quorum] call failed (try {attempt}/{self.retries}): "
                      f"{type(e).__name__}: {e}", flush=True)
                try:
                    self._conn and self._conn.close()
                finally:
                    self._conn = None
                time.sleep(0.5 * attempt)
        raise QuorumError(f"local quorum unreachable after {self.retries} tries: {last}")

    def totals(self):
        return {"calls": self.calls, "failures": self.failures,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "avg_latency_ms": round(self.latency_ms_total / self.calls) if self.calls else 0}
