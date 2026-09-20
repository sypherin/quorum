"""Cloud Jev client: direct calls to TypeSafe's hosted System One API.

Cloud ONLY by design. There is no local/quorum fallback in here, so a run that
says "Jev decided" can only mean api.typesafe.ai decided. Fails loud: after the
retries are spent it raises, it never substitutes an action of its own.
"""
import http.client
import json
import os
import time

HOST = "api.typesafe.ai"
PATH = "/v1/systemone"
MODEL = os.environ.get("JEV_CLOUD_MODEL", "jev-latest")
KEY_FILE = os.path.expanduser("~/.config/typesafe/env")


class JevError(RuntimeError):
    pass


def _load_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k
    for line in open(KEY_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'") if "=" in line else line
    raise JevError(f"no API key in {KEY_FILE}")


class Jev:
    def __init__(self, retries=3, timeout=30):
        self._key = _load_key()
        self._conn = None
        self.retries, self.timeout = retries, timeout
        self.calls = self.failures = self.input_tokens = self.output_tokens = 0
        self.latency_ms_total = 0

    def _post(self, body):
        if self._conn is None:
            self._conn = http.client.HTTPSConnection(HOST, timeout=self.timeout)
        self._conn.request("POST", PATH, body=body, headers={
            "content-type": "application/json", "authorization": f"Bearer {self._key}"})
        r = self._conn.getresponse()
        raw = r.read()
        if r.status != 200:
            raise JevError(f"HTTP {r.status}: {raw[:200]!r}")
        return json.loads(raw), r.getheader("x-typesafe-request-id")

    def ask(self, state, questions):
        """returns {"answers", "model", "usage", "request_id", "latency_ms"}"""
        body = json.dumps({"model": MODEL, "state": json.dumps(state), "questions": questions})
        last = None
        for attempt in range(1, self.retries + 1):
            t0 = time.time()
            try:
                out, rid = self._post(body)
                lat = round((time.time() - t0) * 1000)
                usage = out.get("usage", {})
                self.calls += 1
                self.latency_ms_total += lat
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                return {"answers": out["answers"], "model": out.get("model"), "usage": usage,
                        "request_id": rid, "latency_ms": lat}
            except Exception as e:          # noqa: BLE001 - every failure is counted and re-raised below
                last = e
                self.failures += 1
                print(f"[jev] call failed (try {attempt}/{self.retries}): {type(e).__name__}: {e}", flush=True)
                try:
                    self._conn and self._conn.close()
                finally:
                    self._conn = None
                time.sleep(0.5 * attempt)
        raise JevError(f"cloud Jev unreachable after {self.retries} tries: {last}")

    def totals(self):
        return {"calls": self.calls, "failures": self.failures,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "avg_latency_ms": round(self.latency_ms_total / self.calls) if self.calls else 0}
