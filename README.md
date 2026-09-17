# sysone — local TypeSafe SystemOne shim

Drop-in-ish replication of `POST https://api.typesafe.ai/v1/systemone` running on
the box's own llama-server (default: the 4B judgment-gate on 127.0.0.1:8005).

Same request shape (state + questions) and same answer shapes for the three
primitives (noul / choice / score), with probabilities extracted from
llama-server token logprobs. No data leaves the machine.

## Differences vs typesafe (honest)

- probabilities are constrained-LLM logprobs, **not** trained calibration —
  treat them as ranking signal, not ground truth
- `confidence` = winning option's probability (concentration proxy)
- no account, no billing; usage/latency reported per call

## Run

```bash
cd ~/dev/sysone
SYSONE_PORT=8017 uvicorn sysone.serve:app --host 127.0.0.1 --port 8017
# optional: SYSONE_UPSTREAM=http://127.0.0.1:8005 SYSONE_MODEL_ALIAS=...
```

## Call

```bash
curl -s 127.0.0.1:8017/v1/systemone -H 'content-type: application/json' -d '{
  "state": "any text or JSON object",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does this express urgency?"},
    "team":   {"type": "choice", "instructions": "Which team",
               "criteria": {"billing": "payments", "technical": "bugs"}},
    "mood":   {"type": "score", "instructions": "Frustration level",
               "criteria": ["calm: factual", "annoyed: irritated", "angry: hostile"]}
  }
}'
```

## Tests

```bash
cd ~/dev/sysone && python -m pytest tests/ -q
```
