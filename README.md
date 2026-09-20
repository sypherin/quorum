# quorum

A local, private re-implementation of the [TypeSafe AI](https://typesafe.ai)
SystemOne request contract -- one state, N typed judgment questions,
probabilities back -- running on your own hardware, plus a calibration harness
that turns logged judgments into a training-adjacent corpus.

Point it at any llama-server-compatible endpoint. It answers `noul`, `choice`
and `score` questions by constraining the model's output and reading token
logprobs, so a "probability" falls out of the decode instead of being parsed
out of prose. No account, no billing, nothing leaves the machine.

## Why Jev, and why this exists

[Jev](https://typesafe.ai) is TypeSafe's System One model: small, fast
language-model judgments with *trained* calibration, trained with RLCD against
human judgment references. It's the idea that made this project worth doing --
judgments as typed primitives (`noul` / `choice` / `score`) that code can
combine, rather than prompt-and-parse string guessing.

We ran Jev in production for months, and hit three walls that are fine for a
demo but not for an always-on gate:

- **cost per call.** Our judgment hooks fire on every prompt, every file diff,
  every completion -- thousands of small calls a week. Per-call pricing turns a
  guardrail into a meter.
- **privacy.** The states we judge are unreleased code diffs, client names,
  draft emails. They shouldn't transit an API to be graded.
- **no knob on the model itself.** When we measured the local-vs-cloud gap on a
  trace-classification benchmark, what we needed was to *retrain on our own
  labeled judgments* -- something a hosted model can't offer.

So: keep the wire contract, swap the brain. `quorum` speaks the same
`POST /v1/systemone` shape, runs against a local 4B model, and adds the piece
the hosted API doesn't give you -- a calibration loop over your own corpus.

The name: a *quorum* is multiple independent judgments called over one state.
That's literally what each request does.

## What it is / isn't

- **Is:** a drop-in ASGI shim (`serve.py`) over any llama-server-compatible
  upstream, with structured-output decoding, an opt-in chain-of-thought mode,
  per-question temperature calibration, and a client (`gate_client.ask`) that
  logs every call.
- **Is not:** a trained System One model. The probabilities are constrained-LLM
  logprobs -- good for *ranking* ("diff A riskier than that one"), not certified
  calibrated numbers. `confidence` is the winning option's probability, a
  concentration proxy. This is the honest gap vs Jev and we say so up front.
  `quorum.calibrate` narrows the gap; it does not erase it.

## Quick start

```bash
# serve on :8017 against a llama-server chat-completions upstream on :8005
uvicorn quorum.serve:app --host 127.0.0.1 --port 8017
# env: QUORUM_UPSTREAM, QUORUM_MODEL_ALIAS, QUORUM_PORT, QUORUM_COT,
#      QUORUM_LOG, QUORUM_CALIBRATION, QUORUM_MAX_STATE_CHARS
```

One request, three primitives:

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

From Python (stdlib only):

```python
from quorum.gate_client import ask
r = ask({"review_type": "outbound email", "situation": "...draft text..."},
        {"regrettable": {"type": "noul", "instructions": "Would we regret sending this?"}},
        caller="my-hook")
print(r["probs"]["regrettable"])   # float 0..1
```

## The calibration-corpus story

This is the part the hosted API can't do, and the reason the project exists
locally.

Every call through `gate_client` is appended to a JSONL log (`QUORUM_LOG`):
state, questions, probabilities, latency, caller. Run long enough and the log
becomes what an API bill never is -- **a labeled record of the judgments your
system actually made**, with outcomes you can fill in later.

Four things grow out of that corpus:

1. **Measure the gap.** `python3 -m quorum.report` audits the log itself --
   per-question Brier, log-loss, 10-bin ECE, a reliability table and a
   confident-error list (high-prob calls that were wrong), with the
   majority-class floor printed next to decision accuracy so a 95%-one-class
   question doesn't look like a win. Borrowed from the kev/Jev eval harnesses:
   score calibration, not just accuracy. Needs labels -- that's the next step.
2. **Label what mattered.** `python3 -m quorum.label` walks the log
   newest-first, one judgment per screen; Enter agrees with quorum, `n`
   overrides per question. Verdicts merge into the log as a `labels` dict --
   the bridge between "logged" and "calibratable".
3. **Runtime calibration.** `python3 -m quorum.calibrate labeled.jsonl` fits a
   per-question-type temperature (NLL minimisation over the logged probs) and
   writes `calibration.json`, which `serve.py` picks up automatically. Raw
   logprob → tuned logprob, no retraining. (`bench/fit_calibration.py` goes
   further -- Platt maps fitted against the cloud-Jev teacher with 5-fold CV;
   `bench/calibration_teacher.json` is the honest result: vs cloud probs,
   ECE 0.23 → 0.13 fitted, but vs gold labels the local 4B still doesn't
   separate classes on every task. That gap is what the corpus loop is for.)
4. **A fine-tuning set.** Our own judgment gate -- a Qwen3-4B LoRA trained on
   647 reviewed situation→verdict pairs mined from months of agent logs --
   follows the same loop: log judgments, label the ones that mattered, train,
   probe, ship. The corpus *is* the product; the weights are just its current
   compiled form ([AltronisSG on Hugging Face](https://huggingface.co/AltronisSG),
   Apache-2.0).

The loop, end to end:

```
agent work → hooks ask quorum → judgments logged → humans label what mattered
   → calibrate (T per question type)  →  better probabilities today
   → fine-tune (LoRA on the labeled set) → better base model tomorrow
```

## Tests

```bash
python -m pytest tests/ -q
```

## Credits & license

- The System One framing, the `noul`/`choice`/`score` primitives, and the
  goal of calibrated language-model judgment are TypeSafe AI's -- see
  [typesafe.ai](https://typesafe.ai) and their Jev / RLCD work. "SystemOne"
  and "Jev" are their concepts; this project is an independent local
  re-implementation of the request contract, not their software.
- Base model served here: Qwen3-4B (Apache-2.0) via llama.cpp/llama-server.
- quorum code: Apache-2.0 -- see [LICENSE](LICENSE).
