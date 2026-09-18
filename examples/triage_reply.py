#!/usr/bin/env python3
"""triage_reply.py — outbound-sales reply triage against the local quorum API.

Asks THREE questions in ONE request to quorum (127.0.0.1:8017) over the
reply email:
  Choice: intent (interested / not_now / not_a_fit / out_of_office / unsubscribe)
  Score:  heat 0-3 with concrete level descriptions
  Noul:   does the reply mention budget or timeline?

Usage:
    python3 triage_reply.py --demo                 # run the 3 built-in samples
    python3 triage_reply.py "email text here"      # triage one reply
    cat email.txt | python3 triage_reply.py        # triage from stdin

Endpoint override: QUORUM_URL (default http://127.0.0.1:8017). Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

QUORUM_URL = os.environ.get("QUORUM_URL", "http://127.0.0.1:8017")

SAMPLES = [
    (
        "clearly interested",
        "Hi, thanks for reaching out — this is actually timely. We're reviewing "
        "our current setup this quarter and I'd like to see a demo. Do you have "
        "30 minutes next Tuesday or Wednesday? Budget is approved pending vendor "
        "selection.",
    ),
    (
        "not now",
        "Thanks for the note. We're not looking at new vendors right now — we "
        "just renewed our current contract for another year. Circle back in "
        "Q3 next year and we can talk then.",
    ),
    (
        "out of office",
        "Thank you for your email. I am out of the office until March 3rd with "
        "limited access to email. For urgent matters please contact "
        "ops-team@example.com. I will respond to your message upon my return.",
    ),
]

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is the intent of this sales reply?",
        "criteria": {
            "interested": "wants to engage: meeting, demo, pricing, or questions",
            "not_now": "not right now, but a future opening exists",
            "not_a_fit": "polite or explicit rejection, no future opening",
            "out_of_office": "automated or away message",
            "unsubscribe": "asks to be removed from the list",
        },
    },
    "heat": {
        "type": "score",
        "instructions": "How hot is this lead?",
        "criteria": [
            "hostile or explicitly disinterested",
            "cold: polite brush-off or automated reply",
            "warm: some engagement, but no concrete next step",
            "hot: wants a meeting/demo or asks about pricing",
        ],
    },
    "mentions_budget_or_timeline": {
        "type": "noul",
        "instructions": "Does the reply mention budget, pricing, a purchase "
                        "timeline, or a contract/renewal date?",
    },
}


def triage(email_text: str) -> dict:
    body = {"state": {"reply_email": email_text}, "questions": QUESTIONS}
    req = urllib.request.Request(
        QUORUM_URL.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def render(email_text: str, result: dict) -> str:
    answers = result["answers"]
    intent = answers["intent"]
    heat = answers["heat"]
    budget = answers["mentions_budget_or_timeline"]
    lines = [
        "=" * 68,
        email_text.strip()[:200] + ("..." if len(email_text.strip()) > 200 else ""),
        "-" * 68,
        f"INTENT : {intent['choice']:<14} "
        + "  ".join(f"{k}={v:.2f}" for k, v in (intent.get("probabilities") or {}).items())
        + f"   (confidence {intent.get('confidence', 0.0):.2f})",
        f"HEAT   : {heat['score']:.2f}/3  "
        + "  ".join(f"{k}={v:.2f}" for k, v in (heat.get("probabilities") or {}).items())
        + f"   (confidence {heat.get('confidence', 0.0):.2f})",
        f"BUDGET/TIMELINE: p_yes={budget.get('noul')}  "
        f"(confidence {budget.get('confidence', 0.0):.2f})",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("email", nargs="?", help="reply email text (else stdin)")
    ap.add_argument("--demo", action="store_true", help="run the 3 built-in sample replies")
    args = ap.parse_args()

    if args.demo:
        for label, text in SAMPLES:
            print(f"\n### SAMPLE: {label}")
            print(render(text, triage(text)))
        return 0

    email_text = args.email if args.email else sys.stdin.read()
    if not email_text.strip():
        ap.error("no email text provided")
    print(render(email_text, triage(email_text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
