# Teacher-calibrated judgment decomposition

How we got a 4B to do a job it fails when asked directly, written so it can be reused on
tasks that have nothing to do with Mario. The worked example is [examples/mario](../examples/mario/).

## The problem

A strong model can take a whole state `s` and answer one broad question: which action? A 4B
asked the same question collapses to a constant answer. It is not short of information. The
facts are in the state and it does not use them. What we measured on our 4B:

- it cannot compare decimals ("3.0 tiles away, is that 3 or closer?" is answered no)
- several questions in one call blur into one ("is an enemy within 3 tiles" becomes "is there an enemy")
- nested JSON is read top down and the tail is ignored
- negative phrasing inverts the answer
- without a short rationale its yes/no answers are noise

It does have two reliable skills: it detects presence, and it reads categories written in words.

## The factorisation

Replace the one broad judgment with a policy of this shape:

```
action(s) = g( [ p(q1 | φ(s)) ≥ τ1 ],  [ p(q2 | φ(s)) ≥ τ2 ],  ... )
```

- `φ(s)` is the **abstraction**, computed by code: the state rewritten as a short paragraph in
  words the model reads reliably. Every comparison the model cannot do is done here, by
  turning numbers into categories ("close ahead", "still some way off", "tall").
- `q1, q2` are **atomic yes/no questions**, one per call, phrased positively, about presence.
- `p(q | φ(s))` is the model's P(yes), read from token logprobs under a constrained decode.
  This is the part quorum provides.
- `τ` are **thresholds** fitted on labelled situations, per question and per model.
- `g` is **composition** in code: a fixed map from the yes/no answers to an action. It never
  looks at the state, so it cannot overrule the model.

The model keeps the semantic step: given this described situation, is action needed? Code keeps
arithmetic and control flow. In the Mario example there are no code overrides at all. Where a
rule must never be left to a probability (money, destructive commands), add it as an explicit
constraint outside `g` and report it as one, so nobody mistakes it for the model's judgment.

## Why it works

**1. A readable abstraction.** `φ` should be sufficient for the decision, written in the
model's readable vocabulary, and as small as possible. Quantising in code is what moves the
task from something the model cannot do (arithmetic) to something it can (reading).

**2. A finite abstraction can be verified exhaustively.** `φ` maps a huge state space onto a
small set of distinct situations. In Mario, 159 logged decisions collapse to 27 paragraphs. For
each question the right answer `T_q(s)` is computable from the state, so every (situation,
question) pair can be enumerated and checked, the way you would test a lookup table. The model's
share of the policy is then tested completely on the abstraction, which sampling can never give you.
This is also how wording gets chosen. A 4B's reading moves when a sentence is added or
dropped: removing one sentence changed 4 of 26 readings. So the paragraph template is a design
parameter, selected by the enumeration, and frozen once it passes.

**3. A teacher gives free supervision.** A stronger model's logged decisions `y*(s)` are labels
you already own. Each candidate design gets two scores, offline:

- *reading accuracy*: model score against `T_q(s)`. Isolates what the model reads from the
  teacher's timing noise.
- *teacher agreement*: AUC of the model score against the teacher's action.

When the yes and no scores separate, the threshold is the middle of the gap,
`τ = (max score on no + min score on yes) / 2`, the max-margin choice on a one-dimensional score.

**4. Failures refine the abstraction.** A live failure is a counterexample: either two states
that need different actions were given the same paragraph, or the run reached a paragraph the
enumeration never covered. Find the distinguishing fact, confirm on the logs that the teacher
behaves differently when it holds, add it to `φ`, re-run the enumeration, go again. It is the
same loop as counterexample-guided abstraction refinement in model checking. Our four:
an enemy below a ledge (new fact), a threshold cut too fine (refit), a sentence that appeared
live but never in the verified set (removed), and a hop into an enemy pair (a new atomic
question, because adding a fourth "or" to an existing question broke it).

## The procedure

```
1. log a teacher on the task                       -> (s, y*) pairs
2. write T_q(s) for each candidate question        -> computable truth
3. for each design (φ, q):  score all distinct φ(s) offline, cached
       keep designs with full reading accuracy; prefer the smallest φ
4. fit τ in the score gap; write g
5. run live; on failure: find the missing fact, check it against (s, y*),
   extend φ, repeat from 3
```

Steps 1 to 4 need no live environment. A design costs minutes, which is what makes searching
over representations practical.

## Using it elsewhere

| | Mario | lead triage | document field check | agent guardrail |
|---|---|---|---|---|
| `s` | NES RAM | CRM record, email thread | extracted field, source text | proposed action, context |
| `φ(s)` | hazards in words | "deal size large, last reply stale, asked for pricing" | the value and its source span in one sentence | the action in plain words plus risk facts |
| `q` | need to jump? big obstacle? | buying intent? fits our offer? | does the source support this value? | could this lose data? |
| `T_q` | from game state | from won and lost outcomes | exact match where possible, else labels | from past incidents |
| teacher | cloud Jev logs | a stronger model, or a human's past calls | same | reviewed corrections |
| `g` | run, hop, jump | priority bucket | accept or send to review | allow, ask, block |
| explicit constraints | none | never auto-send | money fields always reviewed | destructive commands always ask |

It fits when the decision breaks into a few binary judgments, each has a computable or labelled
truth, the abstraction stays small, and 1 to 2 seconds per judgment is acceptable.

On text tasks the model's share is larger than in Mario. There the semantic step, such as
whether an email shows buying intent, is the part code cannot compute.

## Latency, and a route to live use

A judgment costs about 1.7 s on our 4B with its one-sentence rationale, so the Mario harness
pauses the emulator while it answers. Without the rationale a judgment takes 373 ms, but the
scores stop separating (AUC 0.71), so that is not an option for this model. A model trained to
answer directly, which is what Jev is, would not pay this cost.

There is a second route, and we built it. The policy only depends on `φ(s)`, and `φ` takes few
values, so judgments can be memoised per situation: judged once by the model, then replayed at
lookup speed. Under greedy decoding the stored answer is exactly what the model would say again.
With the cache warmed over every paragraph seen in logged runs, the 4B cleared the level live,
emulator never paused, at a median lag of 1 frame. A miss is judged in the background while the
last action holds, and costs about 1.7 s of blind play, so coverage of the abstraction decides
whether a live run survives. This works for any task where `φ` is small: the expensive model
call happens once per situation, not once per event.

## Limits

- Readings are brittle to wording and to the model. Any change to the template, the questions
  or the weights means re-running the enumeration and refitting `τ`.
- The abstraction is designed by hand.
- Where every fact is computable, as in Mario, code carries most of the intelligence.
- Teacher mistakes are inherited, so check surprising labels before fitting to them.
- We have verified one level. The abstraction has no words yet for piranha plants, hammer
  brothers, water or moving platforms.
