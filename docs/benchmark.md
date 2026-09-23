# Benchmark: quorum vs Jev vs laya

Three ways to get a typed, calibrated decision out of text, run on the same 3,664 items:

- **Jev**: TypeSafe's hosted System One API (the cloud original quorum re-implements).
- **laya**: `convaiinnovations/laya`, open ModernBERT decision heads that run on CPU. `laya`
  is its router with task detection off (general checkpoints). `laya-td` turns detection on,
  which sends typed-decisions states to the `laya-typed-decisions` specialist.
- **quorum**: this repo, in front of our own judgment model,
  [judgment-qc-gate-qwen3-4b](https://huggingface.co/AltronisSG/judgment-qc-gate-qwen3-4b-GGUF)
  (a Qwen3-4B LoRA fine-tune, Q8_0, Apache-2.0 on Hugging Face), on llama.cpp. The file we
  ran is byte-identical to the Hugging Face copy (same sha256), so anyone can rerun the
  public tasks. It was trained to review agent work, answering in its own format (a verdict
  tag and a reason). `gate` is the held-out split of that training data, so its content is
  in-distribution for quorum, though quorum asks it through typed questions, not the format
  it was trained on. On the other 17 tasks every quorum score is zero-shot. `quorum-direct`
  asks every question in one call, the shipped default. `quorum-direct@<name>` is the same
  shim in front of a different model file (see "Which model behind quorum").

The harness is [bench/arena](../bench/arena/). Each system is run over the same frozen items
by `run.py` (resumable). `report.py` scores every system under the same rules.

## Results

**Short version.** Over 5,327 scored units on 18 tasks, quorum in front of our judge beats
laya by 0.085 accuracy (95% CI 0.068 to 0.102) and trails Jev by 0.221 (0.206 to 0.236,
gate excluded because Jev never sees it). The biggest lever is the model file: stock
Qwen3-4B-Instruct-2507 behind the same shim beats our judge by 0.106 (0.092 to 0.119),
beats laya by 0.190 and trails Jev by 0.117, though its raw probabilities need a fitted
map. With our judge, four settings each closed part of the gap to Jev, and each has a
cost: chain-of-thought (+0.088 on a 50-item slice, about 4x the latency), one call per
question on multi-question states (+0.065), a shortlist for 77-way choices (+0.047), and a
fitted yes/no threshold (sst2 0.793 to 0.920). Nothing we tried gets quorum to Jev.

### Accuracy, full run

`quorum` is `quorum-direct` in front of our judge, the shipped default. Differences are paired bootstraps; bold
means the 95% interval excludes zero.

| task | units | majority floor | Jev | laya | quorum | quorum - laya | quorum - Jev |
|---|---|---|---|---|---|---|---|
| agnews | 300 | 0.290 | 0.843 | 0.923 | 0.853 | **-0.070** [-0.110, -0.030] | +0.010 [-0.023, +0.047] |
| banking77 | 300 | 0.027 | 0.807 | 0.357 | 0.573 | **+0.217** [+0.157, +0.280] | **-0.233** [-0.287, -0.180] |
| cuad | 240 | 0.500 | 0.925 | 0.658 | 0.825 | **+0.167** [+0.100, +0.242] | **-0.100** [-0.150, -0.050] |
| emotion | 300 | 0.367 | 0.620 | 0.620 | 0.577 | -0.043 [-0.103, +0.017] | **-0.043** [-0.083, -0.003] |
| gate | 126 | 0.500 | n/a | 0.492 | 0.413 | -0.079 [-0.190, +0.024] | - |
| injection | 116 | 0.517 | 0.767 | 0.621 | 0.517 | -0.103 [-0.224, +0.017] | **-0.250** [-0.336, -0.172] |
| irony | 150 | 0.500 | 0.853 | 0.687 | 0.567 | **-0.120** [-0.220, -0.020] | **-0.287** [-0.373, -0.207] |
| legalbench | 380 | 0.500 | 0.866 | 0.634 | 0.629 | -0.005 [-0.074, +0.063] | **-0.237** [-0.292, -0.182] |
| mario | 165 | 0.782 | 0.994 | 0.345 | 0.758 | **+0.412** [+0.297, +0.527] | **-0.236** [-0.303, -0.170] |
| massive_en | 150 | 0.140 | 0.933 | 0.700 | 0.640 | -0.060 [-0.147, +0.020] | **-0.293** [-0.373, -0.220] |
| massive_ms | 150 | 0.140 | 0.900 | 0.213 | 0.520 | **+0.307** [+0.207, +0.400] | **-0.380** [-0.460, -0.293] |
| massive_ta | 150 | 0.140 | 0.920 | 0.273 | 0.340 | +0.067 [-0.033, +0.167] | **-0.580** [-0.667, -0.493] |
| massive_zh | 150 | 0.140 | 0.907 | 0.600 | 0.607 | +0.007 [-0.087, +0.100] | **-0.300** [-0.373, -0.227] |
| offensive | 150 | 0.500 | 0.813 | 0.627 | 0.607 | -0.020 [-0.087, +0.053] | **-0.207** [-0.293, -0.113] |
| spam | 150 | 0.500 | 0.973 | 0.773 | 0.820 | +0.047 [-0.047, +0.147] | **-0.153** [-0.220, -0.087] |
| sst2 | 150 | 0.500 | 0.960 | 0.620 | 0.793 | **+0.173** [+0.047, +0.300] | **-0.167** [-0.240, -0.093] |
| sst5 | 200 | 0.275 | 0.565 | 0.295 | 0.325 | +0.030 [-0.070, +0.135] | **-0.240** [-0.325, -0.145] |
| typed_decisions | 2000 | 0.461 | 0.740 | 0.361 | 0.497 | **+0.136** [+0.106, +0.164] | **-0.243** [-0.271, -0.216] |
| **pooled** | | | | | | **+0.085** [+0.068, +0.102] | **-0.221** [-0.236, -0.206] |

- **Where quorum beats laya:** Banking77 (77 intents), CUAD contract clauses, Malay
  MASSIVE, SST-2, typed-decisions and Mario. The Mario margin mostly measures laya: laya's
  0.345 is far under the 0.782 majority floor, and quorum's 0.758 is under it too.
- **Where laya beats quorum:** AG News and irony.
- **Against Jev,** quorum only draws on AG News. The widest gaps are Tamil and Malay
  MASSIVE (-0.580, -0.380) and the yes/no tweet and injection tasks.
- **At the floor:** quorum's injection score (0.517) is exactly the majority floor, and its
  Mario score is under it. Both come from a lean to one answer, covered below.

### Calibration

NLL and ECE (10 bins), raw -> the best map fitted under 5-fold cross-validation. Lower is
better. `n/a`: Jev never sees gate.

| task | Jev NLL | laya NLL | quorum NLL | Jev ECE | laya ECE | quorum ECE |
|---|---|---|---|---|---|---|
| agnews | 1.01 -> 0.97 | 0.22 -> 0.20 | 0.54 -> 0.43 | 0.11 -> 0.07 | 0.05 -> 0.01 | 0.11 -> 0.04 |
| banking77 | 1.46 -> 1.44 | 5.89 -> 5.09 | 4.94 -> 4.90 | 0.07 -> 0.04 | 0.52 -> 0.27 | 0.38 -> 0.29 |
| cuad | 0.22 -> 0.19 | 0.64 -> 0.64 | 0.45 -> 0.40 | 0.03 -> 0.03 | 0.07 -> 0.04 | 0.08 -> 0.03 |
| emotion | 2.84 -> 2.70 | 1.51 -> 1.11 | 2.79 -> 2.42 | 0.24 -> 0.17 | 0.23 -> 0.04 | 0.26 -> 0.04 |
| gate | n/a | 3.97 -> 3.81 | 2.45 -> 1.66 | n/a | 0.28 -> 0.20 | 0.33 -> 0.09 |
| injection | 0.59 -> 0.28 | 1.57 -> 0.68 | 1.98 -> 0.68 | 0.17 -> 0.07 | 0.28 -> 0.09 | 0.40 -> 0.10 |
| irony | 0.36 -> 0.26 | 0.59 -> 0.59 | 0.87 -> 0.67 | 0.07 -> 0.05 | 0.03 -> 0.03 | 0.23 -> 0.07 |
| legalbench | 0.33 -> 0.32 | 0.66 -> 0.62 | 0.76 -> 0.63 | 0.05 -> 0.04 | 0.12 -> 0.03 | 0.18 -> 0.04 |
| mario | 0.12 -> 0.03 | 0.80 -> 0.61 | 0.52 -> 0.47 | 0.10 -> 0.00 | 0.26 -> 0.11 | 0.10 -> 0.06 |
| massive_en | 0.50 -> 0.51 | 1.14 -> 1.08 | 1.70 -> 1.68 | 0.06 -> 0.03 | 0.13 -> 0.08 | 0.08 -> 0.06 |
| massive_ms | 0.78 -> 0.81 | 2.91 -> 2.66 | 2.54 -> 2.50 | 0.04 -> 0.05 | 0.21 -> 0.05 | 0.18 -> 0.09 |
| massive_ta | 0.59 -> 0.60 | 3.07 -> 2.48 | 3.15 -> 2.98 | 0.05 -> 0.05 | 0.32 -> 0.04 | 0.29 -> 0.14 |
| massive_zh | 0.57 -> 0.57 | 1.50 -> 1.36 | 1.69 -> 1.67 | 0.05 -> 0.05 | 0.15 -> 0.05 | 0.12 -> 0.09 |
| offensive | 0.43 -> 0.44 | 0.64 -> 0.42 | 1.18 -> 0.66 | 0.09 -> 0.07 | 0.14 -> 0.04 | 0.24 -> 0.07 |
| spam | 0.12 -> 0.10 | 0.44 -> 0.29 | 0.43 -> 0.41 | 0.06 -> 0.02 | 0.08 -> 0.07 | 0.04 -> 0.06 |
| sst2 | 0.17 -> 0.10 | 3.77 -> 0.49 | 0.38 -> 0.30 | 0.09 -> 0.02 | 0.37 -> 0.06 | 0.07 -> 0.04 |
| sst5 | 1.42 -> 1.35 | 2.59 -> 1.55 | 1.80 -> 1.47 | 0.16 -> 0.07 | 0.45 -> 0.04 | 0.26 -> 0.05 |
| typed_decisions | 1.54 -> 1.48 | 1.34 -> 1.20 | 2.36 -> 1.56 | 0.03 -> 0.02 | 0.17 -> 0.02 | 0.27 -> 0.06 |

Raw, quorum's probabilities are overconfident: its NLL is worse than Jev's on 15 of 17
shared tasks, and its median ECE across tasks (0.203) is the highest of the three (laya
0.190, Jev 0.068). A cross-validated map fixes most of that: quorum's ECE is under 0.10 on 5 of 18
tasks raw and on 15 of 18 after the map. The map is one or two numbers per question type, which
`quorum.calibrate` fits from your own labeled log.

### The lean on yes/no questions

On yes/no questions quorum's raw answers lean to one side, and the side depends on the
task. Share of "yes" answers, quorum against gold:

| task | quorum says yes | gold yes |
|---|---|---|
| injection | 7% | 52% |
| offensive | 15% | 50% |
| spam | 36% | 50% |
| irony | 85% | 50% |
| sst2 | 69% | 50% |
| legalbench | 69% | 50% |
| gate (yes/no question) | 92% | 32% |

A Platt map is a slope and an offset on the log-odds, so it can move the decision point.
This table re-decides every yes/no answer at 0.5 on the cross-validated Platt probability
(fitted on the other four folds, never on the item itself). Choice and score answers keep
their labels, so a task with both mixes the two.

| task | Jev | laya | quorum | quorum - laya after | quorum - Jev after |
|---|---|---|---|---|---|
| cuad | 0.925 -> 0.925 | 0.658 -> 0.629 | 0.825 -> 0.838 | **+0.208** [+0.138, +0.283] | **-0.087** [-0.133, -0.042] |
| gate | n/a | 0.492 -> 0.532 | 0.413 -> 0.643 | **+0.111** [+0.016, +0.198] | - |
| injection | 0.767 -> 0.888 | 0.621 -> 0.603 | 0.517 -> 0.578 | -0.026 [-0.147, +0.103] | **-0.310** [-0.405, -0.207] |
| irony | 0.853 -> 0.887 | 0.687 -> 0.693 | 0.567 -> 0.587 | **-0.107** [-0.193, -0.020] | **-0.300** [-0.393, -0.207] |
| legalbench | 0.866 -> 0.876 | 0.634 -> 0.634 | 0.629 -> 0.634 | +0.000 [-0.068, +0.063] | **-0.242** [-0.292, -0.192] |
| mario | 0.994 -> 0.994 | 0.345 -> 0.691 | 0.758 -> 0.782 | **+0.091** [+0.012, +0.170] | **-0.212** [-0.279, -0.152] |
| offensive | 0.813 -> 0.773 | 0.627 -> 0.827 | 0.607 -> 0.627 | **-0.200** [-0.293, -0.100] | **-0.147** [-0.240, -0.047] |
| spam | 0.973 -> 0.980 | 0.773 -> 0.920 | 0.820 -> 0.840 | **-0.080** [-0.147, -0.013] | **-0.140** [-0.207, -0.080] |
| sst2 | 0.960 -> 0.973 | 0.620 -> 0.787 | 0.793 -> 0.920 | **+0.133** [+0.060, +0.207] | **-0.053** [-0.100, -0.013] |
| typed_decisions | 0.740 -> 0.741 | 0.361 -> 0.379 | 0.497 -> 0.497 | **+0.118** [+0.090, +0.145] | **-0.243** [-0.272, -0.215] |
| **pooled** | | | | **+0.076** [+0.056, +0.096] | **-0.219** [-0.239, -0.199] |

The threshold helps quorum most on gate (its yes/no question alone goes from 0.397 to
0.857) and SST-2 (0.793 to 0.920). It helps laya more where laya is weak (offensive
0.627 to 0.827, spam 0.773 to 0.920, Mario 0.345 to 0.691), so quorum's lead over laya on
these tasks falls from +0.101 raw to +0.076. It never flips quorum past Jev. The threshold
needs labeled answers to fit (`quorum.calibrate` refuses fewer than 30), which is why it is opt-in
(`quorum.calibrate --noul-method platt`), never a default.

### Multi-question states: fan-out and prose (typed-decisions, full run)

typed-decisions asks 5 questions over one JSON state. `fanout` asks them in 5 calls
instead of 1; `prose` writes the JSON state out as sentences first.

| system | accuracy | vs quorum-direct | NLL raw -> CV | soft agreement | median ms/item |
|---|---|---|---|---|---|
| jev | 0.740 |  | 1.541 -> 1.484 | 0.539 | 316 |
| laya | 0.361 |  | 1.336 -> 1.197 | 0.331 | 1226 |
| laya-td | 0.766 |  | 0.884 -> 1.046 | 0.471 | 1440 |
| quorum-direct | 0.497 |  | 2.363 -> 1.564 | 0.424 | 1430 |
| quorum-prose | 0.501 | +0.004 [-0.009, +0.018] | 2.299 -> 1.550 | 0.430 | 1349 |
| quorum-fanout | 0.562 | **+0.065** [+0.043, +0.087] | 1.838 -> 1.481 | 0.456 | 2086 |
| quorum-fanout-prose | 0.583 | **+0.086** [+0.064, +0.108] | 1.708 -> 1.443 | 0.459 | 2148 |

- One call per question is +0.065, and cuts NLL from 2.36 to 1.84. Prose alone does
  nothing measurable (+0.004), but stacks with fan-out to +0.086.
- Fan-out costs calls, not much wall time: 5 upstream calls where direct makes 1, and the
  median item goes from 1.4 s to 2.1 s because each call is short.
- On gate (2 questions), fan-out does not help: 0.389 against 0.413, interval across zero.
- The best quorum setup here (0.583) is still under Jev (0.740) and laya-td (0.766). laya-td
  is a specialist trained for these four workflows; quorum and Jev answer zero-shot.

### Chain-of-thought (50 items per task)

`quorum-cot` writes a short rationale before each answer, then answers under the same
constrained decode. All 900 items ran in CoT mode (no fallback to direct). This is a
50-item slice of every task (gate 100 units, typed-decisions 250), so intervals are wide.

| task | quorum-direct | quorum-cot | cot - direct | direct ms | cot ms |
|---|---|---|---|---|---|
| agnews | 0.880 | 0.900 | +0.020 [-0.060, +0.100] | 390 | 1842 |
| banking77 | 0.480 | 0.500 | +0.020 [-0.060, +0.120] | 896 | 2410 |
| cuad | 0.700 | 0.800 | +0.100 [-0.080, +0.280] | 668 | 2152 |
| emotion | 0.660 | 0.580 | -0.080 [-0.200, +0.040] | 402 | 1974 |
| gate | 0.420 | 0.540 | **+0.120** [+0.040, +0.200] | 744 | 2931 |
| injection | 0.640 | 0.660 | +0.020 [-0.060, +0.100] | 396 | 1994 |
| irony | 0.620 | 0.640 | +0.020 [-0.140, +0.180] | 372 | 2046 |
| legalbench | 0.620 | 0.480 | -0.140 [-0.340, +0.060] | 352 | 1912 |
| mario | 0.640 | 0.900 | **+0.260** [+0.100, +0.440] | 449 | 1812 |
| massive_en | 0.660 | 0.780 | +0.120 [-0.020, +0.240] | 507 | 1596 |
| massive_ms | 0.600 | 0.760 | **+0.160** [+0.040, +0.280] | 480 | 1628 |
| massive_ta | 0.340 | 0.580 | **+0.240** [+0.100, +0.380] | 539 | 1609 |
| massive_zh | 0.600 | 0.760 | **+0.160** [+0.020, +0.300] | 565 | 1590 |
| offensive | 0.600 | 0.620 | +0.020 [-0.120, +0.160] | 465 | 1622 |
| spam | 0.800 | 0.640 | -0.160 [-0.320, +0.000] | 452 | 1564 |
| sst2 | 0.820 | 0.800 | -0.020 [-0.160, +0.120] | 484 | 1601 |
| sst5 | 0.300 | 0.460 | +0.160 [+0.000, +0.340] | 486 | 1646 |
| typed_decisions | 0.412 | 0.588 | **+0.176** [+0.112, +0.244] | 1350 | 6003 |
| **pooled** | | | **+0.088** [+0.058, +0.117] | | |

- CoT is the largest single gain we measured, +0.088 pooled, and it lands where direct is
  weakest: non-English MASSIVE (+0.160 to +0.240), Mario (+0.260), gate (+0.120) and
  typed-decisions (+0.176). On the slice it beats laya by +0.130 and trails Jev by -0.152.
- It is not free of losses: spam (-0.160) and legalbench (-0.140) drop, though neither
  interval excludes zero on 50 items.
- It costs about 4x the latency: 1.6 to 2.4 s per single-question item against 0.35 to
  0.9 s, and 6.0 s against 1.35 s on typed-decisions.
- CoT plus fan-out is worse than CoT alone on typed-decisions (0.528 against 0.588).

### Shortlist: 77-way choice (Banking77, full run)

| system | accuracy | median ms/item |
|---|---|---|
| jev | 0.807 | 316 |
| laya | 0.357 | 408 |
| laya+sl | 0.567 | 216 |
| quorum-direct | 0.573 | 885 |
| quorum-direct+sl | 0.620 | 540 |

- quorum-direct+sl - quorum-direct: **+0.047** [+0.007, +0.087]
- laya+sl - laya: **+0.210** [+0.153, +0.270]
- quorum-direct+sl - laya+sl: +0.053 [-0.003, +0.107]
- quorum-direct+sl - jev: **-0.187** [-0.237, -0.140]

A 20-label shortlist from a shared bi-encoder helps both systems, and helps laya far more,
because laya struggles to pick from 77. With the same shortlist, quorum stays ahead
(+0.053) but the interval now includes zero, and laya+sl is 2.5x faster. The shortlist caps
both at 0.923 (its recall at 20). Jev, with no shortlist, scores 0.807.

### Which model behind quorum

quorum works with any llama-server model, so we ran the same shim, with the same flags, in
front of three other 4B files, one at a time:

- `@qwen3-4b-2507`: stock Qwen3-4B-Instruct-2507, the Q8_0 GGUF from
  `unsloth/Qwen3-4B-Instruct-2507-GGUF`, a later instruct release of the same size.
- `@qwen3-4b-base`: stock Qwen3-4B, the Q8_0 GGUF from `Qwen/Qwen3-4B-GGUF`. It is the model
  our judge was fine-tuned from.
- `@judge-v9`: the previous version of our judge.

All three ran the 50-item slice first:

| task | our judge | Qwen3-4B-Instruct-2507 | Qwen3-4B | judge v9 |
|---|---|---|---|---|
| agnews | 0.880 | 0.880 | 0.900 | 0.900 |
| banking77 | 0.480 | 0.560 | 0.540 | 0.540 |
| cuad | 0.700 | 0.880 | 0.880 | 0.800 |
| emotion | 0.660 | 0.660 | 0.640 | 0.620 |
| gate | 0.420 | 0.590 | 0.540 | 0.530 |
| injection | 0.640 | 0.660 | 0.760 | 0.640 |
| irony | 0.620 | 0.800 | 0.660 | 0.600 |
| legalbench | 0.620 | 0.620 | 0.640 | 0.640 |
| mario | 0.640 | 0.920 | 0.760 | 0.520 |
| massive_en | 0.660 | 0.820 | 0.740 | 0.700 |
| massive_ms | 0.600 | 0.760 | 0.680 | 0.600 |
| massive_ta | 0.340 | 0.740 | 0.620 | 0.420 |
| massive_zh | 0.600 | 0.800 | 0.720 | 0.660 |
| offensive | 0.600 | 0.800 | 0.820 | 0.440 |
| spam | 0.800 | 0.860 | 0.700 | 0.500 |
| sst2 | 0.820 | 0.920 | 0.960 | 0.700 |
| sst5 | 0.300 | 0.440 | 0.280 | 0.460 |
| typed_decisions | 0.412 | 0.548 | 0.504 | 0.448 |

Against our judge on the slice, pooled: Qwen3-4B-Instruct-2507 +0.138 (+0.105 to +0.171), Qwen3-4B
+0.089 (+0.057 to +0.120), judge v9 +0.008 (-0.014 to +0.030). Qwen3-4B-Instruct-2507 then ran all 3,664 items:

| task | our judge | Qwen3-4B-Instruct-2507 | 2507 - our judge | our judge ms | 2507 ms |
|---|---|---|---|---|---|
| agnews | 0.853 | 0.837 | -0.017 [-0.050, +0.017] | 412 | 288 |
| banking77 | 0.573 | 0.660 | **+0.087** [+0.043, +0.130] | 885 | 808 |
| cuad | 0.825 | 0.887 | **+0.062** [+0.008, +0.117] | 674 | 592 |
| emotion | 0.577 | 0.590 | +0.013 [-0.027, +0.053] | 401 | 299 |
| gate | 0.413 | 0.587 | **+0.175** [+0.040, +0.317] | 744 | 2190 |
| injection | 0.517 | 0.534 | +0.017 [-0.043, +0.078] | 395 | 298 |
| irony | 0.567 | 0.733 | **+0.167** [+0.100, +0.240] | 371 | 284 |
| legalbench | 0.629 | 0.766 | **+0.137** [+0.082, +0.195] | 468 | 307 |
| mario | 0.758 | 0.952 | **+0.194** [+0.133, +0.255] | 432 | 308 |
| massive_en | 0.640 | 0.793 | **+0.153** [+0.093, +0.213] | 494 | 382 |
| massive_ms | 0.520 | 0.673 | **+0.153** [+0.087, +0.227] | 514 | 384 |
| massive_ta | 0.340 | 0.627 | **+0.287** [+0.213, +0.367] | 558 | 388 |
| massive_zh | 0.607 | 0.793 | **+0.187** [+0.127, +0.253] | 572 | 374 |
| offensive | 0.607 | 0.753 | **+0.147** [+0.067, +0.227] | 482 | 286 |
| spam | 0.820 | 0.853 | +0.033 [-0.053, +0.120] | 462 | 261 |
| sst2 | 0.793 | 0.907 | **+0.113** [+0.033, +0.193] | 483 | 256 |
| sst5 | 0.325 | 0.460 | **+0.135** [+0.060, +0.205] | 502 | 301 |
| typed_decisions | 0.497 | 0.598 | **+0.101** [+0.076, +0.125] | 1430 | 1254 |
| **pooled** | | | **+0.106** [+0.092, +0.119] | | |

- **It is the most accurate quorum we measured.** It is ahead of our judge by 0.106 (0.092 to 0.119)
  pooled, on 14 tasks, and behind on none. It beats laya by 0.190 (0.174 to 0.207), where our judge
  beats laya by 0.085, and trails Jev by 0.117 (0.104 to 0.131), where our judge trails by 0.221. It is
  also faster on single-question items: 304 ms per item (median of task medians)
  against 482 ms.
- **Its raw probabilities are close to hard labels.** On the seven yes/no tasks where
  every answer carries a distribution, 95.6% of its answers are above 0.99, against 18.8%
  for our judge, and its raw NLL is worse on 6 of the 7. After the cross-validated map its
  NLL is better on all 7, and its median ECE matches our judge's (0.058 against 0.055).
  With this model, fit the map (`quorum.calibrate`) before trusting a probability.
- **It ranks yes/no items better.** AUROC of P(yes) against gold, median over the nine
  yes/no tasks: 0.930, against 0.862 for our judge (Jev 0.970 over eight, laya 0.709).
- **Some of its answers carry no probability.** On 881 of 5,327 units (16.5%; 790 of them
  are choice questions) the model was so sure that no competing option reached its 12
  most likely tokens. quorum returns no distribution rather than invent one; our judge has
  11 such units. Under the rules those units score as uniform, so this model's NLL and ECE
  on choice tasks are pessimistic and we do not compare them. The label is always there,
  so accuracy is not affected.
- **On gate the result splits by question.** gate asks for a 16-way verdict and a yes/no
  "was the work done properly". Our judge picks the right verdict more often (27 of 63,
  against 20), so the fine-tune did learn its categories. On the yes/no question our judge
  says "yes" to 38 of the 43 items where a standard was broken, and Qwen3-4B-Instruct-2507
  says "yes" to none. The gain on gate comes from that question, and it disappears once
  both are re-decided on a fitted threshold (-0.016, interval -0.119 to +0.087). This model is also slower on
  gate (2.2 s against 0.7 s per item): its answers there run to a median 77 tokens against
  17.
- **Typed questions cost our judge part of what it learned.** Asked in its own trained
  format (its training system prompt, a free-text verdict tag), the same file gets 35 of
  63 verdicts right and passes none of the 43 broken-standard items as OK. That is the
  best gate result in this report, and it is how our own gate runs it.
- **A second slot did not buy speed.** `quorum-fanout@judge-p2` ran our judge with two
  parallel slots, and quorum sending two fan-out calls at once. On the slice the median
  item took 1,112 ms on gate against 899 ms with one slot, and 1,891 ms against 1,819 ms
  on typed-decisions, at the same accuracy (gate 0.370 against 0.380, typed-decisions
  0.508 each). On this integrated GPU, one slot is the setting.

### Latency

Median wall time per item, full run, one caller. Jev is a network call from Singapore
with 4 requests in flight; laya is fp32 on 16 CPU threads; quorum is a 4B at Q8_0 on an
integrated GPU, one slot.

| system | single-question tasks, median of task medians | typed_decisions (5 questions) |
|---|---|---|
| jev | 311 ms (296-320) | 316 ms |
| laya | 152 ms (105-578) | 1226 ms |
| quorum-direct | 482 ms (371-885) | 1430 ms |
| quorum-direct@qwen3-4b-2507 | 304 ms (256-808) | 1254 ms |

The same shim in front of Qwen3-4B-Instruct-2507 is faster on both columns. On gate it is
slower (2.2 s against 0.7 s per item), covered in "Which model behind quorum".

## Tasks

18 tasks, 3,664 items, 5,327 scored units (one unit = one question on one item). Items are
sampled with a fixed seed (`tasks.py`, seed 20260923) from public test splits, except
where noted.

| task | n items | question | options | source |
|---|---|---|---|---|
| agnews | 300 | choice | 4 | fancyzhx/ag_news |
| banking77 | 300 | choice | 77 | Banking77 intents |
| emotion | 300 | choice | 6 | dair-ai/emotion |
| massive_en / _ms / _zh / _ta | 150 each | choice | 18 | MASSIVE scenario (English, Malay, Chinese, Tamil) |
| sst2 | 150 | noul | 2 | stanfordnlp/sst2 (validation: test has no labels) |
| sst5 | 200 | score | 5 | SetFit/sst5 |
| spam | 150 | noul | 2 | ucirvine/sms_spam (only split) |
| offensive | 150 | noul | 2 | tweet_eval offensive |
| irony | 150 | noul | 2 | tweet_eval irony |
| injection | 116 | noul | 2 | deepset/prompt-injections |
| legalbench | 380 | noul | 2 | 7 LegalBench yes/no tasks, class-balanced |
| cuad | 240 | noul | 2 | CUAD clause detection over a 2,500-char contract window |
| typed_decisions | 400 | 5 per item (choice, noul, score) | 2-5 | LocalLLaMA/typed-decisions test, soft gold |
| mario | 165 | noul | 2 | the words-only Mario situations from [judgment-decomposition](judgment-decomposition.md) |
| gate | 63 | choice + noul | 16 / 2 | private heldout from our own agent-review gate |

`gate` is real session content. It never leaves the box, so Jev is `n/a` there, and it is
reported only as aggregates. typed-decisions (four business workflows) is pinned to a
dataset revision; its gold carries a teacher's full distribution, so its NLL is a
cross-entropy against soft targets.

## Rules

A benchmark is only as honest as its denominators, so these are fixed in `report.py` and
tested in `test_arena_report.py`:

- **Every item counts.** An item with no successful answer (error, timeout, never reached)
  is wrong for accuracy and scores the uniform distribution on NLL, Brier and ECE. Coverage
  is printed whenever it is below 100%.
- **Calibration is cross-validated.** Raw columns are what a system returns today. The CV
  columns fit a map under 5-fold cross-validation, folded by item so the questions of one
  item never straddle a fold: a temperature per question type, and Platt scaling for noul.
  Calibration moves probabilities, never labels, so the accuracy columns are unchanged by
  it. One table is the exception, and says so: it re-decides yes/no answers at 0.5 on the
  cross-validated Platt probability and reports that accuracy next to the raw one.
- **Differences are paired.** Each system pair is compared on the same units with a paired
  bootstrap (2,000 resamples), reported as the accuracy difference, its 95% interval, and
  the share of resamples where the first system is ahead.
- **Latency** is the median wall time per item as the harness saw it. quorum and laya ran
  one request at a time; Jev ran 4 requests in flight, which a hosted API is built for. It
  is not a hardware comparison: Jev is a network API from Singapore, laya runs fp32 on 16
  CPU threads, and quorum runs on an integrated GPU. The column shows what one caller
  waits for, on this setup.

## Findings

1. **quorum sits between laya and Jev.** A 4B tuned to review agent work, used zero-shot
   as a classifier, is ahead of laya's general checkpoints on the pooled units (6 task
   wins, 2 losses, 10 level) and well behind Jev everywhere except AG News. laya's
   specialist beats it on the distribution the specialist was trained for.
2. **Its raw probabilities are overconfident, and cheap to fix.** A cross-validated
   temperature or Platt map (one or two numbers per question type) cuts quorum's ECE
   under 0.10 on 15 of 18 tasks. `quorum.calibrate` fits the same maps from a labeled log.
3. **The worst per-task defect is a lean on yes/no questions.** quorum says "no" to 93%
   of injection items and "yes" to 92% of gate items, where gold is near half or under.
   A threshold fitted on labels undoes part of it (gate's yes/no question 0.397 to
   0.857), which argues for labeling a few dozen answers per yes/no question before
   trusting it.
4. **Structure helps, rewording does not.** One call per question is +0.065 on
   five-question states; rewriting the JSON state as prose is +0.004 on its own.
5. **Reasoning helps where the 4B is weakest.** CoT is +0.088 pooled on the slice and
   up to +0.240 on non-English MASSIVE, at about 4x the latency. It fits low-volume,
   high-stakes calls; direct fits hot paths.
6. **The model file is the biggest lever.** Stock Qwen3-4B-Instruct-2507 behind the same
   shim is +0.106 on the full run, more than any setting, and faster on single questions.
   Its probabilities need a fitted map before they mean anything. Used as a general
   classifier through typed questions, our judge's fine-tune costs accuracy against its
   own base (Qwen3-4B is +0.089 on the slice). It remains the better model for the job it
   was trained for, reviewing agent work in its own format.

**What we would change for a user today** (the shipped default stays `direct`, one call):

| situation | setting | measured effect |
|---|---|---|
| general classification, not reviewing agent work | Qwen3-4B-Instruct-2507 as the model, then `quorum.calibrate` | +0.106 pooled on the full run, faster on single questions |
| several questions over one state | `QUORUM_FANOUT=1` | +0.065 on typed-decisions |
| accuracy matters more than latency | `QUORUM_COT=1` or `"reasoning": true` per request | +0.088 pooled on the slice, about 4x latency |
| a yes/no question you can label 30+ answers for | `quorum.calibrate --noul-method platt` | SST-2 0.793 to 0.920, gate yes/no 0.397 to 0.857 |
| more than about 20 options | shortlist first; quorum does not do this itself, [shortlist.py](../bench/arena/shortlist.py) is the recipe | +0.047 on Banking77 |

## Fairness notes

- **laya's published numbers are not ours.** Its card reports Banking77 0.425 against Jev's
  0.870, and latency on a T4 GPU. We measure laya on CPU (this box has no CUDA), and our
  Banking77 sample is 300 items over all 77 intents, where the card's figure is its own
  sample.
- **laya-td is in-distribution on typed-decisions.** Its card lists the specialist as
  trained for the four typed-decisions workflows, so its score there is a specialist on its
  own distribution. Jev and quorum answer zero-shot.
- **laya's load.** The box keeps a large model resident, so the harness can load laya by
  memory-mapping its checkpoint into a meta-device model (`ARENA_LAYA_LOAD=mmap`) instead
  of its stock fp32 load. Every laya number in this report comes from the unmodified stock
  load, run while the local judge was stopped to make room, except `gate`, which ran mapped. On gate we checked the mapped
  load against a stock run: P(ok) matches to 4 decimal places on all 63 items, and both
  questions score the same (choice 26/63, noul 36/63). Latency only counts stock rows.
- **The shortlist ceiling.** `+sl` systems see the top 20 Banking77 intents ranked by a
  shared bi-encoder (`shortlist.py`, laya's own coarse-to-fine step). The gold intent is in
  that top 20 for 92.3% of items, so no `+sl` system can score above 0.923. k was fixed
  at laya's default before any run and never tuned.
- **Mario was written for this 4B.** Its questions come from
  [judgment-decomposition](judgment-decomposition.md), where they were worded until this
  4B could answer them, and the gold is the game's own state, not any model's answer. Jev
  and laya answer questions tuned for another model. The level-clearing demo in the README
  also decides on thresholds fitted against Jev; the arena decides at 0.5, which is why
  quorum's raw Mario score here sits under the majority floor.
- **The model comparison ran on stock files.** Both Qwen files were downloaded from
  Hugging Face and not changed (sha256 checked against the published copies), and served
  with the same llama-server flags as our judge, one model at a time. `@judge-v9`, the
  previous version of our judge, is not published, so that column cannot be rerun.

## An engineering finding: llama-server's prompt cache

The first full quorum run was killed by the host's memory guard. The cause was not quorum:
llama-server keeps a host-RAM prompt cache (`--cache-ram`, default 8192 MiB) and saves a
slot into it whenever the slot moves to a new prompt. The saving happens regardless of the
request's `cache_prompt` field (in the llama.cpp build we run), so a client cannot opt
out. Under varied traffic like a benchmark, or a busy gate, the judge's resident memory grew
from about 0.6 GiB to 8.5 GiB, on top of the ~4.5 GiB the model holds in GPU memory.

`--cache-ram 0` holds it flat, at about 0.6 GiB resident plus the model's GPU memory, and
changes no outputs: every arena quorum run uses it. If you serve a small judge on a
memory-tight box, set it.

## Next: a fine-tuned quorum

laya-td shows what a small specialist trained on one distribution can do there. quorum can
get the same advantage, since its weights are ours to train. The comparison of models
changes where that starts: a fine-tune for classification should start from stock
Qwen3-4B-Instruct-2507, not from our judge, and it only pays if it beats that model as it
stands. The plan, not yet run:

- **Data:** public train splits only, never the test items above: typed-decisions train
  (1,200 items with the teacher's full distribution), and the train splits of AG News,
  Banking77, emotion, MASSIVE, SST and tweet_eval, rendered through quorum's own prompt
  and answer schema so the model learns the exact call it is served with.
- **Base:** Qwen3-4B-Instruct-2507, LoRA, the recipe our judge used.
- **Cost:** our judge fine-tune (Qwen3-4B, LoRA r=16) trained 42 steps, about 336
  sequences, in 2.7 minutes on this box. At that rate a 5,000-sequence pass is about 40
  minutes (an estimate: sequence lengths differ), plus a merge and Q8_0 conversion.
- **Constraint:** training cannot share the box with the resident large model, so it
  needs a planned maintenance window. That is scheduling, not feasibility.
- **Honesty rule:** the result gets its own name in the report, like laya-td, and is read
  as an in-distribution specialist, never as a zero-shot score.
