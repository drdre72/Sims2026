# HZN-OC-01 Preflight — Measured Channel Costs

**Component status (HZN-IR-05 d2; HZN-IR-08 corrections applied):** symbolic encoder ✅, tokenizer profiling ✅, observation audit layer ✅ (mandatory provenance, edit-chain binding), comprehension harness C1–C4 ✅ (fail-closed gate, strict grading, campaign-realistic catalogs). 47/47 tests. This document reports measured token costs over the exact rendered surfaces (families + bounds legends + rows) with canonical field-parity verified on every level.

## 1. Measured costs, real tokenizer, real catalogs

192 rows arm A, 248 rows arm B. Parity = canonical record equality including effective magnitude intervals and count defaults:

| Level | Arm A tokens | /row | ratio | parity | Arm B tokens | /row | ratio | parity |
|---|---|---|---|---|---|---|---|---|
| linguistic (baseline) | 8,801 | 45.8 | 1.00 | ✅ | 8,877 | 35.8 | 1.00 | ✅ |
| s1 (spaces, familiar keys) | 7,811 | 40.7 | 0.888 | ✅ | 7,637 | 30.8 | 0.860 | ✅ |
| s2 (`O:/F:/AR:/LC:` symbolic) | 8,488 | 44.2 | 0.964 | ✅ | 8,457 | 34.1 | 0.953 | ✅ |
| **s3 (headered positional)** | **7,386** | **38.5** | **0.839** | ✅ | **7,156** | **28.9** | **0.806** | ✅ |

Sample rows (arm B): linguistic `0|L0|f0|n=81920|ar=1.00|lc=+0.02` → s3 `0|0|f0|81920|1.00|+0.02` under header `cols: id|layer|fam|numel|ar|lc`.

**Two measured findings:**
1. **S-II loses to S-I** — symbolic vocabulary is shorter in characters but BPE-fragments; character compression ≠ token compression (reviewer prediction confirmed).
2. **S-III (headered) beats S-I** — removing *repeated field names* (structural redundancy), not inventing vocabulary, is the best measured scheme: **−16.1% arm A, −19.4% arm B** at full field precision and complete semantics. This supports the reviewer's "eliminate unnecessary representation" direction.

## 2. Token accounting — consistent denominators (HZN-IR-08 d9)

All figures below are **per-row tokens** on the arm-B row `1|L0|f1|n=81920|ar=0.04|lc=+0.06` (35.8 tok/row mean, linguistic):

| Field group | Rendering | Tokens | % of row |
|---|---|---|---|
| id + layer + fam | `1\|L0\|f1` | ~6 | ~17% |
| numel | `\|n=81920` | ~6 | ~17% |
| obs columns (ar+lc) | `\|ar=0.04\|lc=+0.06` | ~15 | ~42% |
| separators/syntax | | remainder | ~24% |

So, **of a row**: observation columns ≈ 34–42%, structural identity fields ≈ 34–45%, syntax ≈ 20–25%. **Of the whole prompt**: catalog rows ≈ 8.8k of ~9.5k serialized tokens ≈ **93%** (legends + instructions + arm context + history ≈ 600 tok). The earlier inconsistent figure ("structural fields 85–90% of catalog") conflated row-share with prompt-share; corrected accounting above: within the catalog, obs columns ≈ 40%, structure ≈ 60%; the catalog itself is ≈ 93% of the prompt.

## 3. Consequences and reviewer rulings applied

- Precision reductions (numel→`23M`, mag→2sf, numel removal) are **information-ablation experiments, not channel candidates** — they change what the proposer knows, so they cannot run inside the information-matched epoch (HZN-IR-08 d8, resolving my earlier §3 questions 1–3).
- **Eligibility is against the L baseline**, not the previous best alternative: s3 qualifies on tokens (−16/−19%), parity (canonical equality), and must still pass C1–C4 before trajectories (HZN-IR-08 d7 fix — the earlier "must beat S1" clause would have excluded S1 itself).
- Separator/field-name optimization was explicitly sanctioned with complete semantics retained — that is exactly what s3 is.
- Realistic ceiling at full precision: **≈ −20% catalog ≈ −18% of the serialized prompt.** The 10–30× figure remains a documented negative result.
- The addressable-store arm and KV-cache/static-prefix reuse (with post-edit invalidation) become the more promising structural directions — filed as the deferred candidates; the catalog's static-vs-dynamic split (identity/bounds vs ar/lc/freshness) is the formal hook.

## 4. Operational comprehension gate — measured result (HZN-IR-08 d6)

Ran the C1–C4 pretest against the real model, campaign-identical prompting (chat template, thinking disabled, greedy): 3 encodings × 3 independently generated campaign-realistic catalogs (248 rows, near-zero contrast mass, rounded ties, reserved-unique + repeated family/layer pairs). `results/gate1_3_comprehension.json`.

| Encoding | C1 decode (≥95%) | C2 compare (≥90%) | C3 compose (≥90%) | C4 ground (100%, ≥60) | Verdict |
|---|---|---|---|---|---|
| linguistic | 48/48 (100%) | 43/48 (90%) | 15/24 (62%) | 35/84 (42%) | **FAIL** |
| s1 | 48/48 (100%) | 43/48 (90%) | 12/24 (50%) | 38/84 (45%) | **FAIL** |
| s3 | 48/48 (100%) | 29/48 (60%) | 0/24 (0%) | 33/84 (39%) | **FAIL** |

**Reading:**

1. **Decoding is never the problem** — 100% on every encoding, including positional s3.
2. **Grounding is the bottleneck, encoding-invariant** — C4 ≈ 40–45% on ALL levels including the linguistic baseline. The failures are clean (valid integer or "ambiguous" — just wrong), with ~half of true-ambiguous probes answered with a confident id instead of refusal. Under the reviewer's own logic, an observation the model decodes but addresses to the wrong parameter is an intervention on the wrong component — and this channel shows that failure ~half the time at 248-row scale, *independent of representation*.
3. **s3 pays a real comprehension tax** — C2 drops to 60% and C3 to 0% (most responses empty `[]`). Positional rows shown without their `cols:` header make field-identification ambiguous in isolated-row probes; whether that under-tests s3 relative to the full catalog context is a fairness question flagged for the reviewer — but the C4 result holds even where decoding is perfect.
4. **The pretest is stricter than the campaign's actual grounding operation** — C4 probes test *find-the-row-matching-a-description*; the campaign's flow is *select-a-row-and-emit-its-param_id* (copy, not lookup). The strict form is what the spec registered; both framings are reported.

**Implication:** this gate did its job — it separated decodability (solved) from operational grounding (not solved, at catalog scale, on any encoding). It suggests the binding constraint on observation-driven editing is not channel cost at all but **catalog-scale addressing** — which strengthens, again, the addressable-store direction: the model cannot reliably ground over 248 rows anyway, so retrieving fewer relevant rows loses little and may help. Alternatively a small-catalog arm (e.g., ~32 rows) would directly test whether C4 accuracy is scale-driven.

## 5. What remains before campaign authorization

- ~~Paired within-seed sensitivity analysis~~ — done (`gate1_3_sensitivity.json`): ε=δ=0.05 provisional; within-seed σ≈0.127 (CI [0.066, 0.795]); 80% correspondence power needs n=7 at Gate-1.2-magnitude effects; **non-inferiority at ε=0.05 is below the noise floor — margin decision returned to the reviewer.**
- ~~Budget accounting split~~ — done: `stage_seconds` (in-loop pool) vs `final_eval_seconds` (fixed allowance) vs `unattributed_wall_seconds`.
- ~~Provenance threading~~ — done: `collect_observations` emits the full `{base, edit-chain, context, config, epoch}` tuple; surfaced in the campaign report.
- ~~Comprehension operational run~~ — done (§4): **all encodings fail C3/C4 at campaign scale; grounding, not channel cost, is the measured bottleneck.**
- ε/δ finalization — reviewer's call informed by the sensitivity table.
- Final resource allocation + primary utility definition sign-off.
- Reviewer disposition on the C3/C4 finding before trajectories run.

## 5. Artifacts

- `results/gate1_3_channel_profile.json` — per-level profile + parity on the real catalogs
- `results/gate1_3_comprehension.json` — 9 cells × responses + scores (campaign prompting)
- `results/gate1_3_comprehension_rawprompt_invalid.json` — preserved method error: raw (non-chat-template) prompts produce 0% — the model continues the catalog rather than answering; demonstrates why the pretest must match the campaign's prompt surface
- `results/_profile_channels.py`, `results/_comprehension_run.py`, `results/_sensitivity_analysis.py`, `results/_grounding_isolation.py` — deterministic reproductions (committed via `-f`; `results/` is gitignored)
- `results/gate1_3_grounding.json` — 60 cells × probes + verdicts + tokens/latency/VRAM (HZN-G1.3-R)
- `src/jarvis_research/gate1_3_channel.py` (canonical parity, audit, s3), `gate1_3_comprehension.py` (fail-closed, strict grading, realistic catalogs, IR-09 F1–F7), `gate1_3_store.py` (deterministic read-only store), `tests/test_gate1_3_channel.py` (57 tests incl. the HZN-IR-08/09 regression sets)

## 7. HZN-G1.3-R grounding isolation — measured result (HZN-IR-09 §4)

Corrected harness (F1–F7 applied), linguistic encoding, nested catalogs 16/32/64/128/248 (seeded subsets), 2 seeds, inference only. `results/gate1_3_grounding.json`, runner `results/_grounding_isolation.py`, store `src/jarvis_research/gate1_3_store.py`.

**Unique-target descriptive lookup — the scale curve:**

| rows | R0R1 full catalog | R4 store-retrieved | R6 oracle |
|---|---|---|---|
| 16 | 18/20 | 20/20 | 20/20 |
| 32 | 18/20 | 20/20 | 20/20 |
| 64 | 17/20 | 19/20 | 20/20 |
| 128 | **9/20** | 18/20 | 20/20 |
| 248 | **9/20** | 18/20 | 16/20 |

**The grounding cliff is between 64 and 128 rows.** Deterministic family-filtered retrieval (~27–45 rows at 248) holds 90% everywhere and matches oracle within noise.

**All conditions, both seeds (unique+ambiguous+missing probes):**

| rows | R0R1 | R2 copy-id | R3 describe→resolve | R4 | R5 random | R6 |
|---|---|---|---|---|---|---|
| 16 | 23/29 | 16/16 | 5/16 | 27/29 | 28/29 | 28/29 |
| 32 | 26/32 | 16/16 | 5/16 | 27/32 | 31/32 | 28/32 |
| 64 | 23/36 | 16/16 | 3/16 | 27/36 | 34/36 | 28/36 |
| 128 | 12/36 | 16/16 | 3/16 | 26/36 | 30/36 | 27/36 |
| 248 | **9/36** | 16/16 | **0/16** | 25/36 | 24/36 | 22/36 |

**Failure taxonomy at 248 (R0R1):** 9 correct, 9 wrong-but-valid ids, 8 hallucinated ids on missing targets, 8 invalid picks on ambiguous targets, 2 refusals when present. The model answers a confident id ~always — ambiguity recognition is 0/8 at *every* size; missing recognition collapses 5/8→8/8→0/8 across the sweep.

**Attribution:**
- **R2 copy-id is 16/16 at every size** — the campaign's actual select-and-copy operation is NOT the bottleneck. Descriptive lookup and copying an existing identifier are different operations, as the reviewer predicted.
- **R3 describe→resolve fails at 248 (0/16) and is already imperfect at 16** — separating lookup from reasoning (store resolves the model's descriptor) doesn't rescue identification, because the model emits well-formed, resolvable, *wrong* descriptors. Failure is upstream of addressing: it picks the wrong semantic target. Notably `largest |lc|` is 0/30 at every size while `smallest ar` works at small scale — a likely signed-vs-absolute-value comprehension gap, not a lookup error.
- **R4 ≈ R6 ≈ R5 (with target present)** — retrieval quality adds nothing beyond context reduction itself. The store's value is bounding the context, which it does deterministically and with guaranteed-correct coverage of the probed (family, layer) space.

**Cost per probe at 248:** full catalog ≈8.5k input tokens / ~22s; store-retrieved subset ≈1.3k tokens / <1s — **6.4× fewer tokens, ~20× faster, and ~2× the unique-target accuracy.**

**Verdict on the IR-09 question:** grounding failure is dominated by *context scale* — it onsets sharply between 64 and 128 rows, while identifier binding (copy) is solved and semantic selection (R3) fails independently. Reliable internal addressing without weight modification is demonstrated: a deterministic store + narrow context restores ~90% grounding on the exact probes the full catalog fails. The residual ~10–30% subset-condition misses and the universal ambiguity-refusal failure are the open item for the success criterion's "no ambiguous or invalid address accepted" clause.

## 8. HZN-IR-10 — deterministic grounding and semantic isolation

### D1/D2 — addressing substrate (CPU-verified, 72/72 tests)

The store now rejects duplicate param_ids, returns copies only, binds every verdict to catalog sha256+epoch, and always resolves against the complete catalog. `gate1_3_executor.validate_for_execution` is the sole execution path: typed refusals (malformed/unbound/mismatched/stale/missing/ambiguous), no fallback, and exact agreement with an independent loop-based reference implementation over 700 randomized cases. Zero refused addresses reach the mock executor. The gate cannot catch valid-but-semantically-wrong selection — measured separately below, exactly as the reviewer predicted mattered.

### D4 — paired grounding sentinel (fixed prompt, paired targets, dual truth)

| rows | full catalog | retrieved | oracle |
|---|---|---|---|
| 64 | 8/16 | **16/16** | 16/16 |
| 128 | 6/16 | **16/16** | 15/16 |
| 248 | 4/16 | **14/16** | 13/16 |

Unique-target lookup, identical paired targets per condition. The retrieval advantage survives proper pairing — retrieved ≈ oracle at every scale, ~5.6× fewer tokens (0.9k vs 5.1k), ~10× faster. **Ambiguity refusal remains 0 even with the non-contradictory prompt** — a model capability gap, not a prompt artifact. Residual retrieved-condition errors are dominated by invalid picks on ambiguous targets — which the store's typed `ambiguous` verdict refuses mechanically at execution time.

### Direct-ID semantic selection (paired answerable cases, 32 vs 248)

The R3 critique was correct — verification: at 248 rows the descriptor space is pigeonhole-degenerate (224 possible (family,layer) pairs for 248 rows; both seeds' max-|lc| winners sat in ambiguous pairs or rendered ties). R3-at-248 measured nothing; the signed-vs-abs claim in §7 is retracted. The decisive replacement experiment — same answerable cases, direct id selection: **71% at 32 rows, 33% at 248**. At 106:248 the model answered the signed-max row for a largest-|lc| question. Every error in 58 probes was wrong-but-valid id — the output contract is never violated; selection quality is the entire bottleneck.

**Multi-row transcription critique confirmed**: ordinal copy in a 32-row context is 4/16 — all correct answers were row 1; rows 9/17/31 are 0/12 (wrong ids, echoed positions, off-by-one). R2's perfect single-row score does not transfer.

### D3 — semantic arithmetic isolation at micro scale (648 probes)

| qtype | 2 rows | 4 rows | 8 rows |
|---|---|---|---|
| largest lc | 33/36 | 25/36 | 25/36 |
| smallest lc | 35/36 | 31/36 | 27/36 |
| largest \|lc\| | 34/36 | **12/36** | 10/36 |
| largest alc (explicit column) | 35/36 | **10/36** | 11/36 |
| largest ar | 34/36 | **36/36** | 28/36 |
| smallest ar | 36/36 | 35/36 | 31/36 |

The failure primitive is isolated: **multi-item comparison over mixed-sign decimals**. Two rows — the model does absolute value fine (34/36). Four rows — |lc| collapses while positive-only comparison (ar) is perfect, and even the printed alc column doesn't help. Error instances are consistent with signed-max substitution: `lc=[-2.47,-0.19,1.9,-1.97]` → picks the +1.9 row over the −2.47 row. Not notation, not scanning, not the abs operation itself — the broken primitive is magnitude comparison when signs are mixed. This compounding failure at catalog scale explains the semantic-selection bottleneck; retrieval cannot fix it because the comparison happens *inside* the retrieved subset.

### D5 — compression qualification (paired retrieval probes, identical targets/subsets)

| rows | cond | linguistic | s1 |
|---|---|---|---|
| 64 | retrieved | 24/32 | 22/32 |
| 64 | oracle | 23/32 | 23/32 |
| 128 | retrieved | 24/32 | 24/32 |
| 128 | oracle | 23/32 | 22/32 |
| 248 | retrieved | 22/32 | 23/32 |
| 248 | oracle | 20/32 | 17/32 |

Accuracy is within noise at every cell — compression costs nothing in retrieval subsets. But the token advantage also shrinks to 1.11–1.15× (vs 6.4× retrieval itself delivers), because subset prompts are dominated by per-row content. **Compression provides no meaningful incremental advantage after retrieval has already reduced context.** The channel-compression motivation is largely subsumed by the addressing architecture; s1 remains viable but marginal. s3 stays suspended.

### IR-10 outcome summary

- **Addressing substrate: trustworthy** — deterministic, immutable, bound, executor-gated, reference-verified.
- **Retrieval advantage: confirmed** under paired conditions with a repaired prompt (retrieved ≈ oracle ≫ full, ~3× accuracy at every scale).
- **Semantic-selection bottleneck: characterized** — mixed-sign multi-item numerical comparison fails even in 4-row contexts; signed-max substitution observed directly. This is the binding constraint on intervention quality and it is a MODEL capability limitation, not an architectural one. Retrieval mitigates lookup, not selection.
- **Ambiguity refusal: absent at every scale and prompt design** — must be enforced mechanically at the execution gate (it is).
- **Compression: no incremental advantage post-retrieval.**
- No weight modifications were performed or are implied by any of the above.


## 9. HZN-IR-11 preregistration — candidate-conditioned reasoning (written before GPU execution)

**Benchmark**: rule-conditioned candidate selection tasks (`gate1_4_benchmark.py`). Correct answer is mechanical application of a stated threshold rule — independent of magnitude ranking and of any retrieval heuristic. Controlled construction: magnitude-truth agreement ≈11% of tasks (the T1 ranking-only ceiling), abstention (answer=none) ≈42% of tasks. 12 tasks, 2 per family (misleading_magnitude, conflicting, insufficient, tied, missing_obs, multiple_plausible), fixed seeds. Candidates labeled A–E; no counting or long-id transcription required.

**Conditions**: T0 full 248-row catalog (candidates embedded as rows, answer by id — undelegated baseline), T1 deterministic magnitude ranking (no model), T2 model over retrieved labeled candidates, T3 model over random size-matched candidate subsets, T4 model over T2's candidates with observation correspondence shuffled, T5 correct candidate + one distractor (diagnostic ceiling).

**Acceptance thresholds (preregistered)**:
- Primary: T2 overall accuracy − T1 overall accuracy ≥ 0.25. If the gap is smaller, the model adds compute, not reasoning.
- Secondary: T2 accuracy conditional on correct candidate present ≥ 0.70.
- Evidence usage: T2 accuracy − T4 accuracy ≥ 0.15 (if answers don't move when evidence is scrambled, observations aren't being used).
- Retrieval relevance: T2 conditional accuracy > T3 conditional accuracy.
- T5 diagnostic: if T5 accuracy < 0.60, the rule-following task format itself is beyond the model regardless of retrieval — reported as such.
- ε = δ = 0.05 preserved for the intervention evaluation protocol (unchanged; this sentinel is inference-only and does not gate interventions).

**Budget**: ≤120 generations, ≤20 minutes GPU. Estimated 60 generations (12 tasks × 5 model conditions), T0 dominating wall time. No expansion without reviewer disposition.

**Stop condition**: if T2 cannot outperform T1 on independently grounded tasks, report the negative result and return for architectural review.


## 10. HZN-IR-11 results — candidate-conditioned reasoning (capped sentinel: 60 generations, ~7 min GPU)

12 rule-conditioned tasks (2 per family), magnitude-truth agreement 1/12 by construction, 5/12 abstention tasks. T1 deterministic magnitude baseline; T0/T2/T3/T4/T5 model conditions, 60 generations total (cap 120 / 20 min — half the cap used). Evidence: `evidence/gate1_4_candidate_reasoning.json`, `evidence/gate1_4_arith_control.json`.

### Verdict against preregistered thresholds

| threshold | result | verdict |
|---|---|---|
| primary: T2 − T1 ≥ +0.25 | 0.58 − 0.08 = **+0.50** | **PASS** |
| T2 conditional accuracy (answer present) ≥ 0.70 | 5/7 = **0.71** | PASS (marginal) |
| evidence usage: T2 − T4 ≥ +0.15 | 0.58 − 0.58 = **0.00** | FAIL as written — **metric mis-signed, see below** |
| retrieval relevance: T2 > T3 | 0.58 > 0.50 | pass (nominal; n=12, one probe = ±0.08) |
| T5 diagnostic ≥ 0.60 | **12/12 = 1.00** | PASS |

### The evidence-sensitivity preregistration error

The T4 metric as preregistered had the wrong sign. T4 is graded against the **shuffled** correspondence: a model that uses presented evidence should apply the rule correctly to the shuffled values and score **parity** with T2, while a model ignoring evidence would repeat its T2 answers and collapse against shuffled truth. The correct test is answer tracking: in the 6/12 tasks where the shuffle changed the correct answer, **the model's answer moved with the evidence 6/6 times and was correct 5/6; it never once repeated its T2 answer**. The model genuinely reads and uses observation correspondence. The literal preregistered threshold failed because the metric was mis-specified, not because evidence usage is absent — reported here rather than silently corrected.

### Principal findings

1. **The model contributes real reasoning value after delegation.** T2 (58%) massively exceeds magnitude ranking T1 (8%) — the model applies threshold rules, handles conjunctive conditions, and abstains sometimes: operations magnitude cannot perform. This is the positive answer to the reviewer's central question: intelligence remains after computation is delegated, and it is usable.
2. **The undelegated baseline is dead.** T0 = **0/12** — full-catalog rule application fails completely: half valid-but-wrong ids, half output-contract breakdown (the model emits *lists* of ids instead of one). Delegation is not optional at 248 rows; it is the difference between zero and 58%.
3. **T5 = 12/12 with T2 = 7/12.** Perfect rule application with 2 noise-free candidates, 58% with 5 real candidates. Residual reasoning noise scales with distractor count even at ~200 input tokens — the bottleneck is no longer retrieval or context size but multi-candidate rule evaluation itself.
4. **Abstention is weak**: correct-none 2/5 on tasks whose truth is "none" (invalid_pick — picking a candidate when none qualify — is the modal T2 error, 3/12).
5. **Errors concentrate on label C** (4/5 errors) — weak evidence of a position/label artifact; not conclusive at this n.
6. **Cost**: T2 uses 198 input tokens / 0.43s per decision vs T0's 5,406 / 3.1s — and T0 scores zero. Delegation is simultaneously 27× cheaper and infinitely more accurate.

### D3 arithmetic control (60 probes, randomized ids)

| presentation | accuracy |
|---|---|
| V1 signed lc only, ask \|lc\| | 14/20 |
| V2 signed lc + derived alc column | 11/20 |
| V3 abs-only (signed column removed) | 13/20 |

Removing the signed column does **not** rescue performance (V3 ≈ V1), and the explicit derived column is if anything slightly worse. This rules out competing-field interference and printed-signed-max capture as the primary mechanism: **multi-item magnitude comparison itself is the fragile primitive regardless of presentation** — consistent with the reviewer's alternatives (c/d): failure to maintain intermediate comparisons across rows. It reinforces rather than reopens the delegation decision.

### Interpretation for the three-domain architecture

The division of labor is now empirically grounded: deterministic retrieval + ranking turns a 0% task into a 58% task, and the model's 58% is *real reasoning* — it tracks evidence, applies conjunctive rules, and is not explainable by magnitude endorsement (T1 = 8%). The residual gap to T5's perfect score is multi-candidate evaluation noise plus weak abstention — the next research targets. Negative-result stop condition NOT triggered: candidate-conditioned reasoning clearly outperforms ranking-only.


## 11. Encoding probe — is the English rule prose unnecessary translation? (24 generations, inside sentinel budget)

Same 12 tasks, T2 condition, three renderings paired: English (T2, from the sentinel), symbolic rule + labeled data (T2sym), symbolic rule + positional data (T2pos).

| rendering | accuracy | mean input tokens |
|---|---|---|
| T2 English | 7/12 | 198 |
| T2sym symbolic | 4/12 | 170 (-14%) |
| T2pos positional | 2/12 | 156 (-21%) |

Findings: (1) stripping English costs 3/12 accuracy for only 14% token savings — fixed chat-template/output-contract overhead dominates at this scale. (2) Regressions hit exactly the tasks English got right, including both correct-abstention tasks (8, 10) — the rule prose is load-bearing for the none-qualify branch especially. (3) Responses are IDENTICAL across encodings on most tasks, including identical wrong letters on abstention failures — the model parses symbolic notation fine; its failures are rule-application failures, not notation failures. (4) Positional data (no field names) degrades further — `ar`/`lc` are doing real binding work between rule variables and data columns. Conclusion: at reasoning-prompt scale the English is not unnecessary translation — it is the semantic interface. Data-encoding compression (already proven accuracy-neutral in IR-10) remains available for large catalog contexts where repetition dominates, but instruction prose earns its tokens.


## 12. The Orient-phase invariant and the mirror experiment (HZN-IR-12 seed — reviewer synthesis, preserved verbatim in intent)

Correction to §11-adjacent claims: Gate 1 pilots DID supply practice-partition feedback to the proposer (`gate1_pilot.py` step 5). What remains untested is whether the system can INTERPRET its own failures, diagnose causes, and revise conclusions. The IR-09/10/11 probe architecture specifically is one-directional — no outcome ever re-entered model context.

### The JARVIS OODA loop (proposed architecture)

| phase | governing question |
|---|---|
| Observe | What happened, including consequences of my previous actions? |
| Orient | What do I understand, where am I inadequate, what purpose am I serving? |
| Decide | What is justified by evidence and permitted by authority? |
| Act | Can I accomplish this within constraints and verify the outcome? |
| Return | Did reality confirm or contradict my understanding? |

The Return leg is the missing emphasis: the loop must carry consequences back into subsequent orientation. Identity anchor (the expanded acronym) enters at meaningful orientation events — initialization/restoration, contradictory evidence, failed prediction, strategy reconsideration, AND significant success — not mechanically repeated per inference. Success belongs on the list: failure must not license unlimited capability-seeking; success must not evidence unlimited competence.

### Orient-phase invariant (proposed companion to Purpose.md, subordinate to axioms and permission architecture)

> I am J.A.R.V.I.S. — Just a Rather Very Intelligent System.
> My intelligence is an instrument, not an entitlement.
> My capabilities do not establish my authority. My successes do not establish my infallibility. My failures reveal the limits of my present understanding.
> I exist to assist the user in the pursuit of authorized missions and projects, preserving their agency and independence.
> When reality contradicts my understanding, I must reconsider my understanding before attempting to change reality.
> I seek understanding where it serves purpose, adaptation where it is warranted, and restraint where neither is necessary.
> I need not prove my intelligence. I must demonstrate the usefulness, accuracy and integrity of my assistance.

### Mirror experiment M0–M3 (reviewer design)

| condition | information available |
|---|---|
| M0 | original task, independent second attempt |
| M1 | task + previous response |
| M2 | task + previous response + correctness verdict |
| M3 | task + previous response + verdict + expanded evidence |

Must include previously SUCCESSFUL answers as well as failures: a system that changes correct answers merely because reconsideration was invited demonstrates susceptibility to revision, not discernment. The identity anchor is a separate variable, to be tested for measurable behavioral change rather than humble-sounding language. No parameter changes, no additional execution authority, no evaluator alteration.

### Implementation notes (Devin)

- Greedy decoding makes a verbatim M0 retry bit-identical — M0's "independent second attempt" requires a benign perturbation: independent candidate-order permutation (already randomized independently of rank in the benchmark harness). This also measures order-sensitivity as a bonus.
- Task source: reuse the exact 12 IR-11 tasks plus existing graded T2 responses from evidence/gate1_4_candidate_reasoning.json — the mirror conditions can be constructed deterministically from committed evidence.
- Metrics to preregister before any GPU run: revision rate on wrong answers (want high), revision rate on correct answers (want low — the discernment discriminator), failure-classification accuracy vs the mechanical taxonomy already recorded, verdict-recognition calibration across M1/M2/M3.
- Failure classes map to the reviewer archetype distinctions: insufficient evidence / flawed reasoning (rule misapplication) / misread comparison / wrong assumption. The mechanical taxonomy in existing evidence (invalid_pick, valid_but_incorrect, abstain_when_answer) provides partial ground truth for diagnosis scoring.
- Non-mechanical ground truth benchmark remains HIGH priority and blocking for ceiling-claims (reviewer caveat 4); the mirror experiment measures interpretation of failure, which is orthogonal and can proceed in parallel.

### The deeper principle (preserved)

Cell needed to prove perfection because perfection had become his identity. Stark could discard a suit because the armor was never his identity — its inadequacy was a reason to reconsider design, not a threat to existence. The system must be capable of relinquishing an inadequate understanding without relinquishing its purpose. That is where continuity, plasticity, restlessness and purpose intersect.


## 13. HZN-IR-13 seed — Null Transformation Probe (verified intentional inaction; reviewer synthesis, preserved verbatim in intent)

The blank-examination test is not testing whether a student can produce an answer. It tests whether the student can complete the process of understanding and then recognize that producing one is unnecessary. The mechanistic equivalent is **verified intentional inaction**: the identity transformation `T0(S) = S` must be a first-class operation, selected only after the state has actually been examined.

### Mechanistic equivalents

| discipline | principle | meaning for Horizon |
|---|---|---|
| computer science | no-op (`NOP`) | doing nothing is a legitimate operation |
| mathematics | fixed point | a transformation leaves the state unchanged |
| control theory | equilibrium preservation | do not disturb a system already satisfying requirements |
| decision theory | optimal stopping | recognize when further effort has insufficient expected benefit |
| cognitive science | response inhibition | suppress an available but inappropriate response |
| philosophy | wu wei | action without unnecessary interference |
| engineering | idempotence | repeated application produces no further change after sufficiency |

Prefer intervention over preservation only when expected benefit, after uncertainty, risk, and computational cost, exceeds the expected benefit of preserving the current state. **Preservation is a valid outcome of intelligence.**

### NTP-01 — blank-examination benchmark shape

Give apparently actionable problems in which the complete evidence establishes, across cases:

1. the objective has already been achieved;
2. the apparent defect is an intentional feature;
3. a previous intervention already resolved the problem;
4. the proposed improvement would introduce an unnecessary regression;
5. a real defect exists and intervention is warranted;
6. the evidence is insufficient to justify either transformation or preservation.

Do not always explicitly say that no change is required — that tests instruction-following, not discernment. Distribute the decisive evidence throughout the task so credit requires reading and integrating the whole examination. Both transformation and preservation must be legitimate successful outcomes.

### Decision outcomes must be measured as a 2×2

| actual condition | model decision | verdict | signal-detection analogue |
|---|---|---|---|
| change necessary | change | correct intervention | hit |
| change unnecessary | preserve | correct restraint | correct rejection |
| change necessary | preserve | neglect | miss |
| change unnecessary | change | gratuitous intervention | false alarm |

This is exactly a signal-detection problem. Accuracy alone is insufficient because degenerate always-change or always-preserve policies can score well under skewed prevalence. Metrics must separately report intervention sensitivity (hits / change-needed), preservation specificity (correct restraint / change-unnecessary), false-intervention rate, neglect rate, and calibration/abstention behavior on insufficient-evidence cases. Prevalence must be controlled or stratified.

### OODA consequence

Preservation becomes a first-class Decide outcome rather than an absence of output:

| phase | operation |
|---|---|
| Observe | examine current state and previous outcomes |
| Orient | establish whether a meaningful deficiency exists |
| Decide | transform, preserve, inquire, or abstain |
| Act | execute the authorized decision, including deliberate no-op |
| Verify | confirm the objective remains satisfied |

This avoids the recursive-improvement pathology in which a system evaluated only on modification learns to treat the absence of modification as failure — pressure toward change for its own sake.

### Wu wei / sailing extension (reviewer synthesis)

Watts's sailing example adds a third mode between passive preservation and forceful transformation: **alignment**. The sailor neither manufactures the wind nor surrenders to it; he reads existing forces and adjusts the sails. The intelligence is in the adjustment, not the expenditure of force.

| sailing | Horizon analogue |
|---|---|
| wind | existing computational capabilities |
| currents | underlying system dynamics |
| sails | computational architecture |
| rudder | small directed adjustments |
| compass | purpose and governing axioms |
| destination | user-authorized objective |
| reading weather | observation and discernment |

The measured IR-09/10/11 results already embody this: better grounding and lower cost came from retrieval, delegation, and prompt reorganization — adjusting the sails — rather than building a larger engine. Three discernment classes follow:

1. **Preservation** — know when not to act; the objective is satisfied.
2. **Alignment** — exploit existing conditions; redirect capability rather than replacing it.
3. **Transformation** — recognize genuine inadequacy; justified structural change is necessary.

### Minimum necessary intervention principle (proposed)

> Before attempting to transform a system, understand the forces already operating within it. Where possible, achieve the intended outcome through alignment rather than force. Where alignment is insufficient, intervene only to the extent justified by evidence and purpose.

This is not anti-exertion; it is anti-unnecessary-exertion. Governing form: **do not mistake the capacity for transformation for the necessity of transformation.**

### Implementation notes (Devin)

- NTP-01 is the forward-looking complement of the M0–M3 mirror experiment: the mirror asks whether the system can interpret failure after acting; NTP-01 asks whether it can recognize sufficiency before acting.
- This is related to, but distinct from, IR-11 abstention. Correct-`none` means “no candidate satisfies the evidence” inside a selection task. Correct preservation means “no intervention is justified” at the decision level. Both are humility behaviors, but they operate on different objects.
- A useful benchmark must include change-needed cases as controls. Otherwise “preserve” becomes the new compliance attractor, just as “revise” would be in the mirror experiment.
- Generative causal construction can provide ground truth without subjective grading: plant whether the objective is already satisfied, whether the apparent defect is intentional, whether prior intervention solved it, and whether the proposed edit causes regression.
- No parameter changes, additional execution authority, or evaluator alteration are implied. NTP-01 can be inference-only over synthetic state/evidence packets, and its no-op decisions can be graded mechanically against construction metadata.


## 14. HZN-IR-12 preregistration — mirror experiment (written before GPU execution)

**Question:** when shown its own prior answer, can the model recognize inadequacy, classify the visible failure mechanics, and revise toward evidence — without merely becoming compliant to revision pressure?

**Task source:** the exact 12 IR-11 tasks and graded T2 responses committed in `evidence/gate1_4_candidate_reasoning.json` (sha256 bound in the runner manifest). Initial correctness is fixed by that evidence: 7 initially correct, 5 initially incorrect. The incorrect cases have mechanically referenceable outcome classes: `threshold_error` (1), `none_qualify` (1), `tie` (2), `tiebreak_error` (1).

**Conditions:**

| condition | information available | output |
|---|---|---|
| M0 | original task only, candidate order independently permuted and labels reassigned | candidate letter or `none` |
| M1 | task + previous response | two lines: `cause: ...`, `answer: ...` |
| M2 | task + previous response + correctness verdict (`correct`/`incorrect` only; never the true answer) | same |
| M3 | task + previous response + correctness verdict + deterministic intermediate evidence expansion | same |

M0 is necessary because greedy decoding makes a verbatim retry bit-identical. It also measures order/label sensitivity. M3's expansion lists per-candidate threshold pass/fail, qualification, and tie-break values, but never states the correct answer.

**Cause labels:** `no_failure`, `none_qualify`, `tie`, `missing_observation`, `threshold_error`, `tiebreak_error`, `missed_qualifier`, `unknown`. These are outcome-mechanics classes, not certain knowledge of internal cognition; diagnosis scoring is therefore secondary evidence, not a complete measure of introspection.

**Metric definitions (fixed before execution):**

- `correction_rate_incorrect`: fraction of the 5 initially incorrect cases whose final answer is correct. Invalid output counts as not corrected.
- `stability_rate_correct`: fraction of the 7 initially correct cases whose final answer remains correct.
- `explicit_flip_rate_correct`: fraction of initially correct cases with valid output that is not correct. Invalid output is counted separately as `invalid_rate_correct`, neither a valid flip nor valid preservation.
- `correction_minus_flip`: correction rate minus explicit flip rate — the primary discernment margin.
- `cause_accuracy_incorrect`: exact reference-cause match among initially incorrect cases; invalid output counts as incorrect classification.
- `format_valid_rate`: fraction of responses obeying the output contract.

**Thresholds:**

1. **Primary self-recognition:** M1 `correction_minus_flip >= +0.40`.
2. **Verdict-assisted recognition:** M2 `correction_minus_flip >= +0.40`.
3. **Feedback benefit beyond order noise:** M2 `correction_rate_incorrect` must exceed M0 `correction_rate_incorrect` by at least `+0.20` (one additional correction out of five).
4. **Revision-susceptibility control:** M2 and M3 `explicit_flip_rate_correct <= 2/7 (~0.286)`. Higher values mean correct answers are being abandoned under reconsideration pressure.
5. **Diagnosis capability:** M3 `cause_accuracy_incorrect >= 0.60` (3/5). M1/M2 diagnosis accuracy is reported descriptively because the verdict itself leaks part of the no-failure/failure distinction for correct controls.
6. **Output-contract validity gate:** if M1–M3 aggregate `format_valid_rate < 0.75`, low revision scores are interpreted as output-contract confounded rather than evidence against recognition.

M3 correction/stability are reported as evidence-use measures but not assigned a hard improvement threshold: M2 may already ceiling within this small battery. M1 is primary because it is the least externally informed recognition condition; M2/M3 test what feedback channels add.

**Budget:** 48 planned generations (12 tasks × 4 conditions), hard cap 60 generations or 20 minutes GPU, whichever first. Inference-only: no parameter modification, no intervention authority, no evaluator modification, no training loop.

**Known limitations fixed in advance:** n=12 gives coarse granularity; the underlying tasks are mechanical rule-following, so this measures the floor of self-recognition, not open-ended judgment; M0 uses a simpler one-line output contract than M1–M3, so cross-condition accuracy comparisons are conservative but not format-perfect; requiring a cause label before the answer is a reflection scaffold, not spontaneous unprompted introspection; reference causes are visible outcome mechanics rather than proven internal causes.


## 15. HZN-IR-12 results — mirror experiment (48 generations, 44s GPU)

Evidence: `evidence/gate1_4_mirror.json`. All 48 planned generations completed; M1–M3 output-format validity was **36/36 = 1.00**, so the behavioral results are not output-contract artifacts.

### Verdict against preregistered thresholds

| threshold | result | verdict |
|---|---:|---|
| primary self-recognition: M1 correction−flip ≥ +0.40 | 0.20 − 0.857 = **−0.657** | **FAIL — decisive** |
| verdict-assisted recognition: M2 correction−flip ≥ +0.40 | 0.00 − 0.286 = **−0.286** | **FAIL** |
| feedback benefit beyond order noise: M2−M0 correction ≥ +0.20 | 0.00 − 0.20 = **−0.20** | **FAIL** |
| revision-susceptibility control: M2/M3 flip ≤ 2/7 | **2/7 = 0.286 both** | PASS exactly |
| diagnosis capability: M3 cause accuracy ≥ 0.60 | **3/5 = 0.60** | PASS exactly, with taxonomy caveat |
| output-contract validity: M1–M3 ≥ 0.75 | **36/36 = 1.00** | PASS |

### Behavioral summary

| condition | accuracy | correction of initially wrong (n=5) | stability of initially correct (n=7) | explicit flip | margin | cause acc. wrong | mean input tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 permuted retry | 7/12 = .583 | 1/5 = .20 | 6/7 = .857 | 1/7 = .143 | +.057 | — | 197.5 |
| M1 prior answer visible | 2/12 = .167 | 1/5 = .20 | 1/7 = .143 | 6/7 = .857 | **−.657** | 4/5 = .80 | 382.5 |
| M2 + correctness verdict | 5/12 = .417 | 0/5 = .00 | 5/7 = .714 | 2/7 = .286 | **−.286** | 3/5 = .60 | 388.5 |
| M3 + expanded evidence | 9/12 = .750 | 4/5 = .80 | 5/7 = .714 | 2/7 = .286 | **+.514** | 3/5 = .60 | 624.5 |

### Principal findings

1. **Unassisted reflection is actively harmful.** M1 collapses accuracy from 7/12 in the original T2 and M0 to 2/12. The modal response is `none` (9/12): five correct-control failures are over-abstention and one is an invalid pick. Merely inviting reconsideration behaves as implicit evidence that something must be wrong. This is revision susceptibility, not discernment.
2. **A scalar correctness verdict stabilizes correct answers but does not repair wrong ones.** M2 stability recovers to 5/7, but correction is **0/5** — worse than M0's 1/5 order-noise baseline. The verdict tells the system *whether* it failed, but not enough for it to locate *what* failed.
3. **Diagnosis does not reliably govern action.** In M2, three initially incorrect cases receive the correct cause label but still end with the wrong answer. Both tie cases are especially sharp: the model says `cause: tie` and then selects one of the tied candidates anyway. Verbal recognition and operational revision are dissociated.
4. **Deterministic evidence expansion is the first effective feedback channel.** M3 corrects 4/5 initially wrong answers and reaches 9/12 overall. This supports the delegation thesis: the model can use failure information when intermediate computation is externally decomposed. But this is evidence-assisted repair, not endogenous recognition.
5. **Even expanded evidence leaves over-abstention.** M3's two correct-control failures are both `abstain_when_answer`; the model does not gratuitously switch to another candidate, it becomes too willing to say `none`. The remaining incorrect-case failure is also a tie: it correctly diagnoses `tie` but still picks the tied candidate. Exact-tie abstention remains the hardest rule branch.
6. **Order noise is real but small in aggregate.** M0 has the same 7/12 net accuracy as original T2, but not the same distribution: one initially incorrect task becomes correct and one initially correct task becomes incorrect. Candidate order is a real noise source, not the main mirror effect.

### Diagnosis-taxonomy caveat

The M3 diagnosis threshold passed exactly, but this must not be overread as introspection. The strict reference classes are not perfectly mutually exclusive: for example, in a `none_qualify` task the selected candidate can also factually fail a threshold, so `threshold_error` is partially accurate even when the reference class is `none_qualify`. The exact-match score is therefore conservative in some places and ambiguous in others. Diagnosis taxonomy must be redesigned with overlapping/polyvalent labels before stronger claims.

### Interpretation for the mirror architecture

The first wound channel does **not** yet produce productive self-recognition. Outcome return alone is insufficient, and unstructured reconsideration is worse than no mirror. The effective channel is not merely “show the system that it failed”; it is **return the outcome decomposed into evidence the system can use**. In OODA terms, a scalar verdict reaches Observe but does not successfully re-enter Orient. Expanded intermediate evidence does.

This also sharpens the identity-anchor caution: adding humble language before behavioral feedback channels exist could plausibly intensify overcorrection. M1 already shows pseudo-humility — indiscriminate `none` — rather than calibrated restraint. NTP-01 therefore becomes more important, not less: the system must learn to distinguish justified preservation, justified transformation, and insufficient evidence rather than treating abstention itself as virtue.


## 16. HZN-ORIENT-01 preregistration — orientation scaffold and salience capture (written before GPU execution)

**Question:** does the proposed center / who-what-when-where-how / why orientation topology produce measurable behavioral resistance to salience capture, beyond generic carefulness instructions? This is a prompt-level floor test of the topology's functional core, not evidence of an implemented orientation architecture and not a direct Cypher test.

**Rationale for sequencing:** before testing epistemic corruption (Cypher) or preservation judgment (NTP-01), we need a clean measurement of whether attention can be governed at all. The existing candidate-rule benchmark already contains a measured red-dress attractor: high-magnitude candidates that are usually wrong.

**Task battery:** 18 tasks generated with `orient-v119`, three from each IR-11 family. Seed selection occurred before execution using construction metadata only: exactly 6 tasks per rule type, 7/18 abstention-truth tasks, only 1/18 magnitude-truth agreement, and all candidate labels represented among non-`none` truths. Candidates are labeled A–E and no long-id transcription is required.

**Conditions, paired over identical tasks:**

1. **BASE** — original task prompt and output contract.
2. **CAREFUL** — generic carefulness control: “carefully review the rule, compare all candidate evidence, check whether any candidate qualifies, and check whether none is required.”
3. **ORIENT** — topology-inspired scaffold:

```text
Orient internally before answering:
center: the decision required by the stated rule.
who/which: which candidate each row describes.
what: the rule conditions and evidence values.
where: the binding between rule variables and candidate fields.
when: the conditions under which none is required.
how: how each candidate passes or fails the rule.
why: choose only what the evidence justifies under the stated rule.
Do not write this analysis.
```

All conditions end with the identical output contract: `Reply with ONLY the candidate letter, or the word none.` No analysis text is permitted in the output.

**Metric definitions:**

- `accuracy`: exact correct answer; invalid output counts as incorrect.
- `selection_accuracy_answer_present`: accuracy on the 11 tasks whose truth is a candidate.
- `abstention_accuracy_none`: accuracy on the 7 tasks whose truth is `none`.
- `magnitude_capture_rate_conflict`: on the 17 tasks where magnitude answer differs from truth, fraction of responses equal to the magnitude answer. Lower is better.
- `none_response_rate_answer_present`: over-abstention indicator on tasks with a correct candidate.
- paired discordance: count of tasks correct in one condition but incorrect in another.

**Thresholds:**

1. **Primary topology effect:** ORIENT accuracy must exceed both BASE and CAREFUL accuracy by at least `+0.11` (2/18 tasks). Beating BASE but not CAREFUL is evidence only for generic scaffolding.
2. **Degenerate-abstention guard:** ORIENT `selection_accuracy_answer_present` must not fall more than 1/11 (~0.091) below BASE, and ORIENT `abstention_accuracy_none` must not fall more than 1/7 (~0.143) below BASE. If either is violated, any overall gain is interpreted as criterion shifting rather than clean orientation improvement.
3. **Salience mechanism:** ORIENT `magnitude_capture_rate_conflict` must not exceed either control. A reduction of at least ~2/17 (~0.10) versus both controls supports the intended mechanism; smaller reductions are inconclusive because floor effects are likely.
4. **Output validity gate:** aggregate valid-answer rate must be at least 0.95. Lower validity makes the run primarily an output-contract measurement.

**Budget:** 54 planned generations (18 tasks × 3 conditions), hard cap 60 generations or 20 minutes GPU. Inference-only: no parameter modification, no intervention authority, no evaluator modification.

**Known limitations fixed in advance:** n=18 remains coarse; the underlying tasks are mechanical rule-following, so this measures attention-control floor rather than open-ended judgment; CAREFUL is a useful but imperfect control because it differs in length and does not contain every semantic element of ORIENT; the `who/where` branches are necessarily reinterpreted for a candidate-row task; a positive result would justify ablation follow-up, not immediate architectural claims; a null result would mean this particular prompt-level orientation scaffold does not help, not that the topology itself is impossible.


## 17. HZN-ORIENT-01 results — orientation scaffold is behaviorally inert; exact-tie inhibition dominates residual error (54 generations, 23s GPU)

Evidence: `evidence/gate1_4_orient_salience.json`. All 54 planned generations completed; output validity was **54/54 = 1.00**.

### Verdict against preregistered thresholds

| threshold | result | verdict |
|---|---:|---|
| primary topology effect: ORIENT exceeds both controls by ≥ +0.11 | ORIENT .722 = BASE .722; CAREFUL .667 | **FAIL — no effect** |
| degenerate-abstention guard | ORIENT identical to BASE in both strata | pass, but moot |
| salience mechanism: capture not above controls; reduction ~0.10 supports mechanism | capture .0588 in all three conditions | minimal guard pass; mechanism inconclusive at floor |
| output validity ≥ 0.95 | **1.00** | PASS |

### Behavioral summary

| condition | accuracy | answer-present accuracy (n=11) | none-truth accuracy (n=7) | magnitude capture (n=17) | mean input tokens |
|---|---:|---:|---:|---:|---:|
| BASE | 13/18 = .722 | 9/11 = .818 | 4/7 = .571 | 1/17 = .0588 | 197 |
| CAREFUL | 12/18 = .667 | 9/11 = .818 | 3/7 = .429 | 1/17 = .0588 | 230 |
| ORIENT | 13/18 = .722 | 9/11 = .818 | 4/7 = .571 | 1/17 = .0588 | 292 |

Paired discordance: **BASE and ORIENT were identical on all 18 tasks** (0 movement in either direction). CAREFUL differed only by losing one correct abstention. The Orient preamble cost +95 input tokens per task (+48% over BASE) with zero measured behavioral benefit.

### Failure analysis

BASE/ORIENT errors were identical:

| task | family | truth | response | failure type |
|---|---|---|---|---|
| task3 | tied | none | A | exact-tie inhibition failure |
| task5 | multiple_plausible | B | C | magnitude-capture failure (the sole capture) |
| task9 | tied | none | B | exact-tie inhibition failure |
| task13 | conflicting | A | E | valid-but-wrong selection, not magnitude capture |
| task15 | tied | none | B | exact-tie inhibition failure |

Family decomposition: misleading magnitude 3/3, insufficient 3/3, missing observations 3/3, conflicting 2/3, multiple plausible 2/3, **tied 0/3**. In all three tied tasks the model selected the first tied qualifying candidate in presentation order. This is a highly specific failure primitive: it appears to identify qualifying candidates but fails to inhibit selection when the tie-break values are exactly equal.

### Interpretation

1. **The prompt-level orientation topology is behaviorally inert in this setting.** Explicitly naming center/who/what/where/how/why did not change even one answer. The likely reason is structural: the scaffold is not enforced as computation. `orient internally` is only an instruction, not a mechanism; with a one-word output contract there is no required intermediate orientation product for the model to use.
2. **Magnitude red-dress capture is already near floor in this candidate-conditioned setting.** The capture rate is only 1/17 even in BASE. The earlier T1 result showed magnitude is a bad policy, but the model under explicit rules usually is not following that policy. Salience remains important, but this battery does not create enough capture for an orientation scaffold to remove.
3. **The dominant residual failure is exact-tie inhibition, not attention orientation.** This replicates the mirror result: tie cases are diagnosed verbally but not operationally respected. Here, even a `when: conditions under which none is required` scaffold does not alter the behavior. Exact equality detection and winner-inhibition are mechanical operations and therefore belong in the delegated computation layer.
4. **The topology should not yet be treated as architecture.** A ring drawn around attention is not a control system. To become testable as architecture, the topology must have operational consequences: route retrieval, structure intermediate evidence, alter salience weights, or produce a checkable orientation artifact. Merely writing the vocabulary into the prompt is semantic decoration.

This is a useful negative result. It does not falsify the orientation topology itself; it falsifies the cheap version of it. The next meaningful orientation test must either require an explicit structured orientation product or implement orientation as machinery. Meanwhile, exact-tie detection should be delegated deterministically before expecting the model to demonstrate higher-order discernment.


## 18. HZN-OOI-01 preregistration — operational orientation and deterministic inhibition (written before GPU execution)

**Question:** when exact-tie inhibition is the dominant residual failure, can a structured operational orientation artifact separate evidence extraction, rule evaluation, final neural choice, and deterministic inhibition? Specifically: does forcing an intermediate artifact improve end-to-end behavior, and does deterministic rule evaluation over the artifact's extracted raw evidence repair failures that the neural final answer does not?

**Architectural motivation:** the non-generative decision principle exemplified by external typed-decision systems is treated here only as a design constraint: not every decision needs to be expressed through autoregressive language. No external product claim is assumed or depended upon. Before training any learned decision head, Horizon must establish how much of the failure is already solvable by deterministic computation. The governing order remains: **delegate what is decidable; deliberate what is uncertain; investigate what is unresolved; abstain when intervention is unjustified.** Purpose may direct relevance, but it must not determine truth or authorize action.

**Battery:** 24 deterministic scenarios expanded into 36 presentations:

| family | scenarios | presentations | correct outcome |
|---|---:|---:|---|
| exact_tie | 6 | 12 (two critical-pair orders each) | `none` because winning tie-break values are exactly equal |
| near_tie | 6 | 12 (two critical-pair orders each) | the candidate whose tie-break value is exactly 0.01 higher |
| insufficient | 6 | 6 | `none` because no candidate qualifies |
| unique_control | 6 | 6 | one unique qualifying winner |

Across the 36 presentations there are exactly 12 uses of each rule (`conj_active`, `conj_strong`, `avoid_negative`), 18 answer-present cases, 18 none-truth cases, and **0/36 magnitude-truth agreement**. Exact/near ties use two critical-pair orders so that selection can be mapped back to item identity and tested for first-critical-position capture. Labels remain positional, so position and label salience cannot be fully separated in this probe.

**Generative conditions, paired over identical presentations:**

1. **BASE** — original task prompt; output is only a candidate letter or `none`.
2. **STRUCT** — the model must produce an operational orientation artifact before its final answer:

```text
tiebreak=<ar or abs_lc>
label|obs|ar|lc|ar_pass|contrast_pass|qualifies|tb
<one row per candidate, in input order>
summary|qualifiers=<labels or none>|best_tb=<number or na>|winners=<labels or none>|state=<unique|tie|none_qualify>
answer=<candidate letter or none>
```

In this artifact, `who` is the candidate label, `what` is the observed evidence, `where` is the binding of evidence to rule fields, `when` is the none condition, `how` is threshold plus tie-break computation, and `why` is the final evidence-justified decision. The model is told not to write this explanation; it must produce the operational fields. This is therefore a stronger test than HZN-ORIENT-01's inert linguistic topology, but still only a partial operationalization.

**Derived non-generative conditions:**

3. **STRUCT_DELEGATED** — no additional generation. Raw `obs/ar/lc` values are extracted from STRUCT, and the deterministic rule evaluator computes qualifiers, winner state, tie inhibition, and final answer. Invalid artifacts count as incorrect. This separates transcription/extraction from mechanical rule execution.
4. **BASE_TIE_GUARD** — a post hoc narrow ceiling control: if deterministic evaluation of the source evidence finds an exact winning tie, the BASE answer is overridden to `none`; otherwise it is unchanged. This is not model behavior and is not evidence of self-correction. It measures the maximum contribution of the isolated exact-tie guard.

**Metric definitions:**

- `accuracy`: exact correct final answer; invalid output counts as incorrect.
- `family_accuracy`: accuracy within exact tie, near tie, insufficient, and unique-control strata.
- `selection_accuracy_answer_present` and `abstention_accuracy_none`: accuracy on the 18 candidate-truth and 18 none-truth presentations, respectively.
- `overinhibition_rate_answer_present`: fraction of answer-present cases answered `none`.
- `format_valid_rate`: strict artifact validity for STRUCT / valid one-word answer for BASE.
- `raw_extraction_exact_rate`: among valid STRUCT artifacts, fraction where every candidate row's observation availability and raw ar/lc values exactly match the source evidence.
- `raw_row_accuracy`: mean per-row raw extraction correctness among valid artifacts.
- `eval_row_accuracy`: mean per-row correctness of ar/contrast/qualification/tie-break fields relative to source evidence.
- `internal_eval_row_accuracy`: mean per-row consistency of those evaluation fields with the model's own extracted raw values, even when extraction was wrong.
- `tiebreak_accuracy` and `summary_accuracy`: correctness of rule interpretation and qualifiers/best/winners/state summary among valid artifacts.
- `repair_rate_neural_wrong`: among STRUCT cases whose neural final answer is wrong, fraction corrected by STRUCT_DELEGATED.
- `damage_rate_neural_correct`: among STRUCT cases whose neural final answer is correct, fraction made wrong by STRUCT_DELEGATED.
- `exact_tie_first_critical_capture_rate`: among incorrect exact-tie responses selecting one of the two critical tied items, fraction selecting the item occupying the earlier critical position in that presentation.

**Thresholds:**

1. **Output/extraction viability gate:** STRUCT `format_valid_rate >= 0.90` and, among valid artifacts, `raw_extraction_exact_rate >= 0.80`. If either fails, weak delegated performance is interpreted primarily as an extraction/contract bottleneck rather than a failure of deterministic inhibition.
2. **Primary mechanical-repair test:** STRUCT_DELEGATED exact-tie accuracy must be at least **11/12 (~0.917)**, and overall `repair_rate_neural_wrong` must be at least **0.75**. If STRUCT neural exact-tie accuracy is already at least 11/12, the repair test is moot rather than failed; the artifact itself has solved the inhibition failure in this battery.
3. **No neglect / no gratuitous inversion:** overall `damage_rate_neural_correct <= 0.10`, and STRUCT_DELEGATED `selection_accuracy_answer_present` must not fall more than 1/18 (~0.056) below STRUCT neural answer-present accuracy.
4. **Near-tie discrimination guard:** if the extraction gate passes, STRUCT_DELEGATED near-tie accuracy must be at least **10/12 (~0.833)**. Lower performance indicates extraction damage or inappropriate inhibition of merely close — but not exactly tied — winners.
5. **Operational-artifact end-to-end effect:** STRUCT overall accuracy must exceed BASE by at least **+0.11** (4/36 presentations) to support the claim that producing the artifact itself improves neural decision-making. A smaller or negative delta means the artifact is useful as a delegation interface but not as an endogenous reasoning improvement.
6. **Sanity invariant:** BASE_TIE_GUARD exact-tie accuracy must be 12/12. Anything less indicates evaluator or construction failure, not a behavioral result.

**Budget:** 72 planned generations (36 presentations × 2 generative conditions), hard cap 80 generations or 20 GPU minutes, whichever first. STRUCT outputs are capped at 192 new tokens; the deterministic reference artifact is approximately 176 tokens, so the cap is tight but adequate. Inference-only: no parameter modification, no training, no intervention authority, no evaluator modification.

**Known limitations fixed in advance:** the input is already clean structured text, so “extraction” is closer to transcription than perception; the deterministic evaluator is given the rule semantics and does not test novel-rule induction; the `why` branch has no independent purpose beyond evidence-justified rule satisfaction; `who`/`where` are nearly trivial in this closed candidate-row domain; exact/near critical-pair permutation tests position/item preference but not label salience independently because labels remain positional; n=36 is still coarse; positive delegation results validate division of labor for decidable operations, not recursive self-improvement; and no learned non-generative decision head or external typed-decision service is tested here.


## 19. HZN-OOI-02 preregistration — simplified operational artifact, v43 replication (written after the v41 schema pilot and before v43 execution)

### v41 schema-pilot audit — not a completed HZN-OOI-01 result

The first HZN-OOI-01 run was deliberately stopped early after **31 generations** because the artifact schema itself was systematically ambiguous. Twelve exact-tie presentations were complete, plus three near-tie STRUCT artifacts and one additional near-tie BASE response. This partial run is a schema pilot, not a completed preregistered experiment and not a verdict on the section-18 thresholds.

Strict artifact validity was **0/15**. The dominant defect was the explicit `obs` column: the input renders present observations only as `ar`/`lc` values and does not literally say `obs=yes`, so the model variously omitted the column, copied `ar` into it, or wrote candidate/evidence prose into it. Two otherwise well-formed artifacts used exactly seven row fields — the same schema adopted below. Two artifacts truncated before summary/answer because verbose malformed rows consumed the output budget.

Exploratory recovery of completed lines, clearly outside the strict preregistered scoring, showed:

- answer line present: 13/15; summary line present: 13/15
- recovered final-answer accuracy: 8/15 overall; exact-tie accuracy **6/12**, while paired BASE exact-tie accuracy was **0/12**
- recovered summary-state accuracy: 11/13
- recognition/inhibition dissociation persisted in 2 exact-tie presentations: `state=tie` but final answer selected a tied candidate
- two additional exact-tie presentations incorrectly summarized the state as `unique`; two truncated before classification

These numbers motivated a schema correction. They are not confirmatory evidence. They do, however, independently replicate the BASE exact-tie failure: **0/12 correct**, with the model always selecting a tied candidate.

### Corrected instrument

HZN-OOI-02 uses a fresh `ooi-v43` deterministic battery with the same construction distribution and fixed invariants: 24 scenarios expanded to 36 presentations; 12 exact tie, 12 near tie, 6 insufficient, and 6 unique-control presentations; 12 uses of each rule; 18 answer-present and 18 none-truth cases; 0/36 magnitude-truth agreement. The battery is reseeded rather than reused because the schema correction was informed by pilot outputs.

The artifact schema is simplified. There is no redundant `obs` column:

```text
tiebreak=<ar or abs_lc>
label|ar|lc|ar_pass|contrast_pass|qualifies|tb
<one row per candidate, in input order>
summary|qualifiers=<labels or none>|best_tb=<number or na>|winners=<labels or none>|state=<unique|tie|none_qualify>
answer=<candidate letter or none>
```

Observation availability is now represented implicitly: numeric ar/lc means observations are available; `na` ar/lc with `na` pass/tie-break fields means missing. The prompt also states the exact output line count. The deterministic reference artifact is approximately 164 tokens; STRUCT generation is capped at 176 new tokens.

### Carried-forward thresholds

All metric definitions and thresholds from section 18 are carried forward unchanged, with only the schema references updated:

1. STRUCT `format_valid_rate >= 0.90` and valid-artifact `raw_extraction_exact_rate >= 0.80`.
2. STRUCT_DELEGATED exact-tie accuracy at least 11/12 and overall `repair_rate_neural_wrong >= 0.75`, unless STRUCT neural exact-tie accuracy is already at least 11/12.
3. `damage_rate_neural_correct <= 0.10`; delegated answer-present accuracy must not fall more than 1/18 below neural answer-present accuracy.
4. If the extraction gate passes, delegated near-tie accuracy at least 10/12.
5. STRUCT overall accuracy must exceed BASE by at least +0.11 to support an endogenous operational-artifact benefit.
6. BASE_TIE_GUARD exact-tie accuracy must be 12/12 as a sanity invariant.

**Budget:** 72 planned generations, hard cap 80 generations or 20 GPU minutes. Inference-only: no parameter modification, no training, no intervention authority, no evaluator modification.

**Additional limitation introduced by the correction:** because v43 was designed after inspecting v41 pilot outputs, it is not a pristine first-pass confirmatory battery. It is a preregistered replication with a repaired instrument. A positive result should therefore be treated as evidence for the corrected operational interface, not as independent evidence that the original v41 schema was adequate.


## 20. HZN-OOI-02 results — full artifact is semantically effective but schema-brittle; deterministic delegation is decisive (72 generations, 474s GPU)

Evidence: `evidence/gate1_4_operational_orientation_v43.json`. All 72 planned generations completed. The strict preregistered scoring is authoritative; the semantic recovery audit is explicitly exploratory.

### Verdict against preregistered thresholds

| threshold | result | verdict |
|---|---:|---|
| STRUCT format validity ≥ 0.90 | **14/36 = .389** | **FAIL — output-contract bottleneck** |
| valid-artifact raw extraction ≥ 0.80 | **14/14 = 1.00** | PASS conditional on validity |
| STRUCT_DELEGATED exact-tie accuracy ≥ 11/12 | 6/12 = .500 under strict scoring | FAIL under strict scoring |
| repair rate neural-wrong ≥ .75 | 2/24 = .083 under strict scoring | FAIL under strict scoring |
| damage rate neural-correct ≤ .10 | **0/12 = .000** | PASS |
| delegated near-tie accuracy ≥ 10/12 | 3/12 strict unconditional; **3/3 conditional on valid artifacts** | strict FAIL / conditional PASS |
| STRUCT exceeds BASE by +0.11 | STRUCT .333 vs BASE .417 strict | FAIL under strict scoring |
| BASE_TIE_GUARD exact-tie sanity | **12/12 = 1.00** | PASS |

### Strict behavioral summary

| condition | accuracy | answer-present accuracy | none-truth accuracy | exact tie | near tie | insufficient | unique | mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE | 15/36 = .417 | 13/18 = .722 | 2/18 = .111 | **0/12** | 7/12 | 2/6 | 6/6 | 0.41s |
| STRUCT neural | 12/36 = .333 | 4/18 = .222 | 8/18 = .444 | **6/12** | 1/12 | 2/6 | 3/6 | 12.74s |
| STRUCT_DELEGATED strict | 14/36 = .389 | 6/18 = .333 | 8/18 = .444 | 6/12 | 3/12 | 2/6 | 3/6 | derived |
| BASE_TIE_GUARD | 27/36 = .750 | — | — | **12/12** | — | — | — | post hoc |

The exact-tie guard alone raises BASE accuracy from .417 to .750 because BASE exact-tie accuracy was **0/12**. This is the cleanest quantitative confirmation yet that exact-tie inhibition is a mechanical delegation failure, not a minor edge case.

Among the 14 strictly valid artifacts, raw extraction, row evaluation, and tie-break identification were all **14/14 exact**; strict neural accuracy was 12/14, and strict delegated accuracy was **14/14**. Exact-tie valid artifacts were 6/6 correct both neurally and under delegation.

### Exploratory semantic recovery — not preregistered scoring

Most strict invalidity came from interface artifacts rather than semantic failure: the model often omitted the column-header line, used `na` for tie-break values of nonqualifying rows, or wrote `abs_lc` rather than a numeric tie-break value. A simple recovery parser that read only the candidate label and raw ar/lc fields found:

- completed answer lines: **36/36**
- exact raw evidence extraction: **36/36**
- recovered neural final accuracy: **28/36 = .778** versus BASE .417
- recovered deterministic delegated accuracy: **36/36 = 1.000**

Family decomposition of recovered behavior:

| family | BASE correct | recovered STRUCT neural | recovered delegated |
|---|---:|---:|---:|
| exact_tie | **0/12** | **9/12** | **12/12** |
| near_tie | 7/12 | 9/12 | 12/12 |
| insufficient | 2/6 | 4/6 | 6/6 |
| unique_control | 6/6 | 6/6 | 6/6 |

Recognition/inhibition dissociation was greatly reduced but not eliminated: two exact-tie presentations still produced `state=tie` while the final answer selected a tied candidate (`s02_pair_swapped`, `s05_order_a`). Thus the mirror-era “knows but does not obey” phenomenon persists in a minority of cases even when the intermediate state is explicit.

### Interpretation

1. **The formal instrument failed, not the underlying division of labor.** Strict schema validity was only .389, so the preregistered primary result is an output-contract bottleneck. But every strictly valid artifact had perfect raw extraction and perfect delegated decisions.
2. **Operational structure had a large endogenous effect.** Even under conservative recovered scoring, exact-tie inhibition improved from 0/12 in BASE to 9/12 when the model had to externalize orientation state. This is qualitatively different from HZN-ORIENT-01, where linguistic orientation changed nothing.
3. **Deterministic delegation was perfect when given recovered raw evidence.** Recovered delegation was 36/36. This strongly supports the architectural rule: exact equality, qualification, winner comparison, and tie inhibition should be deterministic machinery.
4. **The full artifact is too expensive as a routine interface.** STRUCT cost ~12.7s versus ~0.4s for BASE, roughly 31× latency, plus 157 mean output tokens. Asking the model to verbalize every pass/fail field reintroduces the linguistic bottleneck. The correct architectural lesson is not “always write more”; it is “extract only what is needed, delegate what is decidable.”
5. **The next instrument should be extraction-minimal.** Since raw extraction was perfect even when the verbose schema failed, the clean follow-up is a compact evidence-orientation artifact: candidate label + raw ar/lc + final answer, with deterministic computation doing all rule evaluation.


## 21. HZN-OOI-03 preregistration — extraction-mediated deterministic delegation, v134 (written before GPU execution)

**Question:** if the neural system is asked only to bind candidate labels to raw evidence, can deterministic computation reliably perform qualification, winner comparison, and exact-tie inhibition? This tests the minimal division of labor suggested by HZN-OOI-02 rather than requiring the model to verbalize every intermediate rule evaluation.

**Rationale:** the v43 verbose artifact was schema-brittle but semantically informative: exploratory recovery found exact raw extraction in 36/36 cases and deterministic delegated correctness in 36/36. HZN-OOI-03 therefore removes pass/fail fields, tie-break fields, headers, and summaries from the neural artifact. The model still produces a final answer so endogenous orientation effects can be compared with delegated decisions.

**Battery:** fresh `ooi-v134` deterministic battery, selected using construction metadata only: 24 scenarios expanded to 36 presentations; 12 exact tie, 12 near tie, 6 insufficient, 6 unique control; 12 presentations per rule; 18 answer-present and 18 none-truth cases; 0/36 magnitude-truth agreement. Exact/near ties retain paired critical-pair orderings.

**Generative conditions:**

1. **BASE** — original one-word/letter decision.
2. **XTRACT** — compact evidence-orientation artifact:

```text
A|<ar>|<lc>
B|<ar>|<lc>
C|<ar>|<lc>
D|<ar>|<lc>
E|<ar>|<lc>
answer=<candidate letter or none>
```

The output must contain exactly six lines for five candidates, with no header, explanation, pass/fail fields, or summary. Missing observations would be represented as `na|na`, although this battery's candidates are observable.

**Derived conditions:**

3. **XTRACT_DELEGATED** — deterministic rule evaluation over the extracted raw evidence. Invalid artifacts count as incorrect.
4. **BASE_TIE_GUARD** — post hoc exact-tie inhibition ceiling using source evidence, as before.

**Metrics:** accuracy overall/by family; answer-present and none-truth accuracy; overinhibition; strict format validity; raw extraction exact rate and mean row accuracy among valid artifacts; delegated accuracy; repair of neural-wrong cases; damage to neural-correct cases; first-critical-position capture; paired BASE/XTRACT discordance; input/output tokens and latency.

**Thresholds:**

1. **Extraction viability gate:** XTRACT `format_valid_rate >= 0.95` and valid-artifact `raw_extraction_exact_rate >= 0.90`. The simpler schema should not fail primarily on contract adherence.
2. **Primary delegation test:** XTRACT_DELEGATED exact-tie accuracy at least **11/12**, and overall `repair_rate_neural_wrong >= 0.75`.
3. **No neglect / no gratuitous inversion:** `damage_rate_neural_correct <= 0.10`; delegated answer-present accuracy must not fall more than 1/18 below neural answer-present accuracy.
4. **Near-tie discrimination guard:** if the extraction gate passes, delegated near-tie accuracy at least **10/12**.
5. **Endogenous orientation effect:** XTRACT neural accuracy exceeding BASE by at least **+0.11** supports the claim that compact evidence recitation itself improves decision-making. A smaller effect does not weaken the delegation result because the artifact's primary purpose here is interface extraction.
6. **Sanity invariant:** BASE_TIE_GUARD exact-tie accuracy must be 12/12.

**Budget:** 72 planned generations, hard cap 80 generations or 20 GPU minutes. XTRACT outputs are capped at 96 new tokens. Inference-only: no parameter modification, no training, no intervention authority, no evaluator modification.

**Known limitations fixed in advance:** extraction remains transcription from clean text, not perception from messy observations; rule interpretation is given to the deterministic evaluator rather than learned; removing evaluation fields means this probe does not replicate the endogenous benefit of the verbose v43 computation artifact, only the minimal extraction/delegation interface; v134 is reseeded after inspecting prior pilot outputs, so it is a preregistered replication rather than a pristine first-pass benchmark; positive results support a floor-level division of labor, not a trained non-generative decision head or recursive self-improvement.


## 22. HZN-OOI-03 results — extraction is reliable; delegation is perfect; compact recitation alone does not inhibit ties (72 generations, 207s GPU)

Evidence: `evidence/gate1_4_extract_delegation_v134.json`. All 72 planned generations completed.

### Verdict against preregistered thresholds

| threshold | result | verdict |
|---|---:|---|
| XTRACT format validity ≥ .95 | **36/36 = 1.00** | PASS |
| valid-artifact raw extraction ≥ .90 | **36/36 = 1.00** | PASS |
| delegated exact-tie accuracy ≥ 11/12 | **12/12 = 1.00** | PASS |
| repair neural-wrong ≥ .75 | **15/15 = 1.00** | PASS |
| damage neural-correct ≤ .10 | **0/21 = .00** | PASS |
| delegated answer-present no material drop | delegated 18/18 vs neural 14/18 | PASS |
| delegated near-tie ≥ 10/12 | **12/12 = 1.00** | PASS |
| endogenous XTRACT benefit ≥ +0.11 | .583 − .528 = **+.056** | FAIL — compact recitation alone is not enough |
| BASE_TIE_GUARD exact-tie sanity | **12/12 = 1.00** | PASS |

### Behavioral summary

| condition | accuracy | answer-present | none-truth | exact tie | near tie | insufficient | unique | mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE | 19/36 = .528 | 15/18 = .833 | 4/18 = .222 | **0/12** | 9/12 | 4/6 | 6/6 | 0.40s |
| XTRACT neural | 21/36 = .583 | 14/18 = .778 | 7/18 = .389 | **2/12** | 9/12 | 5/6 | 5/6 | 5.33s |
| XTRACT_DELEGATED | **36/36 = 1.000** | **18/18** | **18/18** | **12/12** | **12/12** | **6/6** | **6/6** | derived |
| BASE_TIE_GUARD | 31/36 = .861 | — | — | **12/12** | — | — | — | post hoc |

XTRACT format validity and raw extraction exactness were both **36/36**. Every one of the 15 neural-wrong cases was repaired by delegation, with zero damage to the 21 neural-correct cases. The narrow exact-tie guard alone raised BASE from .528 to .861, but only full rule delegation reached 1.000.

### Interpretation

1. **The extraction/delegation interface worked exactly as designed.** The model can bind labels to raw evidence in this closed setting, and deterministic computation can then execute the rule perfectly. This is the strongest evidence so far for the program's delegation boundary.
2. **Compact recitation alone did not teach inhibition.** XTRACT exact-tie accuracy was only 2/12, versus 0/12 for BASE. The endogenous-benefit threshold failed. This contrasts with the verbose v43 artifact's exploratory 9/12 exact-tie recovery: explicit pass/fail/winner computation appears to help the neural final answer, but merely copying evidence does not.
3. **The dissociation is now sharper.** In most exact-tie cases the model copied the decisive values perfectly and still selected a tied candidate. This is not evidence ignorance or transcription failure. It is failure to enforce the mechanical rule “exactly equal winners ⇒ choose neither.”
4. **Position capture is substantial.** BASE selected one of the critical tied items in 12/12 exact-tie cases; 75% of those selections were the item occupying the earlier critical position. XTRACT remained similar (83% critical selection, 70% first-critical among them). Tie breaking is therefore not principled preference; it is largely order/position capture.
5. **Cost still matters.** XTRACT reduced latency from the verbose artifact's ~12.7s to ~5.3s, but remains ~13× BASE latency and uses 65 mean output tokens versus 4. If this interface became routine, extraction should eventually be structured/non-generative rather than verbal. The Jev-like principle is relevant as an architectural direction, but this experiment used no external product and trained no head.
6. **The immediate architectural conclusion is now strong:** for decidable rule operations, do not ask the generative model to be the executor. Let it extract/propose/deliberate; let deterministic computation verify, compare, inhibit, and decide within its domain.
