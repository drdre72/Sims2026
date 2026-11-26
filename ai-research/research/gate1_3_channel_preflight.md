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
