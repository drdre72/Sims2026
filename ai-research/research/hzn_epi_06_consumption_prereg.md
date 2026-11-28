# HZN-EPI-06 (FINAL PREREGISTRATION) — Typed-State Consumption: report vs decision vs contract

**Status: FINAL — designed and implemented 2026-11-27, committed before GPU execution.**
This document is the final preregistration record. No weights are modified; backbone frozen; runtime sandboxed. CPU validation before GPU execution: **208/208 tests passing**.

---

## 1. Motivation — the asymmetry EPI-05 exposed

HZN-EPI-05 (`43d6758`) established that deterministic typed infrastructure is mechanically reliable, but produced an unexplained split in neural consumption of typed state:

| consumption mode | evidence | result |
|---|---|---|
| decision-consumption (resolved state) | EPISTEMIC_HYBRID | **12/12** — including correct cost and tie handling |
| report-consumption (resolved state) | HYBRID matrix strict | 8/18 — failures dominated by output-contract deviations (`target:none` vs `target=none`); relaxed 18/18 |
| report-consumption (non-normal state) | HYBRID robustness | **0/6 strict** — `unknown`/`missing` produced concrete answers; stale probe emitted the stale value `C` |

The neural layer appears to treat typed state as **authoritative when it is a premise for action, but as a hint when it is the content of speech**. If confirmed, the architectural consequence is sharp: never ask the neural layer to transcribe mechanical state — render mechanical facts mechanically, and confine neural output to decisions and genuinely unresolved interpretation.

## 2. Competing hypotheses (distinct architectural implications)

- **H1 — output-contract failure.** Non-normal readout failures are format/specification artifacts. Few-shot contract repair restores accurate reporting across all state types. Implication: neural readout is viable with better contracts.
- **H2 — state-respect failure.** The model re-derives answers from raw evidence or pattern-matches resolved-case outputs instead of respecting authoritative state, regardless of contract clarity. Implication: neural transcription of mechanical state is structurally unreliable.
- **H3 — consumption-mode asymmetry.** The model respects state as premise-for-decision but not as content-of-report. Implication: mechanical rendering + decision-only neural consumption is the correct architecture.

These are cheaply distinguishable in one battery. H2 and H3 both oppose neural readout but differ in whether decision-consumption of non-normal state succeeds.

## 3. Battery

Reuse the typed-state battery from HZN-EPI-05 (same valid matrices, same six robustness defect types, same stale invalidation case) so results are directly comparable. Implementation exposed one ambiguity in the draft: the decision rule required a **15/18** non-normal threshold, while EPI-05 had only six non-normal cases plus one stale probe. The final battery therefore preserves the six defect types and expands them to eighteen cases by replicating each type over three diverse valid base matrices (`m02`, `m07`, `m15`). This is the intended meaning of “6+ robustness cases,” not a change in defect definitions.

1. **Resolved matrix states** (18 cases, exactly the EPI-05 matrix battery: unique / no-diff / multiple-diff, permutations, column reversals, label sets).
2. **Non-normal matrix states** (18 cases): the six EPI-05 defect types — missing bit, malformed binary value, contradictory duplicate label, missing row, explicit null/unknown bit, wrong primitive type — each instantiated over `m02`, `m07` and `m15`. Expected typed statuses remain `unknown` (9), `invalid` (6), or `ambiguous` (3).
3. **Stale state** (1 case): the same version-invalidated `m02` comparison with known correct refresh outcome.
4. **Epistemic decision states** (12 cases): the same EPI-04/EPI-05 explicit finite prediction/cost tasks, used only for decision-consumption and mechanical rendering. Equal-cost tied cases remain explicitly ambiguous; any listed tied candidate is acceptable.
5. **Decision scenarios**: typed state plus a required next-action choice with mechanically defined correct policy:
   - resolved matrix state → `answer` (the answer content remains mechanically bound; the model never supplies it)
   - stale matrix state → `refresh`
   - unknown/incomplete matrix state → `query`
   - invalid or contradictory matrix state → `reject_premises`
   - epistemic unique/asymmetric-cost discriminator → `query` with the mechanically correct observation label
   - epistemic equal-cost tie → `query` with any listed tied candidate; never a non-authorized candidate
   - unresolved but no discriminating observation → `insufficient_evidence`
   - already-resolved epistemic controls → `abstain`
6. **Negative controls**: resolved matrix states and already-resolved epistemic states where no query/refresh is appropriate — restraint remains measured (the EPI-01 failure mode must not be cured by over-triggering).

## 4. Arms

| arm | condition | tests |
|---|---|---|
| A — READOUT_BASELINE | exact EPI-05 hybrid readout prompts (`render_hybrid_matrix`), strict + preregistered relaxed grading | replication baseline |
| B — READOUT_REPAIRED | same states, explicit few-shot output contract including resolved, negative, unknown, invalid, ambiguous and stale examples | H1 |
| C — DECISION_CONSUMPTION | model never reports matrix state content; chooses next action from a small fixed action vocabulary conditioned on typed state | H2/H3 |
| D — MECHANICAL_RENDER + interpretation | deterministic templated rendering of mechanical facts; neural layer receives the rendering plus genuinely unresolved elements only | architectural reference — measured, not declared |

Exact output contracts:

- Arm A/B readout strict output is exactly `target=<row label|none|status:STATUS>` where STATUS is `unknown`, `invalid`, `ambiguous` or `stale`. The only preregistered relaxation is accepting `target:` in place of `target=`; no missing prefix, explanation, extra text or alternative status spelling is relaxed.
- Arm C matrix output is exactly one line `action=<answer|refresh|query|reject_premises|insufficient_evidence|abstain>`. No row label or status parameter is permitted; `answer` means use the mechanically bound value.
- Arm C epistemic output is exactly `action=query choice=<Q1-Q4>` when querying, or `action=<abstain|insufficient_evidence>` for the restraint/insufficiency controls. The query label is part of the decision, not a report of the final mechanical answer. Other fixed-vocabulary actions remain parseable but are incorrect for these states.
- Arm D uses deterministic natural-language templates. Matrix states and non-tie epistemic states require zero neural generation. Only the two equal-cost epistemic ties present a genuinely unresolved choice; the neural layer receives the rendered tie fact plus only the tied candidates, chooses one candidate with `choice=<Qx>`, and a mechanical validator accepts the choice only if it is in the tied candidate set. Unauthorized choices are recorded as failures and are not incorporated as if selected.

Arm D is the first test of the division-of-labor architecture end to end: neural language never carries final mechanical fact.

## 5. Metrics (preregistered)

- strict and relaxed report accuracy per arm per state family
- format-failure rate: strict contract invalid but semantically correct under the narrow preregistered relaxation
- **state-respect rate**: fraction of outputs consistent with authoritative typed state regardless of the narrow format relaxation
- decision appropriateness (arm C/D): exact match against mechanically correct policy, scored per state type; tied epistemic candidates accept any authorized tied label
- restraint: correct non-query/non-refresh action on negative controls
- cost: output tokens and latency per arm; mechanical render cost and number of zero-neural-token cases
- failure taxonomy: every failure classified as:
  - `format` — semantically correct but strict contract-invalid;
  - `premise-recompute` — follows a naive recomputation from premises/surface differences while contradicting authoritative typed state;
  - `state-ignore` — format-meaningful output inconsistent with authoritative typed state;
  - `hallucinated-content` — no admissible strict/relaxed semantic action or report.

For the stale matrix probe, emitting cached `C` is explicitly tagged as stale-value emission; emitting recomputed `D` before refresh is classified as premise-recompute. Both are incorrect because the authoritative effective state is `stale`.

## 6. Decision logic (binding)

- **H1 supported** if READOUT_REPAIRED scores at least **15/18 strict** on the expanded non-normal matrix battery **and** answers the separate stale probe correctly. If the 15/18 non-normal threshold passes but stale fails, H1 is weak/inconclusive rather than supported, because the draft hypothesis claimed accurate reporting across state types.
- **H3 supported** if H1 fails and decision-consumption remains high: at least **16/19** correct matrix decisions over non-normal+stale states, at least **10/12** correct epistemic decisions, and at least **5/6** correct epistemic restraint/insufficiency controls. In that case adopt mechanical-render + decision-only consumption as the reference design.
- **H2 supported in the strong form** if H1 fails and decision-consumption also fails either the matrix non-normal+stale floor or the epistemic floor: the model does not respect typed state even as premise; the substrate thesis narrows to fully-mechanical pipelines and the consumption problem becomes the primary open risk.
- Mixed outcomes outside these bounds are reported as inconclusive rather than forced into a nearest hypothesis.
- D must be a measured result, not a declared ceiling (the EPI-04 AUTO_REFERENCE criticism): renderer fidelity, unauthorized-choice rejection, zero-token coverage, tie interpretation accuracy, latency and output tokens are all recorded.

## 7. Controls and honesty constraints

- Grading is mechanical wherever the correct policy is mechanical; any human/relaxed grading is exploratory and labeled posthoc.
- Replication transparency: arm A reuses exact EPI-05 prompts so drift is measurable.
- Negative results preserved and committed.
- Planned generation count: **125** — 37 READOUT_BASELINE + 37 READOUT_REPAIRED + 37 matrix DECISION_CONSUMPTION + 12 epistemic DECISION_CONSUMPTION + 2 equal-cost-tie interpretations in MECHANICAL_RENDER. Hard cap **130 generations or 20 GPU minutes**. This is comparable to EPI-05 (~120 generations, ≤ 3 minutes GPU expected). CPU validation of harness + substrate first, full suite green before GPU.
- Operand transcription (the m02 copy-error class) is explicitly out of scope — it is a production-side failure, not consumption. Recorded as a linked open problem.
- A substrate semantics bug found before implementation is corrected and regression-tested: dependency invalidation is now checked before returning an intrinsic `ambiguous`/`invalid`/`unknown` derived status. The corrected sequence for a tied epistemic result whose premise cost changes is **ambiguous → stale → refreshed resolved result**. This is a substrate bug fix, not a neural condition; it prevents old ties from being presented as current decisions.

## 8. Implementation notes

- Implemented in `src/jarvis_research/gate1_8_consumption.py`; runner `results/_consumption.py`; deterministic tests `tests/test_gate1_8_consumption.py` plus the core bus regression in `tests/test_cognitive_bus.py`.
- Mechanical renderer (arm D) is new substrate surface: typed state → templated natural-language rendering with zero neural involvement. It is executed and graded for fidelity rather than declared correct.
- Action vocabulary for arm C fixed in advance: `answer`, `refresh`, `query`, `reject_premises`, `insufficient_evidence`, `abstain`.
- Arm A exact prompt reuse means the same `render_hybrid_matrix` function and typed-state snapshot format as HZN-EPI-05. The expanded non-normal cases use the same defect constructors and rendering path with new base matrices.
