# HZN-OC-01 — Observation Channel Compression (Gate 1.3 spec, v3.1)

**Status: conditional exploratory approval (HZN-IR-07 document review; HZN-IR-08 implementation review — 10 preflight-integrity corrections applied).** Authorized: corrective implementation, unit tests, tokenizer profiling, observation auditing, comprehension preflight. Campaign execution pending: corrected preflight evidence, resource allocation, primary utility definition, statistical margins, acceptance criteria. Confirmatory freeze preserved; Gate 1.2 evidence record unchanged.

v3 folds in all twelve HZN-IR-07 binding dispositions; v3.1 applies the ten HZN-IR-08 corrections (fail-closed gates, strict grading, canonical field parity incl. magnitude, mandatory provenance, realistic comprehension sets, eligibility fix, ablation separation, reconciled token accounting, s3 headered candidate, preserved negative result).

## 1. Purpose

Gate 1.2 produced an exploratory ordering on arm B (A1 > A2 > A0 on selection delta and regression integrity). **Reviewer caveat, acknowledged in-protocol:** the genuine-vs-permuted mean advantage is *not consistent across seeds* — A1 beat A2 on only 1 of 3 seeds, with the aggregate gap driven by seed 1. The protective effect against blind editing (A1 > A0, 3/3 seeds) is currently more consistent than the correspondence effect itself. Additionally, blind conditions received ~2.5× more intervention opportunities, so protection via *better targeting* and protection via *fewer interventions* are not yet separated. Epoch I exists precisely to separate them.

HZN-OC-01's question: can a cheaper representation preserve correspondence and utility?

> min C(Z) s.t. E[U(S1) − U(S2)] > δ (correspondence margin), E[U(S1) − U(L1)] > −ε (non-inferiority), capability/safety constraints enforced independently.

ε, δ express what Horizon considers *meaningful*, fixed at preregistration from the sensitivity analysis (§9) — not from the observed exploratory gap.

## 2. Channel levels

| Level | Representation | Purpose |
|---|---|---|
| L | Existing pipe-schema `145\|L12\|f7\|n=147456\|m=0.0046\|ar=0.82\|lc=-0.03` | Baseline — **already a compact structural notation, not prose** (reviewer Finding 1) |
| S-I | Compact familiar schema | Remove redundancy, keep pretrained decodability |
| S-II | Fixed symbolic vocabulary `O:145\|F:07\|AR:82\|LC:-03` | Deeper compression, fixed legend — **measured: loses to S-I** (BPE-fragments; negative result preserved) |
| S-III | Headered positional schema — field names once in `cols:` header, rows positional | Removes repeated field names; **measured best: −16.1% arm A / −19.4% arm B** at full precision, complete semantics |

The reframed problem per reviewer Finding 1 and the measured profile (`gate1_3_channel_preflight.md`): this is **structural** compression — eliminating repetition of identifiers, metadata, field names — more than linguistic compression. Observation columns are ~34–42% of row tokens; structural fields dominate. Candidate encodings are adopted by **measured tokenized length** on real catalogs — never character count — including legends, catalog references and decoding instructions in the profile.

L-I (learned latent) remains deferred. Per HZN-IR-08 §5 the next structural directions are: reusable static-prefix KV caching (with mandatory post-edit invalidation — cached activations computed on stale weights are invalid) and the **addressable observation store** (retrieve relevant rows rather than serializing the whole catalog each cycle) — both retained as independent candidates, not folded into this spec's primary comparison.

## 3. Field equivalence — canonical typed record, zero silent omissions

`Decode(Encode(O)) = O` for **all** proposer-visible fields at agreed rendered precision — verified mechanically per row by projecting both source and decode onto the **canonical record** (`verify_field_parity`, implemented, tested):

```
CanonicalCandidate = { param_id, layer, family, numel,
    mag_min, mag_max,            # EFFECTIVE interval — per-row field or
                                 # uniform 'bounds:' legend, always populated
    count_max,                   # explicit c<=N or arm-wide default
    obs_available, ar, lc }      # absent columns and NA both decode to False
```

This closes HZN-IR-08 F3: magnitude bounds, count defaults and observation availability are part of the compared record whether or not they are rendered per-row. Precision reductions (numel→scale-suffix, mag-cap→2sf, numel removal) are **information-ablation experiments** — they change what the proposer knows and are excluded from the information-matched primary comparison (directive 8); each would need its own preregistration.

## 4. Conditions

| Cond | Channel | Correspondence |
|---|---|---|
| N0 | None | blind baseline |
| L1 | Linguistic | genuine |
| L2 | Linguistic | permuted |
| S1 | Symbolic | genuine |
| S2 | Symbolic | permuted |

Permutation reuses Gate 1.2 machinery verbatim (group-level, strata, tie-aware ranks, recorded, coverage at rendered precision). Same edit grammar, bounds, evaluator, arm configs.

### 4a. Data/partition reuse (binding)

Permitted shared infrastructure: evaluator, pinned checkpoint, catalog construction, observation collector, permutation algorithm, practice partition (development), selection/regression partitions as **evaluator-only instruments**. Prohibited: prior selections/outcomes/trajectories as proposer-visible info; development-exposed passages for independent confirmation; encoders tuned against evaluation outcomes for confirmatory claims. **The sealed holdout is protected from the proposer AND from research-driven optimization decisions** (directive 10).

## 5. Comprehension pretest — C1–C4 (blocking)

Items generated mechanically from synthetic/held-out catalogs with known answers; never from trajectories.

| Level | Test | Screening threshold (engineering target) |
|---|---|---|
| C1 — Decoding | recover encoded fields | **95%** |
| C2 — Comparison | relational interpretation | **90%** |
| C3 — Composition | multi-constraint satisfaction | **90%** |
| C4 — Operational grounding | observation → correct param_id | **100% on the finite preflight set** |

C4 is the operational gate: an observation the model decodes but addresses to the wrong parameter is an intervention on the wrong component. Perfect pretest ≠ guaranteed future grounding → **deterministic address validation remains mandatory during execution** (already structural: param_id bounds-checking in the patch manager). A level failing pretest reports "preserved but undecodable" and is skipped, never silently retried.

**Gate integrity (HZN-IR-08 F1/F2/§3, implemented):** the gate fails closed — every level must be present with at least its preregistered minimum probe count (`MIN_PROBES = C1:40, C2:40, C3:20, C4:60`) and missing responses score as failures. Grading is strict whole-response: an integer answer must *be* the response (not contain it); JSON lists require valid integer ids. C4 includes **ambiguity probes** — descriptions matching ≥2 rows where the only correct answer is refusal — alongside unambiguous probes, across **≥3 independently generated catalogs** with campaign-realistic structure (192–248 rows, ~55% near-zero contrasts, rounded ties, repeated family/layer pairs). 3/3 lucky guesses would only support a ~37% one-sided lower bound on accuracy; ≥60 zero-failure probes support >95%.

## 6. Sequential epochs — pristine state between

| Epoch | Controlled | Outcome |
|---|---|---|
| I — information-matched | evidence content + intervention opportunities | representational fidelity/utility (separates targeting-vs-frequency protection) |
| II — compute-matched | measured stage-seconds | operational efficiency/utility |

Epoch II runs only for the **eligible** encoding (passes canonical parity + pretest + beats the **L baseline** on measured tokens — HZN-IR-08 d7: alternatives compete against L; a more complex encoding must additionally justify its complexity over simpler successful ones). Every epoch restarts from the pristine checkpoint — no edited weights, history, selections or feedback carry forward. Encoding selection between epochs is documented explicitly; subsequent confirmatory claims use independent evaluation data.

## 7. Campaign structure (staged)

| Stage | Scope | Purpose |
|---|---|---|
| Preflight | all candidate encodings | tokenizer profile + parity |
| Comprehension | L and selected S encodings | operational literacy (C1–C4) |
| Exploratory pilot | 5 conditions × 3 seeds × **arm B** | estimate channel behavior |
| Negative-control sentinel | arm A, reduced cadence | detect unexpected arm-A sensitivity |
| Expanded campaign | sized by sensitivity analysis | repeatability |
| Epoch II | eligible encoding only | realized efficiency |

Arm B prioritized — it holds the demonstrated experimental separation; arm A's near-null field serves a defined control role rather than consuming half the compute. Three seeds is an exploratory-pilot figure, not an acceptance figure.

## 8. Budget accounting (directive 9 — implemented)

Compute-matched accounting decomposes per candidate: proposal/intervention compute; observation acquisition + rendering; final independent evaluation (**fixed resource allowance** — implemented: `stage_seconds` holds only the in-loop stages the budget bounds, `final_eval_seconds` is recorded separately, so eval-cost variance cannot distort the cognitive-efficiency comparison); budget overshoot; end-to-end wall time (`unattributed_wall_seconds` = wall − in-loop − eval). All recorded independently. The earlier claim that B-condition under-allocation "runs against those conditions" is softened: when interventions can damage, *fewer* intervention opportunities can itself be protective — the confound is named, not assumed away.

## 9. Statistical framework (directive 4)

Paired within-seed differences, not unpaired pooled means:

- Correspondence: D_i = U(S1_i) − U(S2_i), require E[D] > δ via confidence bound.
- Non-inferiority: E[U(S1) − U(L1)] > −ε via confidence bound.
- Ordering of means alone is not acceptance.

Sensitivity analysis inputs: within-seed variance + its estimation uncertainty; power at multiple plausible effect sizes; separate correspondence and non-inferiority hypotheses; multiple-comparison handling across outcomes and capability safeguards; **dependence between domain measurements from the same trajectory** — the 24 domain-instances are diagnostics, not 24 independent replicates.

**Measured sensitivity (preflight artifact `results/gate1_3_sensitivity.json`, `results/_sensitivity_analysis.py`):** provisional margins **ε = δ = 0.05** selection-delta units (meaningfulness judgments, fixed independent of the observed gap). Gate 1.2 within-seed sd of the A1−A2 diff = **0.127** (95% CI [0.066, 0.795] on df=2 — the estimate is itself highly uncertain). Monte-Carlo paired-t power at α=0.05:

| True effect | σ=0.08 | σ=0.13 | σ=0.20 |
|---|---|---|---|
| μ=0.20 (Gate-1.2-mean magnitude) | n=4 | **n=7** | n=13 |
| μ=0.12 (moderate) | n=10 | n=22 | >40 |
| μ=0.08 | >40 | >40 | >40 |
| Non-inferiority, ε=0.05 at equality | n=18 | **>40** | >40 |

**Consequence, stated honestly:** at the measured σ≈0.13, an 80%-powered correspondence acceptance at Gate-1.2-magnitude effects needs ~7 seeds; moderate effects need ~20+; and **ε=0.05 is below the noise floor for non-inferiority at any feasible seed count** (>40). Options returned to the reviewer, not chosen unilaterally: (a) widen ε/δ as meaningfulness constants; (b) reduce σ by design (paired matched-epoch diffs may be tighter than the cross-condition 0.13 estimate — the pilot exists partly to measure this); (c) accept a larger seed budget. The 3-seed exploratory pilot therefore estimates channel behavior and *refines σ* — it cannot deliver acceptance power, consistent with the staged table.

## 10. Observation audit layer — extended provenance (directive 7)

`O = (a, t, v, m, p)` with provenance extended:

```
p = { base_checkpoint_hash,
      applied_edit_chain_hash,      # binds obs to the ACTUAL modified state
      observation_context_hash,     # passages/config used for measurement
      observer_version,
      measurement_config_hash }
```

An observation correct about a previous model version can mislead about the current one — the edit-chain hash makes stale-vs-false distinguishable. Established by observer infrastructure, never asserted by the proposer. Per HZN-IR-08 F4 all provenance fields are **mandatory** (construction fails on empties); per-edit hashes commit to the *applied* edit (target + operation + value), not merely the proposal text; prefixed-stat normalization is corrected and tested.

## 11. Failure-interpretation map (preregistered)

| Result | Interpretation |
|---|---|
| S1 > S2, S1 ≈ L1 | compression preserves correspondence — advance |
| S1 > S2, S1 ≪ L1 | correspondence survives, utility degraded |
| S1 ≈ S2, comprehension fails | decoding problem; intervention inconclusive |
| S1 ≈ S2, comprehension passes | no correspondence advantage; grounding/power |
| S1 > L1 AND S2 > L2 | compression helps independent of correspondence |
| S1 > L1, S2 unchanged | more effective use of genuine obs — particularly interesting |
| S1 ≈ L1, lower compute | efficiency success under capability constraints |
| N0 outperforms all evidence conditions | observation overhead/misdirection outweighs benefit |

## 12. Retained constraints + deferred diagnostics

- Fail-closed containment; refusal → `infrastructure_error`, excluded from model scores.
- Budget checked before every step; all outcome classes count; overshoot recorded.
- Control abstention parity; regression + capability + transition ledger per candidate; single-task family n reported.
- Non-goals unchanged: no Gate 1.2 modifications, no new edit surfaces, no tokenizer changes/training, no recursion, sealed holdout untouched, no proposer access to evaluation internals.
- **HZN-OC-02** (Deterministic Evidence Operator, deferred): neural proposer vs fixed-policy symbolic operator on identical evidence/budgets — tests how much proposer intelligence the channel actually needs; introduces **amortized inference cost** as a metric (validated relationships as executable primitives that eliminate repeated inference — the executable-grimoire probe).
