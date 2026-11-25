# HZN-OC-01 — Observation Channel Compression (Gate 1.3 spec, v3)

**Status: conditional exploratory approval (HZN-IR-07, independent document review complete).** Authorized: implementation, unit tests, tokenizer profiling, observation auditing, comprehension preflight. Campaign execution pending: complete preflight evidence, finalized resource allocation, preregistered statistical acceptance criteria. Confirmatory freeze preserved; Gate 1.2 evidence record unchanged.

v3 folds in all twelve HZN-IR-07 binding dispositions, including the reviewer's seed-level statistical caveat and its two identified confounds.

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
| S-II | Fixed symbolic vocabulary `O:145\|F:07\|AR:82\|LC:-03` | Deeper compression, fixed legend |

The reframed problem per reviewer Finding 1 and the measured profile (`gate1_3_channel_preflight.md`): this is **structural** compression — eliminating repetition of identifiers, metadata, field names — more than linguistic compression. Observation columns are ~34–42% of row tokens; structural fields dominate. Candidate encodings are adopted by **measured tokenized length** on real catalogs — never character count — including legends, catalog references and decoding instructions in the profile.

L-I (learned latent) and S-III (compositional language) remain deferred.

## 3. Field equivalence — mechanical, zero silent omissions

`Decode(Encode(O)) = O` for **all** proposer-visible fields at agreed rendered precision, verified mechanically per row (`verify_field_parity` — implemented, tested). A field that cannot be expressed in S leaves **both** channels; any preregistered precision reduction (numel→scale-suffix, mag-cap→floor-rounded 2sf) applies symmetrically and is listed in the preregistration.

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

## 6. Sequential epochs — pristine state between

| Epoch | Controlled | Outcome |
|---|---|---|
| I — information-matched | evidence content + intervention opportunities | representational fidelity/utility (separates targeting-vs-frequency protection) |
| II — compute-matched | measured stage-seconds | operational efficiency/utility |

Epoch II runs only for the **eligible** encoding (passes parity + pretest + beats S-I on tokens). Every epoch restarts from the pristine checkpoint — no edited weights, history, selections or feedback carry forward. Encoding selection between epochs is documented explicitly; subsequent confirmatory claims use independent evaluation data.

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

## 8. Budget accounting (directive 9)

Compute-matched accounting decomposes per candidate: proposal/intervention compute; observation acquisition + rendering; final independent evaluation (**fixed resource allowance**, so eval-cost variance cannot distort the cognitive-efficiency comparison); budget overshoot; end-to-end wall time. All five recorded independently. The earlier claim that B-condition under-allocation "runs against those conditions" is softened: when interventions can damage, *fewer* intervention opportunities can itself be protective — the confound is named, not assumed away.

## 9. Statistical framework (directive 4)

Paired within-seed differences, not unpaired pooled means:

- Correspondence: D_i = U(S1_i) − U(S2_i), require E[D] > δ via confidence bound.
- Non-inferiority: E[U(S1) − U(L1)] > −ε via confidence bound.
- Ordering of means alone is not acceptance.

Sensitivity analysis inputs: within-seed variance + its estimation uncertainty; power at multiple plausible effect sizes; separate correspondence and non-inferiority hypotheses; multiple-comparison handling across outcomes and capability safeguards; **dependence between domain measurements from the same trajectory** — the 24 domain-instances are diagnostics, not 24 independent replicates.

## 10. Observation audit layer — extended provenance (directive 7)

`O = (a, t, v, m, p)` with provenance extended:

```
p = { base_checkpoint_hash,
      applied_edit_chain_hash,      # binds obs to the ACTUAL modified state
      observation_context_hash,     # passages/config used for measurement
      observer_version,
      measurement_config_hash }
```

An observation correct about a previous model version can mislead about the current one — the edit-chain hash makes stale-vs-false distinguishable. Established by observer infrastructure, never asserted by the proposer.

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
