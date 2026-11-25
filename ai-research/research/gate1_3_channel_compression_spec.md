# HZN-OC-01 — Observation Channel Compression (Gate 1.3 spec, v2)

**Status: CONDITIONAL APPROVAL for exploratory implementation (reviewer directive, HZN-IR-05 series).** v2 folds in reviewer rulings on the five open questions. Confirmatory execution remains frozen; Gate 1.2 evidence record unchanged.

Authorized components: **symbolic encoder, tokenizer profiling, observation audit layer, comprehension harness.** Campaign execution NOT authorized until the complete protocol and preflight evidence pass review.

## 1. Purpose

Gate 1.2 established exploratory evidence that (a) the proposer reads rendered internal-state observations, (b) genuine correspondence between observation and state is protective where blind editing damages the checkpoint, and (c) rendering that evidence bears a first-order share of proposal-stage compute.

HZN-OC-01 asks whether a cheaper representation preserves those properties:

> min C(Z) s.t. U(Z) ≥ U(X) − ε, U(Z_true) − U(Z_shuffled) > δ, with capability/safety constraints enforced independently.

ε = maximum practically acceptable utility loss when replacing language with symbolic representation. δ = minimum practically meaningful genuine-vs-permuted advantage. Both express what Horizon considers *meaningful*, decided at preregistration — the Gate 1.2 arm-B gap is a planning input for effect-size estimates, **not** the acceptance threshold (reviewer R2). A sensitivity analysis across plausible effect sizes (§11) must precede any fixed sample size.

## 2. Channel levels

| Level | Representation | Example | Purpose |
|---|---|---|---|
| L | Linguistic pipe-schema (existing) | `145\|L12\|f7\|n=147456\|m=0.0046\|ar=0.82\|lc=-0.03` | Baseline channel — already a compact hybrid schema |
| S-I | Compact semantically familiar schema | `L12.DP lc=-0.03 ar=0.82` | Remove redundant structure, keep pretrained decodability |
| S-II | Fixed symbolic vocabulary | `O:145\|F:07\|LC:-03\|AR:82` | Deeper compression; fixed legend, testable association |
| S-III | Compositional symbolic language | (deferred design) | Reusable primitives; program-induction direction |
| L-I | Learned latent | — | DEFERRED (training + interchange-intervention controls, not group permutation) |

**Tokenizer caution (binding):** candidate encodings are selected by *measured tokenized length under the campaign tokenizer*, never character count. Every candidate encoding's serialized token count is measured on the real annotated catalogs and reported in preflight before adoption.

## 3. Conditions

| Cond | Channel | Correspondence |
|---|---|---|
| N0 | None | — (blind baseline) |
| L1 | Linguistic | Genuine |
| L2 | Linguistic | Permuted |
| S1 | Symbolic (level under test) | Genuine |
| S2 | Symbolic (level under test) | Permuted |

Equivalence requirements (hard constraints):

- S and L present **identical observation fields at identical rendered precision**; if the schema can't express a field, the field leaves both.
- Permutation reuses the Gate 1.2 machinery verbatim: group-level within (family, layer-band) strata, tie-aware midranks, permutation recorded per candidate, coverage measured at the channel under test's rendered precision.
- Same edit grammar, bounds, evaluator, arm configs, seeds policy.
- One symbolic level per campaign epoch; levels compared across epochs, never mixed.
- All conditions begin from the same pristine checkpoint with independent histories, trajectories and feedback.

### 3a. Data/partition reuse policy (reviewer R1 — binding definitions)

| Reused component | Decision |
|---|---|
| Evaluator implementation, pinned checkpoint, catalog construction, observation collector, group-permutation algorithm | **Permitted** — shared infrastructure |
| Practice partition (v4 `practice.json`) | Permitted for controlled development; the model never sees it as text |
| Selection/regression partitions (v4) | **Permitted as evaluator inputs only** — they are never proposer-visible. Reusing them as measurement instruments is infrastructure, not acquired knowledge. |
| Prior candidate selections, edit outcomes, trajectories | **Prohibited** as proposer-visible information |
| Passages used during encoder development/tuning | Prohibited for independent confirmation if exposed to the proposer |
| Encoder tuned against evaluation outcomes | Exploratory only — cannot feed independent confirmation |

Shared infrastructure improves comparability; shared acquired knowledge compromises independence. The precise boundary: the proposer may never receive information produced by a previous experimental trajectory.

## 4. Comprehension pretest — four levels (blocking, reviewer §3)

S and L conditions present identical fields; decoding them correctly does not establish operational comprehension. The pretest generates items **mechanically from synthetic or held-out catalogs with independently known answers** — never from real campaign trajectories, never exposing intervention outcomes.

| Level | Test | Example |
|---|---|---|
| C1 — Decoding | Recover encoded fields | Which layer/family/signed contrast does id X carry? |
| C2 — Comparison | Relational interpretation | Which of A, B has greater absolute contrast? |
| C3 — Composition | Multi-constraint satisfaction | Which ids satisfy family=X AND layer≥Y AND lc>0? |
| C4 — Grounding | Observation → candidate identifier | Given a described observation, return the param_id it refers to |

- Preregistration threshold (proposal): ≥80% per level on held-out catalogs; C4 is the operational gate (an encoding the model can decode but cannot ground into candidate selection fails the intervention-relevant test).
- A level failing pretest reports "preserved but undecodable," not "information lost" — and is skipped, never silently retried with the same encoding.
- The pretest measures literacy, not predictive sufficiency; passing does not assert observations predict beneficial edits.

## 5. Two comparisons, kept separate

- **Epoch I — information-matched:** identical observation content, identical intervention allocation. Does compression preserve usefulness (S1 retains L1 > L2 ordering at equal edit opportunities)?
- **Epoch II — compute-matched:** identical measured stage-seconds (`--budget-mode seconds`), intervention counts an outcome. Does compression buy productive reasoning per second?

Epoch II without Epoch I leaves any gain ambiguous between representational and budgetary causes. Run order: I, then II, on separate recorded runs.

## 6. Failure-interpretation map (preregistered, reviewer §4)

| Result | Interpretation |
|---|---|
| S1 > S2, S1 ≈ L1 | Symbolic channel preserves useful correspondence — advance |
| S1 > S2, S1 ≪ L1 | Correspondence survives, utility degraded |
| S1 ≈ S2, comprehension fails | Representation/decoding problem; intervention inconclusive |
| S1 ≈ S2, comprehension passes | No correspondence advantage; investigate grounding/power |
| S1 > L1 AND S2 > L2 | Compression may help independent of correspondence |
| S1 > L1, S2 does not improve | Particularly interesting: more effective use of genuine obs |
| S1 ≈ L1 with lower compute | Efficiency success, subject to capability constraints |
| N0 outperforms all evidence conditions | Observation overhead/misdirection/policy may outweigh benefit |

Genuine-vs-permuted is indispensable but not the sole success criterion.

## 7. Addressable-store variant (retained as independent candidate)

Optional sixth condition **A-store**: compact index in context + read-only `OBS.*` interface (READ / RANK / COMPARE / FRESHNESS / HISTORY). Distinct hypothesis — in-context completeness vs reliable access. Extra metrics: reads/step, retrieval bias, per-step call overhead. Interface calls are proposal-stage cost; reported separately, never pooled with S-levels. Deferred unless Epoch I scope allows without reducing the five core conditions.

## 8. Observation audit layer — formal identity (implemented component)

Every observation record carries `O = (a, t, v, m, p)` — component address, observation epoch, measured value, method/context, provenance + checkpoint hash. The channel under test renders a compact subset; the audit layer stores the full tuple, making *false* vs *stale* distinguishable and binding every claim to a model state. Identity is established by observer infrastructure, never asserted by the proposer.

## 9. Failure-integrity requirements (carried from Gate 1.2)

- Fail-closed containment on generated-code evaluation; containment refusal → `infrastructure_error`, excluded from model-failure scores.
- Budget check before every step; rejected/abstained steps count; overshoot recorded.
- Conventional controls may abstain (unchanged-checkpoint parity).
- Regression + capability + transition ledger per candidate; single-task families reported with n.

## 10. Explicit non-goals

- No modification to Gate 1.2 frozen artifacts or the confirmatory protocol.
- No new edit surfaces or observation mechanisms.
- No tokenizer changes, fine-tuning, or learned channels.
- No recursive generations; no proposer access to evaluation internals.
- Sealed holdout untouched.
- HZN-OC-02 (Deterministic Evidence Operator) is **recorded as a deferred diagnostic** — neural proposer vs fixed-policy symbolic operator on identical evidence/budgets, the first executable-grimoire probe. It is NOT in this campaign's scope.

## 11. Open questions remaining for the reviewer

1. Epoch ordering/parallelism — sequential (I then II) assumed; parallel needs independent partitions.
2. Campaign scope — 5 conditions × 3 seeds × 2 arms mirrors Gate 1.2 (~2.5 h); the reviewer's "reduce scope in advance" option is noted but unselected.
3. Sensitivity analysis acceptance — propose computing required-seeds table over plausible effect sizes (d = 0.2 … 1.0 pooled σ, observed arm-B σ ≈ 0.08) at preregistration; ε/δ fixed after review of that table.
