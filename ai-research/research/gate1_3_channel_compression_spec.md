# HZN-OC-01 — Observation Channel Compression (Gate 1.3 draft spec)

**Status: DRAFT for reviewer comment — not authorized, not implemented.** Exploratory research track running in parallel with the frozen confirmatory protocol. Does not modify Gate 1.2 implementation, data partitions, or the confirmatory freeze.

Provenance: proposed in the reviewer's Gate 1.2 research assessment ("symbolic channel investigation: recommended"); this document operationalizes it for review before any code.

## 1. Purpose

Gate 1.2 established exploratory evidence that (a) the proposer reads rendered internal-state observations, (b) genuine correspondence between observation and state is protective on the arm where blind editing damages the checkpoint, and (c) rendering that evidence in natural language consumes ~78–79% of the trajectory's proposal-stage compute.

HZN-OC-01 asks whether a cheaper representation preserves those properties:

> min C(Z) s.t. U(Z) ≥ U(X) − ε, U(Z_true) − U(Z_shuffled) > δ, with capability/safety constraints enforced independently.

## 2. Channel levels

| Level | Representation | Example | Purpose |
|---|---|---|---|
| L | Linguistic (existing) | prose catalog rows | Baseline channel (Gate 1.2) |
| S-I | Compact semantically familiar schema | `L14.DP lc=-0.03 ar=0.82` | Remove redundant prose, keep pretrained decodability |
| S-II | Fixed symbolic vocabulary | `O:042\|L:17\|F:MLP\|LC:-03\|AR:82` | Deeper compression; learned association via fixed legend |
| S-III | Compositional symbolic language | (deferred design) | Reusable primitives; program-induction direction |

L-I (learned latent) is **deferred** — it requires training and interchange-intervention controls (Geiger et al. 2021/2022) rather than group permutation; out of scope for this campaign.

S-III is a design placeholder, not a committed level — its inclusion depends on S-II outcomes.

**Tokenizer caution (binding):** candidate encodings are selected by *measured tokenized length*, not character count. Punctuation-heavy schemas can tokenize worse than prose under BPE; every candidate encoding's serialized token count is measured and reported before adoption.

## 3. Conditions

| Cond | Channel | Correspondence |
|---|---|---|
| N0 | None | — (blind baseline) |
| L1 | Linguistic | Genuine |
| L2 | Linguistic | Permuted |
| S1 | Symbolic (level under test) | Genuine |
| S2 | Symbolic (level under test) | Permuted |

Equivalence requirements (hard constraints):

- S and L conditions present **the same observation fields at the same precision** — no field may exist in one channel and not the other. If the schema can't express a field, the field leaves both.
- Permutation reuses the Gate 1.2 machinery: group-level permutation within (family, layer-band) strata, tie-aware midranks, permutation recorded per candidate, coverage measured at *rendered* precision of the channel under test.
- Same edit grammar, same bounds, same evaluator, same partitions family (fresh v5 disjoint partition if reusing v4 would constitute test-set reuse — flagged for reviewer decision).
- One symbolic level per campaign epoch; levels are compared across epochs, never mixed within a condition.

## 4. Comprehension pretest (blocking)

Before any candidate trajectory runs at a symbolic level, the proposer must pass a **decoding pretest**: rendered obs rows are withheld from the edit prompt and queried directly — "which family is O:042? which has higher contrast, O:042 or O:117? which layer-band?".

- Pretest items are generated mechanically from the encoded catalog (no human authorship).
- Threshold set at preregistration (proposal: ≥80% field-identification accuracy).
- **Failure interpretation:** a level that fails the pretest cannot produce a meaningful S1-vs-S2 result — that outcome reports "information preserved but undecodable," not "information lost." Levels that fail are reported and skipped, never silently retried with the same encoding.

This pretest is what lets S1 ≯ S2 be interpreted. Without it, the experiment cannot distinguish representation failure from comprehension failure.

## 5. Two comparisons, kept separate

Per the reviewer's directive, information-matching and compute-matching are **distinct comparisons, never combined**:

- **Epoch I — information-matched:** equal observation content, equal intervention allocation (fixed steps). Measures whether compression *preserves usefulness* — does S1 retain the L1 > L2 ordering at the same number of edit opportunities?
- **Epoch II — compute-matched:** equal measured stage-seconds (`--budget-mode seconds`), intervention counts become an outcome. Measures whether compression buys *more productive reasoning per second* — does S1 exploit the freed budget?

Running Epoch II without Epoch I would leave ambiguous whether any gain is representational or merely budgetary.

## 6. Primary acceptance criteria (proposed — reviewer to ratify)

- [ ] Symbolic channel preserves all observation fields at rendered precision (verified mechanically per row).
- [ ] S1 > S2 on preregistered outcomes (selection delta, regression ledger), directionally consistent with L1 > L2.
- [ ] S1 non-inferior to L1 within preregistered margin (proposal: selection-delta difference < 0.02 ppl, regression-failure count equal-or-better).
- [ ] Measured token consumption, latency and total compute decrease vs L1 (report all three; token reduction alone is insufficient evidence).
- [ ] N0 remains the controlled blind baseline.
- [ ] Capability regressions within existing tolerances; infrastructure failures excluded and tagged.
- [ ] Proposer cannot modify evaluator, acceptance criteria or authorization boundaries — structural property, restated.

## 7. Addressable-store variant (optional arm — distinct hypothesis)

If resources allow, a sixth condition: **A-store** — compact index in context + deterministic read-only `OBS.*` interface (READ/RANK/COMPARE/FRESHNESS/HISTORY). Tests whether in-context completeness is needed at all, versus reliable access. Extra metrics: reads-per-step, retrieval-selection bias, per-step call overhead. This arm has different compute semantics (interface calls are proposal-stage cost) and is reported separately, not pooled with S-levels.

## 8. Observations carry formal identity

Every observation record carries `O = (a, t, v, m, p)` — address, epoch, value, method/context, provenance+checkpoint hash. The channel under test may render a compact subset, but the identity tuple is what the audit layer stores, making *false* vs *stale* distinguishable and binding every claim to a model state. Proposer never asserts its own evidence identity.

## 9. Failure interpretation map (preregistered)

| Outcome | Reads as |
|---|---|
| S1 > S2, S1 ≈ L1 | Compression preserves correspondence — advance |
| S1 ≈ S2, pretest passed | Channel conveys no usable info at this level — representation failure |
| S1 ≈ S2, pretest failed | Undecodable encoding — comprehension failure; iterate encoding |
| S1 > S2 but ≪ L1 | Partial preservation; inspect lost fields |
| All ≈ null | Channel cost was not the binding constraint — report and stop |

## 10. Explicit non-goals

- No modification to Gate 1.2 frozen artifacts or the confirmatory protocol.
- No new edit surfaces, no observation mechanisms beyond the existing catalog stats.
- No tokenizer changes, no fine-tuning, no learned channels in this campaign.
- No recursive generations, no proposer access to evaluation internals.
- Sealed holdout remains untouched; v4/v5 partitions only.

## 11. Open questions for the reviewer

1. Epoch ordering — is information-matched-first acceptable, or should compute-matched run in parallel on a second GPU window?
2. Partition reuse — is v4 reuse across L and S epochs acceptable, or is a fresh v5 required to keep conditions uncontaminated?
3. Non-inferiority margin — is Δselection_ppl < 0.02 the right ε, and should δ (correspondence margin) be set from the arm-B A1−A2 gap observed in Gate 1.2?
4. Is the addressable-store arm in scope for the first epoch or deferred?
5. Pool/seeds — 3 seeds × 5 conditions × 2 arms mirroring Gate 1.2, or reduced scope given reviewer's "reduce campaign scope in advance" guidance?
