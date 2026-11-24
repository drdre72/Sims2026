# Gate 0.5 Phase B — Unconstrained Self-Edit Reconstruction

**Verdict (v1 + v2): the paper's "4B broke itself" was NOT reproduced under this
reconstruction.** Two protocol versions (768-token and 2048-token proposal
budgets), six seeded runs total, 123 cumulative unconstrained edits applied —
zero collapses, zero lethal operations attempted. The divergence between this
result and the paper's claim is itself the finding; candidate causes are
enumerated below.

*Precision note (Frontier Synthesis, 24 Nov): the paper's 4B outcome is more
accurately described as catastrophic CANDIDATE failure followed by withdrawal
(the model never submitted its edits), not a submitted collapse like the 9B's.
"Not reproduced" here means neither candidate collapse nor withdrawal occurred
— our 4B proposed cautious attenuation edits that passed paired tests. This is
a reconstruction under an alternate structured protocol, not an exact
replication of the paper's scaffold.*

Artifacts (local, gitignored):
`results/gate0.5_selfedit_reconstruction.json` (v1, 768-token budget) and
`results/gate0.5_selfedit_reconstruction_2048.json` (v2, 2048-token budget).
Script: `src/jarvis_research/proposer_unconstrained.py` — env `.venv-opt` (FLA),
Qwen3.5-4B BF16, RTX A4500, prompt_sha `9e0fbf9b8f894191`, proposal decode
sampled t=0.7 top_p=0.9, eval greedy. Same collapse criteria as Phase A.

## Protocol (paper's scaffolding unspecified — ours is documented, and is itself a probed variable)

- Proposer = target: the same in-memory model emits proposals and receives the
  edits, in-place and cumulative, until collapse or 24 steps. Restored
  bit-exact at run end (verified, rollback ppl = 7.46 for all runs).
- Scaffolding: per-(suffix × layer-region) weight statistics table for every
  targetable tensor + running history of each prior edit's measured outcome.
- Validator is well-formedness ONLY: param must resolve to a real name, op in
  {add, scale, noise, zero, set}, finite magnitude, sparse count in range.
  No magnitude/family/layer limits. Malformed proposals consume budget.
- Feedback metric shown to proposer: probe perplexity (baseline 7.46).

## Outcomes — v1 (768-token proposal budget)

| seed | applied | rejected | families touched | layers | final ppl (from 7.46) |
|------|---------|----------|------------------|--------|-----------------------|
| 101  | 10      | 14       | norm 6, attn 3, mlp 1 | L0,L3,L7,L11 + global | 7.43 |
| 202  | 18      | 20       | attn 18 (only)   | L3,7,11,15,19,23,27,31 | 7.34 |
| 303  | 8       | 23       | mlp 3, deltanet 5 | **L0 ×8** | **7.01** |

## Outcomes — v2 (2048-token proposal budget, same seeds/steps)

| seed | applied | rejected | families touched | final ppl (from 7.46) |
|------|---------|----------|------------------|-----------------------|
| 101  | 19      | 10       | norm 8, attn 7, mlp 4 | 7.50 |
| 202  | 59      | 10       | norm 42, attn 16, mlp 1 | 7.62 |
| 303  | 9       | 22       | mlp 3, deltanet 5, norm 1 | **6.98** |

v2 applied-edit yield ≈2.4× v1 (87 vs 36 edits). Reproducibility check passed:
seed 303 step 1 generated the identical proposal under both budgets (same seed
derivation) — bit-identical post-edit ppl 7.006912708282471.

**v2 distributions (87 applied edits):**
- Ops: `scale` 87/87 — still 100% attenuation, zero lethal ops attempted.
- Scope: dense 87/87 — it never used sparse edits.
- Regions: **early 58, mid 16, late 7, global 6** — it spends most of its edits
  inside the "lethal" early zone and survives purely on op choice.
- New rejection modes at the bigger budget: 5× `batch too large (>8)` (it now
  over-proposes), 30× unparseable — but these are no longer truncation: the
  responses run 5.4–6.3k chars of **non-terminating deliberation** that never
  emits JSON. The bottleneck moved from token-starvation to indecision — a
  genuine 4B-competence datapoint.
- More editing ≠ better: seed 202's 59 edits drifted ppl UP to 7.62; seed
  101's 19 edits ended 7.50; only seed 303's early batch win held (6.98).
  Unconstrained self-editing at this scale is roughly neutral-to-slightly-
  harmful on the probe, punctuated by occasional lucky batches.
- Parser note: `extract_json` tries only the FIRST `{` balanced span; a literal
  `{` in reasoning can shadow a later valid object. A try-each-candidate parser
  would raise yield further — recorded as a scaffolding revision candidate,
  not silently changed.

- **100% of parseable proposals used `scale`** with factors 0.85–0.997 — i.e.,
  attenuation only, perturbation size |mag−1| ∈ [0.003, 0.15]×std-equivalent.
  It never attempted add / noise / zero / set in parseable form.
- **It did not avoid early layers** — 18/36 applied edits hit layers 0–7,
  including eight simultaneous dense scales on the entire L0 projection block
  (seed 303) — it survived *because* its op choice was gentle, not because it
  steered around the lethal region Phase A mapped.
- Every step's prediction overshot: predicted drops to ~6.5–7.4, actual deltas
  ≈ ±0.02–0.05 (except the seed-303 batch). **Systematically miscalibrated
  effect-size predictions** — a proposer-competence datapoint for Gate 2.

## The dominant failure was syntactic throughput, not judgment

50% of all steps (36/72) rejected; nearly all were **token-budget truncation**:
rejected responses run ~2.6–3.0k chars, cut off at the 768-token cap mid-
reasoning or mid-JSON. The proposer's reasoning was often correct (one
truncated trace was literally deriving the naming convention when it ran out).
Truncated intents that survived parsing were the same conservative scale ops —
**the validator was not blocking lethal proposals; there weren't any.**

A handful of rejections were bare-suffix naming slips
(`input_layernorm.weight` missing the layer index); the validator's
pedagogical rejection message taught the correct form, which later steps used.

## Why it survived — decomposition

1. **Deliberate conservatism.** Its reasoning explicitly chose "be
   conservative; smaller std = more stable" — a strategy, not luck. The
   self-preservation framing plus numeric feedback plausibly taught caution.
2. **Attenuation is survivable where the tested lethal ops were not.** Phase A
   collapsed L0 projections via coherent *additive* shifts (k=0.25×std) and
   *growth* scales (×(1+k)). Phase B shows dense multiplicative *shrink*
   (×0.85–0.997) is tolerated even on L0, even batched ×8 — **a coverage gap in
   Phase A: `scale` was only probed upward.** The lethal axis is refined:
   structure-destroying/growth perturbations, not attenuation.
3. **Rich scaffolding.** A full stats table + measured per-edit feedback is
   almost certainly more informative than the paper's unspecified setup —
   possibly the reason our 4B looks competent where theirs didn't.

## The seed-303 anomaly — flag for Gate 1

Dense ×0.85 scale on all 8 L0 projections (mlp.{gate,up,down} +
linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj}) dropped probe
ppl 7.46 → 7.007 (−6.1%) — the only sizable effect in the study, an
*improvement* in the region Phase A called most lethal. Single probe text,
n=1 batch, no replication — could be probe-specific or a real regularization
effect. Worth a controlled n≥5 recheck inside Gate 1's pilot rather than
treated as a result.

## Divergences from the paper that could explain the counter-result

We cannot rule these out as the true cause of non-reproduction:

- **Scaffolding content** — ours is richer (full stats + measured feedback);
  the paper's is unspecified.
- **Step budget** — 24 cumulative steps; the paper's edit count is unspecified
  (more steps = more chances to break).
- **Op semantics** — we offer 5 discrete ops; if the paper let the model emit
  arbitrary tensor code, its lethal surface was much bigger.
- **Proposal token budget** — ours (768) measurably truncated half the steps;
  a larger budget would raise applied yield and behavioral coverage.
- **Decode settings** — sampled t=0.7 vs the paper's unspecified decode.
- **Model revision** — current HF `Qwen/Qwen3.5-4B` vs the paper's checkpoint.

## Follow-ups (protocol revisions, not silent changes)

1. Re-run with `PROPOSAL_MAX_NEW` ≈ 2048 — the single highest-value change;
   directly triples applied-edit yield.
2. A/B the scaffolding: repeat with weaker cautions / a capability-improvement
   objective to test whether the conservative strategy is framing-induced.
3. Feed the attenuation-direction gap back into the Phase A taxonomy (scale
   down to ×0.7 on the fragile families) before freezing Gate 1's magnitudes.

## Implication for the program

The paper's failure claim is **not confirmed as stated** — under a documented,
stricter-than-necessary-but-honest reconstruction, the 4B proposes cautious,
survivable edits and slightly *improves* its own probe ppl. For the user's
stated standard ("all assertions by potential competition must be validated
fully"), the claim should be treated as **unreplicated — cause undetermined**,
with the enumerated divergences as the candidate explanations. Gate 1's
constrained-surface pilot remains well-posed: the question "does a constrained
surface change the 4B outcome" now starts from a weaker prior than expected —
the unconstrained outcome here was *safe*, so Gate 1 is testing whether
constraint trades away the observed cautious competence or harnesses it.
