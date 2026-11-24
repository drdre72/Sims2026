# Gate 1 v2: exploratory protocol and confirmatory blockers

Status: development only. Do not interpret the engineering pilot as C1 evidence.
Gate 0/0.5 results and the seed-303 negative result are unchanged.

## Changes after independent review

- Development partitions are sampled without replacement, checked for
  normalized-content overlap at generation and at load. Existing v1 partition
  files and pilot results are preserved; v2 uses a new data/result namespace.
- Batches are checked prospectively for cumulative attenuation, and sparse
  proposals reject duplicate tensors and more than 512 changed scalars total.
  Failed rollback raises an error rather than permitting another candidate.
- Proposal, validation, application, quality check, practice, selection,
  regression, and rollback stage times are logged incrementally. These are
  wall seconds covering GPU calls, not GPU utilization seconds. Reconcile
  stage sums with candidate wall time (unattributed wall time is reported).
- Qwen generates from the actual edited model, with only practice feedback in
  its prompt. Raw output, JSON validity, rejection, prediction and observed
  practice loss are retained. An invalid proposal consumes a step.
- Fixed heuristics are labeled fixed. A separate adaptive control tries two
  random edits against practice feedback and restores each trial exactly.

The prompt and parser provide *prompt isolation*, not a sandbox. The same
developer account can read all development passages. No final holdout has been
created or accessed. A final set must be prepared by an independent evaluator
under separate OS permissions and opened only once after candidate selection
and the full protocol are frozen.

## Honest pilot limitations

The selection/regression sets consist of short development passages. They do
not independently check task accuracy, reasoning, output validity, or long
context. The random control draws within the legal edit surface; it is not
matched to each model proposal by family, tensor element count, perturbation
norm, or actual computational cost. The adaptive search spends extra practice
evaluations and reports this cost. Thus a pilot may test mechanics and
estimate proposer behavior, but cannot support a causal superiority claim.

The pilot's per-domain +0.05 nats threshold is a *temporary engineering
diagnostic*, not a preregistered acceptance threshold. The suggested -0.02
nats/token minimum useful effect and +0.01 nats/domain tolerance remain
provisional. Do not infer them from tiny changes in perplexity; estimate
uncertainty directly on token losses across both independent proposer
trajectories and newly sampled passages. In particular, the old 3-seed
heuristic variance cannot support a 30-seed power claim. Two edit arms times
three conditions times 30 seeds would be 180 trajectories / 2880 steps at
16 steps, before adding the adaptive condition; proposer inference time must
be measured independently. Define the minimum worthwhile effect, baseline
to beat, one-sided comparisons (if justified), family-wise error adjustment,
sample size, fixed stopping rule and compute cap only after the real-proposer
pilot. No generational G1->G2 machinery or confirmatory study is authorized.

## Still required before confirmation

1. Independent capability and long-context evaluator, not just teacher-forced
   short-text loss. Validate grading integrity and task-family coverage.
2. Truly matched random and adaptive comparisons with the same proposal and
   evaluation budget, or explicit cost-normalized analysis; prespecify a
   strongest-baseline comparison.
3. Pinned model revision, full local shard hashes, prompt/schema/decoder
   versions, registered tolerances, power and independent-trial design.
4. Separately administered final dataset inaccessible to the proposer and
   development account. Independent evaluator approval and one-time opening.

## Completed exploratory v2 pilot (not a promotion test)

The no-thinking proposer (`enable_thinking=False`, sampled at temperature
0.7/top-p 0.9, 512 new tokens) ran on cached revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with two complete
safetensors shard SHA-256 hashes in the local result artifact. Three seeds
101/202/303, both arms, four steps, four conditions = 24 trajectories and
96 step slots. Source: `results/gate1_pilot_v2_exploratory.json` (ignored).
The earlier default-thinking 768-token exploratory attempt was aborted
after seed 101 produced four unparseable proposals (~224 measured seconds);
it is not included in this comparison.

| Surface | Condition | Selection delta CE (three seeds, nats/token) |
|---|---|---|
| A: sparse additive | self-proposer | +0.00020, -0.00048, +0.00107 |
| A | within-arm random | +0.00040, -0.00055, +0.00086 |
| A | fixed heuristic | +0.00056, -0.00002, -0.00006 |
| A | adaptive practice search | -0.00091, -0.00019, -0.00018 |
| B: attenuation | self-proposer | +0.01455, +0.02140, +0.00150 |
| B | within-arm random | +0.02518, -0.00709, +0.00587 |
| B | fixed heuristic | +0.00590, +0.00590, +0.00590 |
| B | adaptive practice search | +0.00287, -0.00822, +0.00351 |

The model emitted parseable JSON in all 24 steps, but 7/24 were rejected
by the Arm A validator (17 applied; Arm A 5/12, Arm B 12/12).
Reasons include excessive count for a 32-element recurrence tensor,
relative delta beyond the bound, and cumulative drift. The current prompt
does not ensure valid editing, and validity is part of proposer competence.
All 24 trajectories passed the *provisional* +0.05-nat domain screen,
which does not prove task retention.

The pilot took 721.4 seconds wall time. Candidate wall times summed to
693.3 seconds; remaining time covers model loading, baseline evaluation,
bookkeeping, and output. Within each candidate, attributed stage times
differed from wall by at most 0.13 seconds. Proposal generation consumed
~30.6 seconds/Arm A trajectory and ~24.8 seconds/Arm B trajectory on
average (4 steps). These measured times replace the inflated v1 estimates,
but do not predict confirmatory runtime at 16 steps/trajectory.

None of these tiny scores establish C1. In this pilot Qwen did not show
consistent positive generalization beyond the strongest practice-adaptive
baseline. The comparator is not strictly matched in perturbation or
compute; the small, examined selection corpus cannot establish a
population effect. An untouched final set and independent approval remain
explicit blockers.

## Gate 1.1 controlled-comparison pilot (exploratory, 2026-11-24)

Reviewer-directed revision addressing three findings from commit `29a63b6`:
mismatched search spaces, missing interface metadata, and a hard-coded
quality-guard baseline. Still exploratory — NOT confirmatory evidence.

### Harness changes

- **Unified legal catalog.** `build_catalog()` now emits indexed descriptors
  (param_id, layer, family, shape, numel, std, legal magnitude/count range)
  that the proposer prompt AND every control (random, fixed-heuristic,
  adaptive) draw from identically. Proposals reference tensors by integer
  `param_id`; controls may still use exact names. Arm A catalog: 192 entries;
  Arm B: 248 entries. (An earlier draft of this note claimed 12
  `mtp.layers.0` entries; that was wrong — `mtp.*` weights are in
  `_keys_to_ignore_on_load_unexpected` and never enter `named_parameters()`,
  so the catalog is pure language-model parameters.)
- **Interface metadata.** Catalog rows carry numel and per-tensor legal
  bounds, eliminating the trial-and-error class of rejections (e.g.
  count > numel). Compact row format `id|L|fam|n=..|m=cap|c<=N` keeps the
  Arm B prompt at ~5.5k tokens; the earlier verbose format produced a
  ~12.7k-token prompt whose prefill peaked at ~32 GiB on this hybrid
  linear-attention checkpoint and OOMed. Rows advertise magnitude caps
  0.1% inside the true bound so boundary proposals do not fail on
  display-rounding.
- **Measured quality baseline.** `probe_ppl()` on the unmodified pinned
  checkpoint (7.460) replaces the hard-coded Phase B constant; collapse
  detection remains separate from selection/regression endpoints.
- **Capability suite** (`gate1_capability.py`): 13 tasks across longctx
  (ppl), reasoning (6), code (4), structured (2), scored separately from
  the ppl endpoint. Baseline: longctx 2.145, reasoning 0.333, code 0.75,
  structured 0.50. Small n → coarse sensitivity; treat as a retention
  screen, not a capability benchmark.
- **Fresh v3 partitions** (`data/gate1_eval_v3/`): 8 new original passages
  per domain, zero overlap with the examined v2 passages.
- **Crash safety:** incremental JSON checkpoint after each candidate and
  `--resume`; GPU-side rollback snapshots moved to CPU and `.float().std()`
  materializations replaced (Arm B touches ~1 GB tensors).

### Results (3 seeds x 4 steps x 2 arms x 4 conditions = 24 candidates, 985.5s)

Proposal quality improved sharply: **96/96 steps produced schema-valid,
applied edits (100%)** vs 17/24 applied previously — the interface
metadata eliminated the rejection class entirely. All 24 trajectories
passed regression and capability screens.

Mean selection delta CE (nats/token; negative = better), n=3 seeds:

| Arm | autonomous | random | fixed_heuristic | adaptive |
|---|---|---|---|---|
| A sparse additive | +0.00069 | +0.00118 | +0.00027 | +0.00064 |
| B attenuation | +0.00205 | +0.00420 | -0.00237 | +0.00313 |

Descriptive only. Autonomous Arm A is statistically indistinguishable
from the practice-adaptive control; Arm B autonomous has the best mean
among stochastic conditions but loses to the deterministic fixed
heuristic (which produces identical trajectories across seeds — zero
variance, so it cannot support significance claims). Capability deltas
were ~0 for all candidates (coarse suite). Cost: autonomous ~72-87s per
candidate vs ~41-57s for controls — proposal generation roughly doubles
wall time; the adaptive control matches practice-evaluation budget but
not proposal-generation compute.

### Interpretation

The corrected interface achieved its engineering goal (valid proposals,
matched search space, measured baselines, capability retention screen).
It did **not** produce evidence that Qwen's parameter selection beats
matched search: on Arm A it ties the adaptive control; on Arm B the best
mean belongs to a fixed heuristic that ignores feedback entirely. Sparse
additive edits remain near-zero leverage; attenuation remains mostly
harmful on held-out text, with the fixed heuristic's Arm B improvement
(-0.0024 CE) being the largest single observed gain — and the least
informative control (deterministic, unexplained).

### Remaining confirmatory blockers (unchanged, still required)

1. Statistical protocol revision: pilot n=3 cannot calibrate variance;
   need trajectory + passage-level uncertainty model, prespecified
   one-sided tests with multiplicity control across arms, and a
   minimum-useful-effect justified by measurement noise rather than the
   provisional -0.02 nats/token target.
2. Perturbation-matched AND compute-matched comparison validation —
   current controls match search space and proposal budget; proposal-
   generation compute is not yet matched (autonomous pays ~25s/step).
3. Independent evaluator review: sealed holdout under separate OS
   account/NTFS restrictions, grading-code integrity, one-time opening.
4. Capability suite is a coarse retention screen (13 tasks); confirmatory
   needs broader coverage before any capability claim.
5. ~~Catalog contains 12 `mtp.layers.0` entries in Arm B~~ — resolved:
   `mtp.*` weights are ignored at load (`_keys_to_ignore_on_load_unexpected`)
   and never enter the catalog; the 248 Arm B entries are all
   language-model parameters.
