# Gate 1.2 — Observation-Layer Specification (REV 1, post-HZN-IR-02)

Status: **specification only — no implementation authorized.** Revised per
independent review HZN-IR-02. All required amendments are incorporated:
evidence-matched and placebo controls, a three-hypothesis falsification
framework, observation-sufficiency requirements, and provenance rules.
It changes no edit surfaces, model families, endpoints, or confirmatory
blockers.

## 1. Scientific motivation

Gate 1.1 showed the proposer reliably produces schema-valid edits (24/24
autonomous steps applied; 96/96 across all conditions) but its selections
do not beat matched controls. Diagnosis, per the Frontier Synthesis and
HZN-IR-01: the catalog is a *parts inventory* — every row is structurally
anonymous. Fan et al.'s successful 27B model intervened *after* identifying
a contrast between internal states associated with different behaviors.
Our 4B model currently has no channel through which functional evidence
can reach it.

This specification distinguishes three separate questions — which prior
wording conflated:

1. Does functional information **change** the model's decisions?
   (behavioral responsiveness)
2. Does that information lead to **better** decisions?
   (diagnostic utility)
3. Does the model interpret that information **more effectively than a
   conventional algorithm** given equivalent resources?
   (interpretive advantage)

A model can be responsive without being useful, and useful without being
better than the algorithm. Each question gets its own hypothesis and its
own evidence (§5). Demonstrating (1) alone is NOT support for the broader
claim; that distinction is the point of the placebo control (§4).

**Non-goals:** proving causal responsibility, mechanistic understanding,
or that the model "understands itself." Activation statistics are
*correlational descriptors* presented as diagnostic input only.

## 2. Proposed observation: high/low-loss activation contrast

Mimic the published diagnostic structure without gradients:

1. Compute per-passage loss on **practice** passages only.
2. Split into high-loss vs. low-loss subsets, **stratified by domain and
   matched on passage length where feasible** (see measurement
   requirements below — a high-loss passage is less predictable, not
   necessarily "incorrect reasoning"; subject matter, token frequency and
   length are confounds).
3. Register forward hooks capturing per-module activation statistics
   (output-tensor RMS and max |activation|) for each catalog parameter's
   owning module, separately on high- and low-loss subsets. **Count and
   record samples per subset** — variation across passages must be
   reported, not just means.
4. Emit per-row values:

   `act` = mean output RMS on all practice passages (scale of use)
   `hilo` = stabilized high/low contrast: `(hi + eps) / (lo + eps)` with a
   fixed `eps` derived from the activation distribution, so near-zero
   denominators cannot produce infinite ratios. Report the raw `hi`/`lo`
   alongside so the stabilization is auditable.

   Both values are also emitted **family-normalized** (rank within the
   row's parameter family) so cross-family magnitudes do not dominate the
   prompt: `act_rank`, `hilo_rank` in [0,1].

5. Catalog row becomes:
   `id|L|fam|n=..[|m=cap][|c<=N][|act=..|hilo=..|ar=..|hr=..]`
   (compact rank values only if prompt cost requires; full stats in the
   machine-readable catalog regardless).

### Measurement requirements (HZN-IR-02 §4)

| Concern | Rule |
|---|---|
| Passage confounding | Stratify hi/lo split by domain; record per-domain counts; note residual confounds (length, token frequency, subject) as limitations, not conclusions |
| Near-zero denominator | Stabilized ratio with fixed eps; flag rows where `lo` is below the stability floor |
| Cross-family comparability | Report family-normalized ranks alongside raw values |
| Measurement uncertainty | Record per-row sample counts and variance across practice passages |
| Shared parent modules | Parameters sharing a hooked parent module receive identical observations; the catalog records `obs_group` so identical values are not mistaken for independent evidence |
| Unexecuted modules | If a catalog parameter's owning module never fires during the observation passes, its observation is marked **unavailable** — never written as 0 |
| Baseline dependence | Every observation is labeled with the checkpoint hash it was measured on |

**No directional prescription.** The prompt text must NOT suggest that
high-`hilo` or low-`act` rows are preferred targets — high activation is
not inherently undesirable and may reflect compensation, not dysfunction.
We measure whether the proposal distribution changes across conditions,
without encoding which direction of use is "correct." Prompt adds one
neutral sentence: *"act/mean output scale and hilo/high-vs-low-loss
contrast are correlational observations measured on the unmodified
checkpoint; ranks are within-family."*

## 3. Implementation sketch (if approved)

- `gate1_observation.py`: forward hooks on the parent module of each
  catalog parameter. **Module mapping must be verified** — hook targets
  are resolved by walking the module tree to each parameter's direct
  owning `nn.Module` (not by name-prefix matching, which can collide);
  each catalog row records `obs_module` (the exact module path hooked)
  and `obs_group` (deduplicated module id). Parameters whose parent is
  shared (e.g. fused QKV projections) carry the same `obs_group`.
- Two practice passes (sorted by baseline loss, stratified) with
  `torch.no_grad()`. Cost: ~2 extra practice forward passes — negligible
  vs. evaluation budget.
- Statistics computed ONCE per run on the unmodified checkpoint and held
  fixed for all candidates/conditions. **Provenance:** the observation
  block carries `checkpoint_sha256` + the per-row sample counts; it is
  hashed into the run manifest (§6 code corrections).
- **Memory:** hook buffers accumulate scalar statistics only (running
  mean/max), never retained tensors; activation tensors are released per
  module call. Verify peak VRAM headroom on the A4500 before the run —
  the observation pass adds no persistent GPU state beyond per-module
  scalars.
- No gradients anywhere — consistent with the no-gradient rule.
- **Static-observation limitation (documented):** observations describe
  the unmodified checkpoint. After cumulative candidate edits, the
  proposer's shown observations may no longer reflect the model's actual
  internal state. This is a known limitation of the static design, not
  evidence against introspection generally; no dynamic observation
  mechanism is introduced in this study.

### Observability ladder — what this study does and does not establish

| Level | Name | Status |
|---|---|---|
| 0 | Structural visibility — parameter names, shapes, families, bounds | Demonstrated (Gate 1.1) |
| 1 | Functional visibility — access to externally measured internal activity | Proposed here |
| 2 | Diagnostic competence — interpretation with measurable utility | Unestablished (H-OBS.2 tests this) |
| 3 | Transfer — diagnostic knowledge generalizes to unfamiliar conditions | Unestablished |

Gate 1.2 tests the 0→1 transition and whether 1 produces measurable
utility toward 2. It does not establish internal self-awareness,
transferable abstractions, or autonomous introspection.

## 4. Control fairness — six-condition comparison architecture

Per HZN-IR-02, three questions need three kinds of conditions. Option C
(model + evidence-matched algorithmic controls) is adopted, plus a
**shuffled-observation placebo**.

| Condition | Proposer | Information | Tests |
|---|---|---|---|
| A0 | model | structural catalog | autonomous baseline |
| A1 | model | structural + real observations | diagnostic usefulness |
| A2 | model | structural + shuffled observations | placebo: controls for "any numbers at all" |
| B0 | conventional adaptive | structural catalog | algorithmic baseline |
| B1 | conventional adaptive | structural + real observations | algorithmic interpretation of the same evidence |
| B2 | conventional adaptive | structural + shuffled observations | secondary placebo |

**Why A2/B2 (shuffled placebo) is required:** a model preferring row 101
because it has the largest `hilo` shows numerical preference, not
diagnostic reasoning. Shuffling preserves numerical content, formatting
and prompt length while breaking the observation↔parameter
correspondence. If A1 ≈ A2, there is little evidence the *correct
association* matters. Shuffle is deterministic and seeded, performed
**within family/layer strata where feasible** (so a shuffled row gets a
plausible observation for its type), and the permutation is recorded in
the run report.

**Budget fairness — two separate notions, both recorded:**

- *Intervention-matched:* every condition gets the same number of
  permitted proposal opportunities per trajectory.
- *Compute-matched:* record wall-clock + GPU seconds for observation
  collection, proposal generation, evaluation, and rollback separately,
  including invalid proposals and abstentions. A conventional algorithm
  can evaluate more candidates in the model's proposal latency; the two
  fairness notions are reported side by side and neither substitutes for
  the other.

**Priority if the exploratory budget cannot cover all six:** A0, A1, A2,
B0, B1. B2 is useful but secondary. The ~30-minute runtime target is a
resource ceiling, not an estimate and not statistical justification for
the sample size.

## 5. Falsification — three nested hypotheses

The single earlier hypothesis conflated responsiveness with benefit.
Replace with progressively stronger claims:

- **H-OBS.1 — Diagnostic responsiveness.** The model responds
  specifically to correctly associated functional observations.
  Primary comparison: A1 vs A2 (and A0). Evidence: proposal-distribution
  change tied to the real correspondence, not to numbers in general.
- **H-OBS.2 — Diagnostic utility.** That response produces better
  independently measured outcomes without unacceptable regressions.
  Primary comparison: A1 vs A0 and A2 on selection CE + retention.
- **H-OBS.3 — Interpretive advantage.** The model extracts more useful
  information than a prespecified conventional search strategy at
  equivalent resources. Primary comparison: A1 vs B1.

Support for one does not imply the next.

### Outcome interpretation

| Observed result | Interpretation |
|---|---|
| A1 ≈ A2 | No demonstrated benefit from correct evidence correspondence |
| Real obs change proposals but worsen outcomes | Responsiveness (H-OBS.1); utility (H-OBS.2) not demonstrated |
| A1 ≈ B1, both improve | Useful diagnostic signal; no demonstrated model advantage |
| A1 > B1 | Preliminary evidence of interpretive advantage (H-OBS.3) |
| Neither model nor algorithm benefits | Observation representation may be insufficient — revise representation, not necessarily abandon |
| Model abstains from all interventions | Valid outcome; evaluate against the unchanged checkpoint |

### Statistical formulation

Candidate utility on the development endpoint:
`U = CE_baseline − CE_candidate` (positive = improvement).

- `D_obs = U_A1 − U_A0` — incremental value of correctly associated observations
- `D_specific = U_A1 − U_A2` — evidence-specific advantage over placebo
- `D_interpret = U_A1 − U_B1` — interpretive advantage over the algorithm

All are evaluated alongside retention requirements, never as sufficient
conditions for promotion. **With n=3 seeds these are descriptive only.**
An inconclusive result is not a falsification. A confirmatory study would
need predefined minimum useful effects, power, independent evaluation,
and a formal equivalence/inferiority criterion — outside this protocol.

## 6. Explicit non-changes

- No new edit surfaces, magnitudes, or arm boundaries.
- No gradient-based attribution (prohibited by protocol).
- No selection/regression/holdout content in observations — practice
  passages only.
- Prompt isolation remains prompt isolation; no new access boundary.
- n=3 seeds remains exploratory; confirmatory protocol still frozen.
- Code-integrity corrections listed in HZN-IR-02 §6 are applied to the
  *existing* Gate 1.1 harness (manifest descriptor hashing, shard
  binding, MTP execution-path membership, ledger strictness, grader
  boundary documentation) — they do not authorize execution of this
  specification.

## 7. Review provenance

- HZN-IR-01 (Gate 1.1): approved premise; identified interface vs.
  selection distinction and §6 code findings.
- HZN-IR-02 (this spec, REV 0): approved in principle; required
  amendments in §§2–5 incorporated here. Execution remains pending
  review of this revision.

*Doctrine note (carried from HZN-IR-02): observability precedes
diagnosability; diagnosability does not guarantee controllability;
controllability does not guarantee improvement. Each is a separate
capability requiring its own evidence.*
