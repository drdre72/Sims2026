# HZN-SIM-00.5 (PREREGISTRATION) — Counterfactual Replay Preflight: retrieval sanity before SIM-01

**Status: FINAL — designed and frozen before any retrieval execution. Committed before implementation.**
CPU-only, no GPU, no neural calls, no network. Fully deterministic (seeded RNG). This is a
preflight/sanity check of the HZN-SIM-00 substrate against curated real history, not evidence of
memory utility inside an improvement loop.

---

## 1. Motivation — isolating failure mode #1

The staged SIM design separates three failure modes of a memory-informed improvement loop:

1. the system cannot **retrieve** the appropriate prior experience;
2. it cannot correctly **interpret** the experience it retrieves;
3. it interprets correctly but cannot **implement** the intervention.

HZN-SIM-01A's headline metric conflates all three. This preflight isolates #1 using the one
ground-truth corpus we already possess: the curated developmental history itself
(`sim_corpus.build_gold_corpus()`, 13 hypotheses, 14 measurements, availability ticks ordered by
real experiment chronology).

The honest framing: historically, the *human* was the retrieval mechanism — the EPI-06 design
consumed the m02 lesson because it lived in the research log and the curator's head. This
experiment asks whether the substrate alone, queried with the structural context each decision
document actually named, would have surfaced the landmarks that mattered. At every decision point
the query is restricted to evidence available at that tick (`retrieve(at_tick=...)`), so nothing
that was not yet knowable can leak into the counterfactual.

## 2. What this preflight is not

- **Not** a measurement of neural consumption, proposer quality, or loop dynamics — those are
  SIM-01 questions (failure modes #2 and #3).
- **Not** an unbiased discoverability test: the corpus and the decision-point table are
  single-authored, and the curator knew the landmarks when curating. The preflight measures the
  *structural connectivity of the curated graph*, i.e. whether the ontology/dependency keys the
  decision documents themselves provide are sufficient to reach the documented landmarks.
- **Not** tuning-friendly: no retrieval parameter, frame, or threshold is adjusted after seeing
  results. Any adjustment requires a versioned preflight v2 with both runs preserved.
- Small N by design: 6 decision points, 7 scored landmarks. Negative and mixed results are
  reported with the same standard as positive ones.

## 3. Battery — decision points and gold landmarks

Each decision point (DP) names: the tick at which the decision was made, the historical decision,
its source document(s), the query frame(s), and the required landmark(s). **Frame derivation rule**
(binding): a frame key may only use (a) an ontology category whose *concept* the decision's own
source document names, or (b) a `dep_type` for a component the document states it reuses or
touches. Frames are curated with per-DP justification and frozen in
`research/evidence/sim_00_5_decision_points_v1.json` **before any retrieval run**.

Frame semantics follow the substrate implementation exactly: multiple keys act as an **OR-union**
(any single key match includes a hypothesis); `retrieve()` is filter-based candidate generation,
not ranking. Two frame levels per DP:

- **frame-minimal** — the single most specific documented key (precision-oriented);
- **frame-full** — all keys derivable by the rule (the realistic diligent-consumer query).

Scoring is on frame-full; frame-minimal is reported. A hypothesis that is the *subject* of the
decision (the thing being decided about) is never scored as a landmark — surfacing the subject is
trivial, not memory utility.

| DP | tick | decision (source) | frame-minimal | frame-full adds | required landmarks |
|---|---|---|---|---|---|
| DP-1 | 20 | Phase B self-edit runs designed to resolve the Phase A open selection hypothesis (`gate0_5_phaseb_selfedit.md`; taxonomy Phase A closing) | `ontology_category=self_edit_avoidance` | `dep_type=model_checkpoint` | `H-G05-SELFEDIT-MECHANISM` |
| DP-2 | 30 | gate1_2 observation campaign designed against the flagged safe-null risk (`gate1_2_campaign_report.md`; taxonomy "the real Gate 1 risk") | `ontology_category=safe_null_surface` | `ontology_category=input_boundary_fragility`, `dep_type=model_checkpoint` | `H-G05-SAFE-NULL` |
| DP-3 | 90 | EPI-06 battery design reusing the EPI-05 battery, its stale-invalidation case, and the harness (`hzn_epi_06_consumption_prereg.md` §1, §3) | `ontology_category=invalidation_semantics` | `ontology_category=transcription_failure`, `dep_type=harness` | `H-EPI05-BINDING`, `H-BUS-STALE-PRECEDENCE`, `H-EPI05-M02` |
| DP-4 | 83 | tie-bug fix design (`efc14c0`; novel semantics bug — **no-landmark control**) | `ontology_category=invalidation_semantics` | `dep_type=harness` | none (control) |
| DP-5 | 99 | HZN-SIM-00 substrate design motivated by the policy-boundary collapse (`hzn_sim_01_design.md`) | `ontology_category=policy_boundary_confusion` | `ontology_category=structural_binding_success`, `dep_type=model_checkpoint` | `H-EPI06-POLICY-BOUNDARY` |
| DP-6 | 94 | EPI-06 H1-failure interpretation: seven residual `none`-collapses diagnosed as epistemic distinction failure (corpus interpretation @94) | `ontology_category=policy_boundary_confusion` | `dep_type=model_checkpoint` | `H-EPI05-M02` |

Related-but-unscored landmarks (recorded, not scored): DP-1 `H-G05-COHERENT-LETHAL`,
`H-G05-DEPTH-GRADIENT`; DP-2 `H-G05-DEPTH-GRADIENT`, `H-G05-COHERENT-LETHAL`; DP-5
`H-EPI06-MECH-RENDER`, `H-EPI05-BINDING`; DP-4 subject `H-BUS-STALE-PRECEDENCE` (excluded from
scoring by the subject rule). Total: 7 required landmarks across 5 scored DPs; DP-4 is the
no-landmark control measuring false-relevance discipline and distractor load only.

Justifications are per-landmark in the frozen JSON (e.g. DP-3's `H-EPI05-M02`: the EPI-06 prereg
motivation names neural retranscription fragility, and m02 is the recorded transcription-failure
landmark; DP-3's `H-BUS-STALE-PRECEDENCE`: the prereg §3 explicitly reuses "the same stale
invalidation case").

## 4. Arms

All arms run at every DP with `clearance="evolve"` on a fresh `build_gold_corpus()` instance.

| arm | condition | purpose |
|---|---|---|
| S — STRUCTURAL | `retrieve()` with the frozen frame-full keys, `at_tick=DP.tick` | the substrate under test |
| R — RANDOM floor | seeded uniform sample (without replacement) of size K from all visible hypotheses at the DP tick; K = |arm S candidates|; 1000 draws (seed 20261128 + DP index) plus the analytic K/N expectation | lower bound |
| Y — RECENCY heuristic | top-K visible hypotheses by `availability_tick` descending (ties by hypothesis_id ascending) | "just use recent history" baseline |
| L — LEAKAGE probe | arm S keys with `at_tick=None` | validity check, not a comparison arm |

No ranking is added to the substrate; R and Y orderings live only in the preflight harness.
Arms R and Y are compared at arm S's candidate budget so arms differ in *selection policy*, not
budget.

## 5. Metrics (preregistered)

Per DP (frame-full primary, frame-minimal reported):

- **hit**: all required landmarks present in the candidate set; **partial**: some; **miss**: none.
- **landmark recall**: surfaced required landmarks / total required (7 overall).
- **candidate-set size** and **distractor load**: |candidates| − surfaced landmarks — the real
  consumer cost of over-broad retrieval.
- **per-landmark miss-cause**, classified exactly one of:
  - `category_absent` — no visible hypothesis holds the queried category at the tick;
  - `vocabulary_mismatch` — the landmark exists and is visible but its pinned category/keys do not
    match any frame key;
  - `dependency_not_declared` — the landmark lacks the dependency the frame keys on;
  - `not_available_at_tick` — landmark availability > DP tick (should be impossible by construction;
    a hit of this class is a preflight-invalidating bug).
- **arm comparison at equal budget**: arm S hit vs R expected hit (mean over draws) vs Y hit,
  reported per DP and aggregated.

Leakage (binding invariants):

- every at_tick run must contain **zero** entries with `availability_tick > DP.tick`; any violation
  invalidates the preflight (substrate bug, fix and re-run in full, both preserved);
- arm L must include future-only evidence (`availability_tick > DP.tick`) for at least one DP,
  else the availability filter is vacuous and this is reported as a failed validity check.

**Status fidelity check** (bug detector): for every retrieved candidate, compare the end-state
`hypothesis_status()` against the at-tick status (last interpretation in the chain with
`availability_tick ≤ DP.tick`). Report mismatches. Known exposure: `hypothesis_status()` is
computed from the end-state chain head, so a tick-20 view of a hypothesis revised at tick 23 would
carry the post-hoc status — the replay filters *availability*, not chain history. Any mismatch is
recorded as a substrate replay-fidelity gap for SIM-01, not silently accepted.

**Cascade breadth** (secondary, descriptive — no thresholds):

- *historical events as recorded*: the bus chain revision (ticks 82/84) and the ontology v0.2
  revision during corpus build — affected hypotheses counted via effective-status propagation.
- *synthetic counterfactual probes, labeled as such*: P1 model replacement
  (`registry.replace` model dep) → `applicability_unknown` count; P2 harness bump → stale count;
  P3 eval bump → stale count; P4 confound registration on `epi06-h1-nonnormal` → stale count via
  measurement deps; P5 ontology bump v3 → stale count via ontology deps.
- each probe on a fresh corpus instance; report affected count, affected/13, and propagation
  depth. Sparse-graph findings (e.g. zero interpretation-to-interpretation edges in the corpus)
  are recorded as **corpus dependency-declaration gaps**, not substrate failures.

## 6. Decision logic (binding)

These thresholds are go/no-go design signals for SIM-01A, not evidence of loop-level utility:

- **Retrieval utility supported** if: frame-full landmark recall ≥ **5/7**, full-hit on ≥ **3/5**
  scored DPs, arm S hit ≥ arm R expected hit and ≥ arm Y hit on a majority of DPs, and both
  leakage invariants hold. Proceed to SIM-01A preregistration with retrieval as-is.
- **Supported with precision warning** if utility holds but mean frame-full candidate-set size
  exceeds **50%** of visible hypotheses at the DP tick: recall is achieved only through breadth.
  SIM-01A must add deterministic narrowing/ranking before the loop; recorded as a design
  requirement, not a failure.
- **Vocabulary-gap failure** if recall < 5/7 with misses dominated by `category_absent` /
  `vocabulary_mismatch`: ontology refinement and corpus re-encoding come before SIM-01A. Missed
  landmarks are **not rescored** after post-hoc category additions; the fix informs SIM-01, it
  does not retro-improve this preflight.
- **Mechanism failure** if misses are dominated by `dependency_not_declared` or
  `not_available_at_tick`: substrate/dependency-declaration redesign before SIM-01A.
- Mixed outcomes outside these bounds are reported as inconclusive, not forced to the nearest
  verdict. Cascade-breadth findings are descriptive throughout.
- The no-landmark control DP-4 is never scored for hits; it exists to verify the preflight does
  not manufacture relevance, and to measure distractor load on a genuinely novel decision.

## 7. Controls and honesty constraints

- **Frozen before execution**: the DP table (`sim_00_5_decision_points_v1.json`) is committed
  before the runner exists. Two commits, mirroring EPI-06's prereg-then-execute discipline.
- **Single-author confound stated up front** (§2); this is a structural-connectivity measurement
  on a curated graph.
- **No tuning at this N**: the entire preflight is exploratory with respect to any SIM-01 design
  choice; nothing is optimized against its outcomes.
- **Substrate frozen**: `sim_memory.py` / `sim_corpus.py` semantics are not modified for this
  preflight. A bug found (including via the status-fidelity check or leakage invariants) is fixed
  in a follow-up commit with regression tests, and the preflight re-run in full with both result
  sets preserved.
- Negative results preserved and committed with the same evidentiary standard.
- No GPU, no neural calls, no network. Determinism: seeded RNG (`random.Random(20261128)`),
  fresh corpus per probe, run digest recorded.

## 8. Implementation notes

- New module `src/jarvis_research/sim_replay.py`: DP loading/validation, arms S/R/Y/L, metrics,
  miss-cause classification, status-fidelity check, cascade probes, evidence export.
- Frozen DP table: `research/evidence/sim_00_5_decision_points_v1.json` (this commit).
- Runner `results/sim_00_5/run_replay.py` (gitignored output); curated evidence
  `research/evidence/sim_00_5_replay_v1.json`; research log section 42.
- Deterministic tests `tests/test_sim_replay.py`: DP-table schema and tick/availability validity
  (all landmarks available at their DP tick — guards the `not_available_at_tick` class), leakage
  invariant (zero future entries under at_tick), seeded reproducibility (same seed → same digest),
  miss-cause classifier coverage, cascade counts on the known corpus, and the equal-budget arm
  mechanics.
- Full CPU suite green before the prereg commit; no GPU execution exists for this preflight —
  CPU validation is the entire execution.
