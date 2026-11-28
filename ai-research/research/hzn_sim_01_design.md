# HZN-SIM — Self-Improvement Memory and Developmental Lineage

Status: DESIGN DRAFT for review. Not preregistered. No implementation has
started. HZN-SIM-00 is proposed as a CPU-only substrate probe; HZN-SIM-01
would be preregistered separately before any GPU execution.

## 1. Origin and positioning

External reference: Yu et al., "Self-Improvement Memory Enables Continual
Recursive Self-Improvement" (arXiv:2611.18856v1, 13 Nov 2026; Google Cloud
AI / Google DeepMind / Penn State). Verified facts:

- Four transfer conditions: Cold Start, Harness Transfer, Problem-Only
  Memory, Full SIM. Full SIM best in main comparisons; +2.36 mean over
  harness-only across 10 paired coding runs (p=0.003). Other domains are
  single runs; the workspace +11.33 figure is n=1 on an LLM-judged benchmark.
- SIM records problem entries keyed by root-cause hypothesis; attempts record
  `predicted` (pre-evaluation) vs `fixed`/`missed` (post-evaluation),
  provenance via `carried_from`, alias merging. Records persist after edits
  are replaced.
- Entry matching is done by the proposer LLM itself; merging is deterministic
  only for literally identical root-cause text. Applicability checking is
  prompt guidance ("check the domain, benchmark, and model behind each
  entry") — invalidation is advisory prose, not mechanism.
- An Opus-evolved harness starts at 68.41 on Gemini, below the seed's 73.89:
  inherited improvements can be net-negative under transfer.

Horizon positioning: the cross-model transfer failure is structurally the
stale-derived-state problem. HZN-EPI-06 measured what happens when typed
state is consumed through prose. The binding differentiator of this work is
mechanical applicability invalidation — dependency-versioned memory entries
whose validity is computed, not advised. Google cannot test this inside
their design because invalidation semantics do not exist as mechanism in it.

## 2. The core separation: computational state vs epistemic state

User's formulation, adopted:

    M_{t+1} = M_t            (computational state rolls back)
    K_{t+1} = K_t ∪ E_t      (epistemic state advances)

Rollback without amnesia: Jarvis can return to a previous version of himself
without returning to a previous level of understanding. Failed interventions
become navigational landmarks; a failed experiment is not a failed iteration.

Refinement (binding design constraint): the union is correct for the record
but insufficient for interpretation. Two layers must be distinguished:

1. Immutable measurements — raw scores, latencies, token counts, diffs,
   seeds, evaluation identifiers. Append-only, never revised, never deleted.
2. Derived interpretations — diagnoses, root-cause assignments, refutation
   judgments, applicability scope. These are versioned derived states: they
   carry dependency versions and can transition to `stale` or
   `applicability_unknown` when their evidential basis changes, exactly like
   cognitive-bus derived state. The interpretation of old evidence is itself
   subject to invalidation. Without this, K drifts: a confident misdiagnosis
   at iteration 3 becomes load-bearing misinformation at iteration 30.

## 3. Record schema (hypothesis-centric)

Breadcrumbs are organized by hypothesis, not by intervention, because failure
is underdetermined: an intervention can fail because the hypothesis was
wrong, the implementation was wrong, the dose was wrong, or the evaluation
was insensitive. Google's problem-grouping got this right. Interventions are
tests of hypotheses; evidence attaches to the hypothesis.

Proposed fields per hypothesis entry:

    id, created_round, status
    statement              # the causal claim being tested
    ancestry               # which model/harness versions produced it
    rationale              # why it was believed
    attempts[]             # interventions executed as tests
      target, mechanism, parent_state, diff/artifacts (references, not blobs)
      prediction           # quantitative where possible: predicted metric
                           # deltas with direction AND magnitude
      consequence          # observed measurements (immutable)
      prediction_error     # computed mechanically, never narrated
      decision             # accepted / rejected / rolled_back
    failure_conditions     # under which circumstances failures occurred
    reconsideration        # see below
    dependencies           # (model_checkpoint, harness_version,
                           #  eval_suite_version) — drives invalidation

Reconsideration conditions are divided into two classes:

- Machine-checkable: "reconsider if model family changes", "reconsider if
  attention implementation is replaced", "reconsider if eval suite version
  increments". Evaluated mechanically; trigger automatic status transitions.
- Judgment-requiring: "reconsider if new evidence about sparse interventions
  emerges". Cannot be mechanically evaluated. HZN-EPI-06 measured this class
  as fragile; these must be rendered as explicit open questions in proposal
  context rather than silently trusted to neural judgment.

Predictions are quantitative with magnitudes — not Google's binary per-task
form — so calibration is computable.

## 4. Status semantics (mechanical)

Entry statuses: `proposed`, `testing`, `supported`, `refuted_conditional`,
`stale`, `applicability_unknown`, `superseded`, `insufficient_evidence`.

Invalidation propagation follows the cognitive-bus rule fixed in efc14c0:
dependency-version checking precedes intrinsic status. Concrete rules:

- Model checkpoint changes (weight intervention accepted, new base model):
  entries whose evidence depends on model behavior transition to
  `applicability_unknown` unless their dependencies are model-independent.
- Harness version changes: harness-dependent entries invalidate likewise.
- Eval suite changes: entries whose consequences were measured on the old
  suite become `stale` for comparability, not for truth.
- Revalidation is cheap by design: re-test the recorded prediction, not
  re-derive the hypothesis. A revalidated entry returns to its prior status
  with the revalidation appended — history accumulates, never rewrites.

A bad choice today may be viable under a different architecture tomorrow;
evidence against an intervention is preserved, never converted to permanent
prohibition. `refuted_conditional` always carries its conditions.

## 5. Graph structure

Two interlinked graphs, not one:

- State lineage graph (computational): checkpoints and harness versions,
  edges = interventions executed. This is a DAG; rollbacks create branch
  points but the full branch structure is retained — the history of who he
  could have become, not just who he became.
- Evidence/decision graph (epistemic): hypotheses, evidence entries,
  decisions. Edge types: motivated_by, tested_by, refuted_by, supported_by,
  superseded_by, invalidated_by, reconsidered_by.

Cross-references bind them: every evidence entry names the state-lineage node
it was produced under. Chronology establishes sequence; causal relationships
establish meaning. Append-only everywhere; status is derived, never
destructive.

## 6. HZN-SIM-00 — substrate probe (CPU-only, no GPU)

Scope: implement the memory substrate on cognitive-bus machinery and test
its semantics deterministically, the same pattern the bus itself followed.

Components:

1. Versioned causal memory records (schema above) with dependency
   invalidation wired through the existing bus version machinery.
2. Mechanical applicability computation on transfer events.
3. Structural retrieval: given current hypothesis/decision context, retrieve
   relevant entries by structural keys (target component, failure class,
   dependency overlap) — no LLM in the retrieval path.
4. Validation corpus: real Horizon history (gate0_5 failure taxonomy,
   EPI-01..06 interventions and outcomes) encoded as memory entries. The
   corpus is genuine developmental history, including the m02 transcription
   failure and the ambiguous-tie staleness bug — negative results as
   landmarks, which is the point of the exercise.
5. Deterministic tests: invalidation propagation on simulated transfer,
   rollback-without-amnesia semantics, reconsideration triggers, append-only
   integrity, interpretation-staleness recursion.

Gate: all deterministic tests pass and the real corpus round-trips before
any neural-consumption experiment is designed in detail.

## 7. HZN-SIM-01 — memory-conditional improvement (would be preregistered)

Arms (user's proposal, unchanged):

    A  stateless proposer
    B  linguistic SIM (prose history, Google-style consumption)
    C  structural SIM (mechanical retrieval/rendering)
    D  structural SIM with prediction-error feedback

Held constant: initial checkpoint, candidate budget, evaluation partitions,
intervention constraints, proposer model.

Primary endpoint — the user's measurable core: prediction-error calibration
across iterations with M held fixed. Does the predicted-vs-observed gap on
intervention outcomes shrink along the developmental sequence? This
separates epistemic improvement from neural improvement, which held-out loss
alone cannot.

Secondary endpoints: held-out loss improvement; recurrence of previously
unsuccessful interventions; regression frequency; improvement per GPU-second
and per generated token; transfer of useful experience across checkpoints;
premature-closure rate (viable hypotheses never retested).

The no-change option is a first-class action with restraint controls, per
the user's requirement: remembering why an intervention is unnecessary is as
valuable as discovering a successful one. Wu Wei as a measured outcome, not
a slogan.

Budget realism: arms A-D with repetitions are a large GPU commitment because
interventions actually execute and re-evaluate — iteration cost dominates.
The replication standard should exceed the source paper (they repeated only
coding; we should repeat everything or say exactly why not).

## 8. Failure modes and falsification

- Premature closure / negative-result tyranny: over-respecting breadcrumbs
  suppresses legitimate retesting. Mitigated by conditional refutation and
  reconsideration machinery; measured as its own endpoint.
- Epistemic drift: misinterpreted evidence compounds. Mitigated by immutable
  measurements + revisionary interpretations; tested in SIM-00.
- Token-count confound: Full SIM-style arms have strictly more context than
  stateless ones. Arm B vs C comparison partially controls; a token-matched
  irrelevant-history control should be included if affordable.
- Weak proposer: a 4B local model is a far weaker proposer than the Opus-
  class optimizers in the reference paper. Memory effects may be larger or
  smaller; results generalize to weak-proposer regimes only, and the
  writeup must say so.
- What kills the approach: if structural SIM (C/D) does not beat linguistic
  SIM (B) on calibration or recurrence, the core Horizon bet — structure
  over prose in the consumption path — takes a serious hit at exactly the
  scale where it matters most. That outcome must remain genuinely possible.

## 9. Open forks requiring user decision

1. Intervention class for the HZN-SIM-01 loop: harness-level edits (cheap,
   literature-proven, but not Horizon's weight-space program) vs weight-
   space interventions (the real research program, expensive per iteration)
   vs a staged design (harness first as a memory-architecture probe, weights
   after the memory is proven).
2. Structural hypothesis matching: adopt the gate0_5 failure taxonomy as the
   initial ontology, or something simpler and domain-specific.
3. Whether HZN-SIM-00's corpus encoding of real Horizon history is done by
   hand (small, careful) or semi-automatically from evidence JSONs (larger,
   noisier).
