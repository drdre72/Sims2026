# HZN-SIM related work — literature comparison for novelty claims

**Status: focused pass, 2026-11-28. Companion to `hzn_sim_01_design.md` and the HZN-SIM-00.5 prereg.**
Not a systematic review. Each entry is tagged by verification depth: `[read]` paper/repo examined in
detail (in this project's history or this pass), `[abstract]` abstract-level or search-result-level
only, `[third-party]` claims about a paper sourced from secondary reviews. Conclusions at the end
state what novelty claims remain defensible and which must be abandoned.

---

## 1. Classical ancestors — truth maintenance and belief revision

The mechanical core of HZN-SIM-00 (dependency-scoped invalidation, append-only records, derived
status) is a member of a family that predates LLMs by decades:

- **Doyle's justification-based TMS (1979); assumption-based TMS; dependency-directed
  backtracking (Stallman & Sussman 1977)** `[read, survey level]` — beliefs carry justifications;
  retracting a premise retracts everything that depended on it; contradiction triggers
  dependency-directed rather than chronological retraction. SIM-00's `effective_status`
  propagation over `depends_on` pins is this idea. **Any novelty claim must be scoped to the
  application, not the mechanism.**
- **AGM belief revision (1985) and Hansson belief bases** `[read, survey level]` — epistemic
  entrenchment and contraction operators for revising a belief set. Surveys are now explicitly
  roadmaping TMS/AGM revival for LLM agents (arXiv:2608.14567 `[abstract]`).

## 2. Modern belief-revision memory for LLM agents

- **BELIEFMEM** (Findings-EMNLP 2026) `[abstract]` — operationalizes AGM operators as write
  semantics for a conversational fact base: history-preserving, never deletes, superseded beliefs
  retain predecessor pointers, statuses ACTIVE/SUPERSEDED/RETRACTED. This is the closest modern
  academic relative of SIM-00's interpretation chains. Divergences that survive: revision
  classification is an **LLM-driven resolver** (neural judgment, not mechanical dependency
  invalidation); the object domain is conversational facts, not a developmental record with
  versioned external dependencies (model/benchmark/harness/ontology); no availability-time
  replay, no reconsideration machinery, no prediction-error endpoints.
- **Palinode** (independent research runtime, italoc.dev) `[abstract]` — a JTMS over an LLM
  agent's working context: claims as nodes, explicit justifications, refute-and-retract, an
  append-only ledger, and a hard rule that the graph layer makes no LLM calls (semantic judgment
  enters as provenance-tagged facts; consequences are decided deterministically). Notably, this
  independently arrives at Horizon's division-of-labor thesis — mechanical consequence, neural
  judgment only at extraction. Not (to our knowledge) academically published; conversational
  claims domain; no versioned-external-dependency invalidation, no developmental corpus.

## 3. Agent experiential memory (success/failure learning)

- **Reflexion** (Shinn et al. 2023) `[read, prior project history]` — verbal self-critique
  prepended on retry. No persistence structure, no invalidation.
- **ExpeL** (Zhao et al. 2024) `[abstract]` — experience pool of successful and failed
  trajectories; LLM-extracted "insights" from contrasts. Known limits (third-party analysis):
  insight lists are concatenated wholesale — no selective retrieval, no versioning, no
  retirement, no invalidation.
- **Agent memory surveys** (arXiv:2610.35089; 2602.06052; 2603.07670) `[abstract]` — these name
  the exact gap SIM-00 targets, which is worth quoting because it is independent confirmation
  that the gap is recognized and open: procedural memory "abstracts away the evidence that
  justified them" and "therefore needs skill validation, versioning, retirement, and rollback,
  not only retrieval"; reflective memory's central risks are "self-reinforcing error" and
  "over-generalization," with quality gates "still underdeveloped." SIM-00's conditional
  refutation + reconsideration conditions + dependency invalidation are a mechanical answer to
  precisely these two named risks.

## 4. Self-improvement loops with recorded history

- **Google SIM** (arXiv:2611.18856, Yu et al.) `[read, prior session, in detail]` — the
  comparison anchor, previously verified: problem-grouped SIM records, attempts with binary
  predicted/fixed/missed, applicability checking delegated to the proposer LLM via prose prompt
  guidance, cross-model transfer failure. Notably their applicability checking is advisory —
  this remains the sharpest architectural contrast to mechanical invalidation.
- **RRSI** (arXiv:2609.24972, google-research/rrsi) `[abstract + repo + third-party reviews]` —
  regularized recursive self-improvement of agent harnesses. Records for every evaluated
  candidate: component, hypothesis, source diff, score/cost change, accepted flag. The proposer
  conditions on this history; "rejected mechanisms remain negative evidence." Third-party
  reviews surface two design facts directly relevant to Horizon's divergence:
  1. the ledger "is evidence, not belief" — nothing marks a hypothesis falsified beyond outcome
     fields; memory dies by overwrite; and
  2. the accepted flag `a_i = 0` conflates "blocked by a rule" with "lost to a better candidate" —
     the same flat-rejection ambiguity SIM-00's failure_conditions + reconsider_machine
     conditions are designed to separate mechanically.
  RRSI has no applicability invalidation on dependency change, no interpretation revision, no
  availability-time replay. (Also relevant: its structured-exploration stall detector — the
  premature-closure concern Horizon handles via reconsideration instead.)
- **Causal Improvement Graph** `[third-party, via the SIM paper's related work]` — reported to
  link evidence/hypotheses/interventions/outcomes. Not independently reviewed here; treat as
  unverified until read.

## 5. Prediction and calibration for self-modifying agents

This is where one of our intended differentiators is **weaker than the design doc assumed**:

- **Predict Before You Modify / IDM** (arXiv:2610.13991) `[abstract]` — an Improvement Dynamics
  Model predicts task-level outcomes of proposed modifications *before* evaluation, is itself
  editable, and improves from accumulated modification-outcome pairs; "forecast accuracy
  improves as the IDM evolves," with up to 46% average performance gains over generations.
  **Prediction-of-own-interventions as a tracked, improving signal is therefore already
  published.** What remains for Horizon: the prediction and its error live in a *substrate*
  (mechanically computed, never narrated, tied to dependency-invalidatable context) rather than
  an editable neural forecaster; calibration is measured on **common mandatory probes** with
  explicit no-change/abstention controls so restraint cannot masquerade as calibration; and
  prediction error is coupled to applicability (a prediction valid for one dependency frame
  must not survive a model/benchmark transfer silently).
- **Outcome-grounded confidence calibration** (arXiv:2610.18614) `[abstract]`, **"Know Thyself"
  self-prediction studies** `[abstract]` — agents' self-prediction is measurably poorly
  calibrated (Brier worse than base-rate baselines in agentic SWE settings), which supports
  Horizon's motivation: uncalibrated self-assessment is a real, measured failure mode.
- **Calibration Is Not Control** (arXiv:2606.21399) `[abstract]` — recalibrating a scalar
  confidence improves prediction metrics without improving control outcomes. Supports treating
  calibration as an *epistemic* endpoint distinct from intervention-selection quality — exactly
  the separation in Horizon's SIM-01 endpoint design.

## 6. Replay-based evaluation of memory

- **LSREP** (arXiv:2609.16730) `[abstract]` — a longitudinal state-replay protocol for evaluating
  conversational memory: ordered history, checkpoint boundaries, probes whose reference answers
  are derived **only from the history prefix available at that checkpoint**, versioned evidence
  of what ran. This is a direct methodological precedent for HZN-SIM-00.5's design
  (availability-restricted retrieval scored against prefix-only ground truth). The preflight's
  evaluation method is an instance of this family; the distinct part is the object (a curated
  developmental-intervention corpus with structural retrieval and gold landmarks) rather than
  conversational QA.
- **CounterMem** (arXiv:2609.31874) `[abstract]` — world-model-verified counterfactual memory:
  records carry applicability conditions checked against the new task before reuse, plus reuse
  statistics. A precedent for scoped applicability gating at retrieval — but the gate is
  task-condition matching, not invalidation on dependency-version change.
- **PCCC-RL** (arXiv:2606.05263), **ReplayLens** (arXiv:2609.34177) `[abstract]` — counterfactual
  replay for credit assignment and auditing memory writers' use of outcomes. Adjacent
  methodology, different target.

## 7. Scientific knowledge graphs with typed epistemic status

- **Coscientist-style dialectical graphs** (junhoyeo/coscientist; the Boiko et al. Coscientist
  line) `[abstract/repo level]` — evidence spans separated from claims, typed argumentative
  relations (supports/undercuts/fails_to_replicate), claim verdicts with provenance and
  status classes (supported/contradicted/unsupported/unchecked, with explicit "never read
  unchecked as evidence of absence"). Closest relatives for typed epistemic status and
  provenance-queryable verification. Domain is literature synthesis; no versioned external
  dependencies, no developmental lineage, no rollback semantics.

---

## What remains defensible as Horizon's contribution

**Claims that must be abandoned (already published or long-established):**

1. Keeping failed/rejected interventions as negative evidence — RRSI, Google SIM.
2. Problem/hypothesis-grouped rather than flat event memory — Google SIM, case-based reasoning
   broadly.
3. Recording predictions of intervention outcomes and tracking their accuracy — IDM
   (2610.13991) does this and shows it improving.
4. Belief bases with supersession chains that never delete — BELIEFMEM.
5. Mechanical dependency-directed retraction as a mechanism — Doyle's TMS, 1979.
6. Prefix-only replay evaluation methodology — LSREP.
7. Applicability conditions attached to memory records — CounterMem.

**Claims that remain defensible, as a synthesis with a specific domain:**

1. **Mechanical invalidation on versioned external dependencies** (model checkpoint, benchmark,
   harness, ontology) for a developmental record — TMS semantics applied to *transferred
   experimental context*, where the dependency graph is over the artifacts of self-improvement
   rather than over inference steps. RRSI/SIM have no equivalent; BELIEFMEM revises by neural
   resolver; no surveyed system invalidates conclusions when the model or benchmark they were
   established on is replaced.
2. **Conditional refutation with machine-checkable reconsideration conditions** — replacing the
   flat accepted/rejected flag (RRSI's `a_i=0` conflation is a documented live weakness) with
   failure conditions and structured reconsideration triggers, so rejected branches reopen when
   their recorded conditions change rather than being permanently shadow-banned.
3. **Interpretations as invalidatable derived states** — the third-order element: historical
   conclusions go stale when their evidential basis is revised, recursively, mechanically.
   BELIEFMEM has supersession but the decision to revise is neural; TMS has the mechanism but
   was never applied to a developmental experimental record.
4. **Availability-tick chronological replay over a real intervention corpus** as memory
   evaluation (LSREP's protocol, but the corpus is a system's own developmental history with
   curated gold landmarks, and the retrieval under test is structural, not embedding-based).
5. **Prediction error computed mechanically in the substrate** — errors as differences between
   predicted and observed values on the prediction's own metric, dependency-scoped, with
   calibration measured on mandatory probes including no-change/abstention controls. IDM
   predicts; Horizon additionally grounds the prediction bookkeeping mechanically and couples it
   to applicability invalidation.

**The honest one-sentence position:** HZN-SIM is not the first system to remember its failures,
record its predictions, revise its beliefs, or replay its history — it is the first application of
TMS-grade mechanical dependency semantics to a self-improvement system's own developmental
record, evaluated by replay against that record and by calibration endpoints that separate
epistemic from model improvement. Every clause in that sentence is load-bearing; drop the
mechanical-dependency clause and the claim collapses into published work.

## Implications for SIM-01

- The IDM result (forecast accuracy improving across generations, large downstream gains)
  independently supports prediction-error feedback as the primary epistemic endpoint — and
  means SIM-01's contribution must be the *mechanical grounding and anti-gaming controls*, not
  the endpoint itself.
- RRSI's documented `a_i=0` ambiguity and the surveys' "self-reinforcing error" warning are
  direct evidence that premature closure and flat rejection records are live failure modes in
  published self-improvement loops — exactly what SIM-00's reconsideration machinery addresses.
- The preflight (SIM-00.5) follows LSREP's methodological discipline; cite it as the evaluation
  family rather than claiming the protocol itself.
- Before any writeup, the Causal Improvement Graph reference must be read directly, and
  BELIEFMEM and IDM deserve full-paper reads rather than abstract level.
