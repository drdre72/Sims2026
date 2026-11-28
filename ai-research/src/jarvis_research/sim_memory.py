"""Self-improvement memory substrate (HZN-SIM-00).

Persistent, structurally grounded developmental memory: immutable measurements,
revisable interpretations, hypothesis-centric records, versioned ontology,
state-lineage DAG cross-linked to an evidence/decision graph, and mechanical
dependency-scoped invalidation. CPU-only substrate; no LLM in any path here.

Design contract: research/hzn_sim_01_design.md plus the user's binding review
disposition (2026-11-28). Append-only everywhere; status is derived, never
destructive; interpretation invalidation is recursive and mechanically testable.
"""
from dataclasses import dataclass, replace
import json

from jarvis_research.cognitive_bus import freeze, thaw, canonical_json, sha256_text

# ---------------------------------------------------------------------------
# Statuses (intrinsic, recorded on interpretations; effective status derived)
# ---------------------------------------------------------------------------
PROPOSED = "proposed"
TESTING = "testing"
SUPPORTED = "supported"
REFUTED_CONDITIONAL = "refuted_conditional"
STALE = "stale"
APPLICABILITY_UNKNOWN = "applicability_unknown"
SUPERSEDED = "superseded"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
STATUSES = (PROPOSED, TESTING, SUPPORTED, REFUTED_CONDITIONAL, STALE,
            APPLICABILITY_UNKNOWN, SUPERSEDED, INSUFFICIENT_EVIDENCE)

# Interpretation statuses that count as a basis in good standing.
GOOD_BASIS = (PROPOSED, TESTING, SUPPORTED)

# Dependency types
DEP_MODEL = "model_checkpoint"
DEP_HARNESS = "harness"
DEP_EVAL = "eval_suite"
DEP_ONTOLOGY = "ontology"
DEP_MEASUREMENT = "measurement"
DEP_INTERPRETATION = "interpretation"
DEP_TYPES = (DEP_MODEL, DEP_HARNESS, DEP_EVAL, DEP_ONTOLOGY, DEP_MEASUREMENT, DEP_INTERPRETATION)

# Eval partitions / leakage classes
PARTITIONS = ("practice", "evolve", "held_out", "sealed", "na")
# Clearance ordering for retrieval contexts; sealed never clears.
CLEARANCE = {"practice": 0, "evolve": 1, "held_out": 2, "sealed": 99, "na": 0}

UNKNOWN_CATEGORY = "unknown"


@dataclass(frozen=True)
class Dependency:
    dep_type: str
    dep_id: str
    version: int

    def as_dict(self):
        return {"dep_type": self.dep_type, "dep_id": self.dep_id, "version": self.version}


# ---------------------------------------------------------------------------
# Ontology: three layers, versioned, categories never deleted
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OntologyCategory:
    name: str
    layer: str  # "primitive" | "domain" | "hypothesis"
    introduced_in: int
    deprecated_in: object  # ontology version int or None
    description: str

    def as_dict(self):
        return {"name": self.name, "layer": self.layer, "introduced_in": self.introduced_in,
                "deprecated_in": self.deprecated_in, "description": self.description}


class Ontology:
    """Versioned classification vocabulary. Revisions create new version
    objects; historical versions are retained immutably. Categories are never
    deleted — deprecation only. `unknown` is mandatory and cannot be deprecated."""

    def __init__(self, version, categories, renames=None, prior=None, revision_note=None):
        self.version = version
        self._categories = dict(categories)
        self.renames = dict(renames or {})  # old_name -> new_name
        self.prior = prior
        self.revision_note = revision_note
        if UNKNOWN_CATEGORY not in self._categories:
            raise ValueError("ontology must contain mandatory 'unknown' category")

    def category(self, name):
        return self._categories.get(name)

    def active(self, name):
        cat = self._categories.get(name)
        return cat is not None and cat.deprecated_in is None

    def names(self, layer=None):
        rows = sorted(self._categories)
        if layer is not None:
            rows = [n for n in rows if self._categories[n].layer == layer]
        return tuple(rows)

    def resolve(self, name):
        """Follow renames to the current equivalent candidate name, or None."""
        seen = set()
        cur = name
        while cur in self.renames and cur not in seen:
            seen.add(cur)
            cur = self.renames[cur]
        return cur if cur in self._categories else None

    def revise(self, add=(), deprecate=(), renames=None, note=None):
        if UNKNOWN_CATEGORY in deprecate:
            raise ValueError("'unknown' category cannot be deprecated")
        cats = dict(self._categories)
        new_version = self.version + 1
        for cat in add:
            if cat.name in cats and cats[cat.name].deprecated_in is None:
                raise ValueError(f"category already active: {cat.name}")
            cats[cat.name] = replace(cat, introduced_in=new_version, deprecated_in=None)
        for name in deprecate:
            if name not in cats:
                raise ValueError(f"cannot deprecate missing category: {name}")
            cats[name] = replace(cats[name], deprecated_in=new_version)
        merged_renames = dict(self.renames)
        merged_renames.update(renames or {})
        return Ontology(new_version, cats, merged_renames, prior=self, revision_note=note)

    def as_dict(self):
        return {"version": self.version,
                "categories": [self._categories[n].as_dict() for n in sorted(self._categories)],
                "renames": dict(self.renames), "revision_note": self.revision_note}


def seed_ontology():
    """v0.1 — gate0_5-derived seed plus epistemic failure classes observed in
    EPI-01..06. Domain layer only; primitive-layer schema lives in this module's
    type system; hypothesis layer is the memory entries themselves."""
    def cat(name, desc):
        return OntologyCategory(name, "domain", 1, None, desc)
    cats = {c.name: c for c in [
        cat(UNKNOWN_CATEGORY, "unexplained observation; mandatory escape hatch"),
        cat("coherent_dense_edit", "directed all-elements perturbation of a tensor"),
        cat("sparse_scalar_edit", "scattered low-count scalar perturbation"),
        cat("input_boundary_fragility", "early-layer / embedding-boundary catastrophic sensitivity"),
        cat("quality_collapse_perplexity", "perplexity explosion without numeric blowup"),
        cat("degenerate_output", "uniform-random or otherwise degenerate generations"),
        cat("safe_null_surface", "provably-safe intervention surface is also low-leverage"),
        cat("self_edit_avoidance", "proposer avoids coherent additive edits under self-editing"),
        cat("transcription_failure", "neural retranscription corrupts mechanical state"),
        cat("display_order_copying", "operand order copied from display rather than structure"),
        cat("output_contract_failure", "format/contract misuse despite correct latent content"),
        cat("invalidation_semantics", "staleness/dependency precedence defect"),
        cat("policy_boundary_confusion", "none/unknown/invalid/insufficient/abstain collapse"),
        cat("evaluator_exploitation", "optimizing the evaluator rather than the objective"),
        cat("intervention_frequency_confound", "outcome correlated with intervention count, not content"),
        cat("structural_binding_success", "structural binding preserves mechanical results"),
    ]}
    return Ontology(1, cats)


# ---------------------------------------------------------------------------
# Immutable measurements
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Measurement:
    measurement_id: str
    kind: str               # e.g. benchmark_accuracy, perplexity, collapse_count
    value: object           # frozen payload; immutable forever
    partition: str          # one of PARTITIONS
    provenance: object      # frozen: source experiment, evidence file, commit, extraction method
    availability_tick: int  # when this became knowable to the system (replay ordering)
    created_tick: int
    confounded: bool = False  # set via confound record; measurement itself never mutates

    def as_dict(self):
        return {"measurement_id": self.measurement_id, "kind": self.kind,
                "value": thaw(self.value), "partition": self.partition,
                "provenance": thaw(self.provenance),
                "availability_tick": self.availability_tick, "created_tick": self.created_tick,
                "confounded": self.confounded}


@dataclass(frozen=True)
class Prediction:
    metric: str
    predicted_value: float
    comparison: str         # ">=" | "<=" | "approx"
    contemporaneous: bool   # recorded before evaluation; never fabricated retrospectively
    source: str             # e.g. preregistration doc + commit


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    target: str
    mechanism: str
    parent_state: str       # lineage node id the intervention was applied to
    prediction: object      # Prediction or None (None for historical entries lacking one)
    decision: object        # accepted | rejected | rolled_back | no_change | None pending
    consequences: tuple     # measurement ids (immutable once recorded)
    prediction_error: object  # float or None — computed mechanically, never narrated
    provenance: object
    availability_tick: int
    created_tick: int

    def as_dict(self):
        return {"attempt_id": self.attempt_id, "target": self.target, "mechanism": self.mechanism,
                "parent_state": self.parent_state,
                "prediction": None if self.prediction is None else {
                    "metric": self.prediction.metric, "predicted_value": self.prediction.predicted_value,
                    "comparison": self.prediction.comparison,
                    "contemporaneous": self.prediction.contemporaneous, "source": self.prediction.source},
                "decision": self.decision, "consequences": list(self.consequences),
                "prediction_error": self.prediction_error,
                "provenance": thaw(self.provenance),
                "availability_tick": self.availability_tick, "created_tick": self.created_tick}


@dataclass(frozen=True)
class Interpretation:
    interpretation_id: str
    hypothesis_id: str
    version: int
    status: str
    statement: str
    rationale: str
    ontology_category: str
    ontology_version: int
    depends_on: tuple       # Dependency refs, version-pinned
    retrospective: bool     # written after the fact rather than contemporaneously
    superseded_by: object   # interpretation id or None
    provenance: object
    availability_tick: int
    created_tick: int

    def as_dict(self):
        return {"interpretation_id": self.interpretation_id, "hypothesis_id": self.hypothesis_id,
                "version": self.version, "status": self.status, "statement": self.statement,
                "rationale": self.rationale, "ontology_category": self.ontology_category,
                "ontology_version": self.ontology_version,
                "depends_on": [d.as_dict() for d in self.depends_on],
                "retrospective": self.retrospective, "superseded_by": self.superseded_by,
                "provenance": thaw(self.provenance),
                "availability_tick": self.availability_tick, "created_tick": self.created_tick}


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    ancestry: str           # lineage node id under which it was first proposed
    ontology_category: str  # pinned classification — revisions preserve provenance
    ontology_version: int
    failure_conditions: tuple
    reconsider_machine: tuple   # structured machine-checkable reconsideration conditions
    reconsider_judgment: tuple  # free-text judgment-requiring conditions (never mechanical)
    attempts: tuple         # append-only tuple of Attempt
    interpretations: tuple  # append-only tuple of interpretation ids, revision order
    provenance: object
    availability_tick: int
    created_tick: int

    def current_interpretation_id(self):
        return self.interpretations[-1] if self.interpretations else None

    def as_dict(self):
        return {"hypothesis_id": self.hypothesis_id, "statement": self.statement,
                "ancestry": self.ancestry, "ontology_category": self.ontology_category,
                "ontology_version": self.ontology_version,
                "failure_conditions": [thaw(x) for x in self.failure_conditions],
                "reconsider_machine": [thaw(x) for x in self.reconsider_machine],
                "reconsider_judgment": list(self.reconsider_judgment),
                "attempts": [a.as_dict() for a in self.attempts],
                "interpretations": list(self.interpretations),
                "provenance": thaw(self.provenance),
                "availability_tick": self.availability_tick, "created_tick": self.created_tick}


# ---------------------------------------------------------------------------
# State lineage DAG (computational) — abstract nodes; content lives elsewhere
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LineageNode:
    node_id: str
    kind: str           # checkpoint | harness
    label: str
    parents: tuple      # (parent_node_id, edge_kind, attempt_id_or_None)
    created_tick: int


class Lineage:
    def __init__(self):
        self._nodes = {}
        self.current = None

    def add_node(self, node_id, kind, label, parents=(), tick=0):
        if node_id in self._nodes:
            raise ValueError(f"lineage node exists: {node_id}")
        for parent_id, _edge, _attempt in parents:
            if parent_id not in self._nodes:
                raise ValueError(f"missing parent node: {parent_id}")
        node = LineageNode(node_id, kind, label, tuple(parents), tick)
        self._nodes[node_id] = node
        if self.current is None:
            self.current = node_id
        return node

    def intervene(self, new_node_id, kind, label, attempt_id, tick):
        return self.add_node(new_node_id, kind, label,
                             parents=((self.current, "intervention", attempt_id),), tick=tick)

    def rollback_to(self, ancestor_id, new_node_id, tick):
        """Rollback creates a NEW node whose content derives from the ancestor,
        with an explicit rollback edge — the branch structure is never rewritten.
        Epistemic memory is untouched by construction (it lives elsewhere)."""
        if ancestor_id not in self._nodes:
            raise KeyError(ancestor_id)
        node = self.add_node(new_node_id, self._nodes[ancestor_id].kind,
                             "rollback-to:" + ancestor_id,
                             parents=((ancestor_id, "rollback", None),), tick=tick)
        self.current = node.node_id
        return node

    def advance(self, node_id):
        if node_id not in self._nodes:
            raise KeyError(node_id)
        self.current = node_id

    def node(self, node_id):
        return self._nodes.get(node_id)

    def nodes(self):
        return tuple(self._nodes[k] for k in sorted(self._nodes))

    def descendants_of(self, node_id):
        out = set()
        changed = True
        while changed:
            changed = False
            for node in self._nodes.values():
                if node.node_id in out or node.node_id == node_id:
                    continue
                if any(p[0] == node_id or p[0] in out for p in node.parents):
                    out.add(node.node_id)
                    changed = True
        return out


# ---------------------------------------------------------------------------
# External dependency registry — current versions of the world memory depends on
# ---------------------------------------------------------------------------
class DependencyRegistry:
    """Current versions plus an append-only event log with ticks, so the
    version of any entity can be reconstructed as of any replay tick.
    Historical effective-status evaluation depends on this: an end-state
    version bump must not mark a conclusion stale at a tick that predates it."""

    def __init__(self):
        self._entities = {}  # (dep_type, dep_id) -> version
        self._events = []   # append-only: dicts with kind/dep_type/dep_id/version/tick

    def _validate_tick(self, tick):
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("registry tick must be a non-negative integer")
        if self._events and tick < self._events[-1]["tick"]:
            raise ValueError("registry events must be appended in tick order")

    def _log(self, kind, dep_type, dep_id, version, tick):
        self._validate_tick(tick)
        self._events.append({"kind": kind, "dep_type": dep_type, "dep_id": dep_id,
                             "version": version, "tick": tick})

    def register(self, dep_type, dep_id, version=1, tick=None):
        if tick is None:
            raise ValueError("registry changes require an explicit availability tick")
        self._log("register", dep_type, dep_id, version, tick)
        self._entities[(dep_type, dep_id)] = version

    def version(self, dep_type, dep_id):
        return self._entities.get((dep_type, dep_id))

    def version_at(self, dep_type, dep_id, at_tick):
        """Version as of the tick — the latest event at or before at_tick.
        None when the entity was not yet registered, or had been substituted
        away, as of that tick."""
        best = None
        for e in self._events:
            if (e["dep_type"] == dep_type and e["dep_id"] == dep_id
                    and e["tick"] <= at_tick
                    and (best is None or e["tick"] >= best["tick"])):
                best = e
        if best is None or best["kind"] == "remove":
            return None
        return best["version"]

    def bump(self, dep_type, dep_id, tick=None):
        if tick is None:
            raise ValueError("registry changes require an explicit availability tick")
        key = (dep_type, dep_id)
        if key not in self._entities:
            raise KeyError(dep_id)
        new_version = self._entities[key] + 1
        self._log("bump", dep_type, dep_id, new_version, tick)
        self._entities[key] = new_version
        return self._entities[key]

    def replace(self, dep_type, old_id, new_id, version=1, tick=None):
        """Entity substitution (different model family, different benchmark):
        the old entity ceases to be current entirely."""
        if tick is None:
            raise ValueError("registry changes require an explicit availability tick")
        self._validate_tick(tick)
        if (dep_type, old_id) in self._entities:
            self._log("remove", dep_type, old_id, None, tick)
            del self._entities[(dep_type, old_id)]
        self._entities[(dep_type, new_id)] = version
        self._log("register", dep_type, new_id, version, tick)

    def events(self):
        return tuple(dict(e) for e in self._events)


# ---------------------------------------------------------------------------
# SimMemory — the substrate
# ---------------------------------------------------------------------------
def _check_acyclic(interpretations, new_hypothesis_id, deps):
    """Rejects interpretation-dependency cycles at hypothesis-chain granularity.
    Interpretations depend on chains, not frozen objects: because a dependency
    pin tracks the chain head, an edge exists between the hypotheses, and cycles
    at that level would make invalidation propagation circular."""
    graph = {}
    for iid, versions in interpretations.items():
        latest = versions[-1]
        owner = latest.hypothesis_id
        edges = graph.setdefault(owner, set())
        for d in latest.depends_on:
            if d.dep_type == DEP_INTERPRETATION and d.dep_id in interpretations:
                target_owner = interpretations[d.dep_id][-1].hypothesis_id
                edges.add(target_owner)
    new_edges = graph.setdefault(new_hypothesis_id, set())
    for d in deps:
        if d.dep_type == DEP_INTERPRETATION and d.dep_id in interpretations:
            new_edges.add(interpretations[d.dep_id][-1].hypothesis_id)
    stack = list(graph[new_hypothesis_id])
    seen = set()
    while stack:
        cur = stack.pop()
        if cur == new_hypothesis_id:
            return False
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(graph.get(cur, ()))
    return True


class SimMemory:
    def __init__(self, ontology=None):
        self.ontology = ontology or seed_ontology()
        self.registry = DependencyRegistry()
        self.lineage = Lineage()
        self._measurements = {}       # id -> Measurement (immutable, plus confound flag copies)
        self._measurement_original = {}  # id -> Measurement as first recorded (tamper evidence)
        self._hypotheses = {}         # id -> Hypothesis (replaced wholesale on append)
        self._hypothesis_history = {} # id -> [Hypothesis, ...] append-only
        self._interpretations = {}    # id -> [Interpretation, ...] append-only versions
        self._confounds = {}          # measurement_id -> [confound record dict]
        self._reconsideration_events = []  # append-only log of eligibility events
        self._tick = 0

    def _next_tick(self):
        self._tick += 1
        return self._tick

    # -- measurements: immutable ------------------------------------------
    def record_measurement(self, measurement_id, kind, value, partition, provenance,
                           availability_tick=None):
        if partition not in PARTITIONS:
            raise ValueError(f"unknown partition: {partition}")
        if measurement_id in self._measurements:
            raise ValueError(f"measurement exists: {measurement_id}")
        if partition == "sealed":
            payload = thaw(value) if not isinstance(value, dict) else dict(value)
            if isinstance(payload, dict) and payload.get("redacted") is not True:
                raise ValueError("sealed measurements must be reference-only with redacted=True")
        tick = self._next_tick()
        m = Measurement(measurement_id, kind, freeze(value), partition, freeze(provenance),
                        availability_tick if availability_tick is not None else tick, tick)
        self._measurements[measurement_id] = m
        self._measurement_original[measurement_id] = m
        return m

    def measurement(self, measurement_id):
        return self._measurements.get(measurement_id)

    def measurement_unmodified(self, measurement_id):
        """Tamper evidence: current record equals the first-recorded payload
        modulo the confounded flag, which is metadata about the measurement,
        never a mutation of its value."""
        cur = self._measurements[measurement_id]
        orig = self._measurement_original[measurement_id]
        return (cur.kind == orig.kind and cur.value == orig.value
                and cur.partition == orig.partition and cur.provenance == orig.provenance
                and cur.availability_tick == orig.availability_tick
                and cur.created_tick == orig.created_tick)

    def register_confound(self, measurement_id, note, provenance, availability_tick=None):
        """A confound does not modify the measurement. It is recorded as new
        evidence; interpretations depending on the measurement inherit
        invalidation through effective-status computation. The tick is the
        confound's availability — replay at earlier ticks must not see it."""
        if measurement_id not in self._measurements:
            raise KeyError(measurement_id)
        record = {"measurement_id": measurement_id, "note": note,
                  "provenance": provenance,
                  "tick": (availability_tick if availability_tick is not None
                           else self._next_tick())}
        self._confounds.setdefault(measurement_id, []).append(record)
        cur = self._measurements[measurement_id]
        self._measurements[measurement_id] = replace(cur, confounded=True)
        return record

    # -- lineage passthrough -----------------------------------------------
    def add_lineage_node(self, *args, **kwargs):
        return self.lineage.add_node(*args, tick=self._next_tick(), **kwargs)

    # -- hypotheses -----------------------------------------------------------
    def propose_hypothesis(self, hypothesis_id, statement, ancestry, ontology_category,
                           failure_conditions=(), reconsider_machine=(), reconsider_judgment=(),
                           provenance=None, availability_tick=None, initial_status=PROPOSED,
                           initial_rationale="", depends_on=(), retrospective=False):
        if hypothesis_id in self._hypotheses:
            raise ValueError(f"hypothesis exists: {hypothesis_id}")
        if self.ontology.category(ontology_category) is None:
            raise ValueError(f"unknown ontology category: {ontology_category}")
        for cond in reconsider_machine:
            if not isinstance(cond, dict) or "kind" not in cond:
                raise ValueError("machine reconsideration conditions must be structured dicts with kind")
        tick = self._next_tick()
        hyp = Hypothesis(hypothesis_id, statement, ancestry, ontology_category,
                         self.ontology.version, tuple(failure_conditions),
                         tuple(freeze(c) for c in reconsider_machine),
                         tuple(reconsider_judgment), (), (), freeze(provenance or {}),
                         availability_tick if availability_tick is not None else tick, tick)
        self._hypotheses[hypothesis_id] = hyp
        self._hypothesis_history[hypothesis_id] = [hyp]
        self.add_interpretation(hypothesis_id, initial_status,
                                statement or "initial proposal",
                                initial_rationale, depends_on=depends_on,
                                retrospective=retrospective, provenance=provenance,
                                availability_tick=availability_tick)
        return self._hypotheses[hypothesis_id]

    def _store_hypothesis(self, hyp):
        self._hypotheses[hyp.hypothesis_id] = hyp
        self._hypothesis_history[hyp.hypothesis_id].append(hyp)

    def hypothesis(self, hypothesis_id):
        return self._hypotheses.get(hypothesis_id)

    def hypothesis_history(self, hypothesis_id):
        return tuple(self._hypothesis_history.get(hypothesis_id, ()))

    def hypotheses(self):
        return tuple(self._hypotheses[k] for k in sorted(self._hypotheses))

    # -- attempts -------------------------------------------------------------
    def record_attempt(self, hypothesis_id, attempt_id, target, mechanism, parent_state,
                       prediction=None, provenance=None, availability_tick=None):
        hyp = self._hypotheses[hypothesis_id]
        if prediction is not None and not prediction.contemporaneous:
            raise ValueError("predictions must be contemporaneous; retrospective prediction fabrication is prohibited")
        tick = self._next_tick()
        att = Attempt(attempt_id, target, mechanism, parent_state, prediction, None, (), None,
                      freeze(provenance or {}),
                      availability_tick if availability_tick is not None else tick, tick)
        new = replace(hyp, attempts=hyp.attempts + (att,))
        self._store_hypothesis(new)
        return att

    def record_outcome(self, hypothesis_id, attempt_id, decision, consequence_ids,
                       availability_tick=None):
        """Attach measured outcomes and compute prediction error mechanically.
        Errors are differences between predicted and observed values on the
        prediction's own metric — never narrated, never omitted."""
        hyp = self._hypotheses[hypothesis_id]
        idx = next((i for i, a in enumerate(hyp.attempts) if a.attempt_id == attempt_id), None)
        if idx is None:
            raise KeyError(attempt_id)
        att = hyp.attempts[idx]
        error = None
        if att.prediction is not None and consequence_ids:
            observed = None
            for mid in consequence_ids:
                m = self._measurements.get(mid)
                if m is not None and m.kind == att.prediction.metric:
                    val = thaw(m.value)
                    observed = val.get("value") if isinstance(val, dict) and "value" in val else val
                    break
            if isinstance(observed, (int, float)) and not isinstance(observed, bool):
                error = abs(float(observed) - att.prediction.predicted_value)
        tick = self._next_tick()
        new_att = replace(att, decision=decision, consequences=tuple(consequence_ids),
                          prediction_error=error)
        attempts = hyp.attempts[:idx] + (new_att,) + hyp.attempts[idx + 1:]
        self._store_hypothesis(replace(hyp, attempts=attempts))
        return new_att

    # -- interpretations --------------------------------------------------------
    def add_interpretation(self, hypothesis_id, status, statement, rationale,
                           ontology_category=None, depends_on=None, retrospective=False,
                           provenance=None, availability_tick=None):
        if status not in STATUSES:
            raise ValueError(f"unknown status: {status}")
        hyp = self._hypotheses[hypothesis_id]
        prior_id = hyp.current_interpretation_id()
        if depends_on is None:
            # A revision re-examines the same evidential basis unless it explicitly
            # redeclares its dependencies — inheriting is the default because silently
            # dropping the dependency frame would rewrite the meaning of the chain.
            depends_on = self._interpretations[prior_id][-1].depends_on if prior_id is not None else ()
        deps = tuple(d if isinstance(d, Dependency) else Dependency(*d) for d in depends_on)
        version = 1
        if prior_id is not None:
            prior = self._interpretations[prior_id][-1]
            version = prior.version + 1
            if prior.superseded_by is not None:
                raise ValueError("interpretation chain already continued elsewhere")
        base_id = hypothesis_id + ":interp"
        tick = self._next_tick()
        iid = base_id + ":" + str(version)
        if not _check_acyclic(self._interpretations, hypothesis_id, deps):
            raise ValueError("interpretation dependency cycle detected")
        category = ontology_category or hyp.ontology_category
        interp = Interpretation(iid, hypothesis_id, version, status, statement, rationale,
                                category, self.ontology.version, deps, retrospective, None,
                                freeze(provenance or {}),
                                availability_tick if availability_tick is not None else tick, tick)
        self._interpretations.setdefault(iid, []).append(interp)
        if prior_id is not None:
            prior_versions = self._interpretations[prior_id]
            superseded = replace(prior_versions[-1], superseded_by=iid)
            self._interpretations[prior_id] = prior_versions[:-1] + [superseded]
        new_hyp = replace(hyp, interpretations=hyp.interpretations + (iid,))
        self._store_hypothesis(new_hyp)
        return interp

    def interpretation(self, interpretation_id):
        versions = self._interpretations.get(interpretation_id)
        return versions[-1] if versions else None

    def interpretation_history(self, interpretation_id):
        return tuple(self._interpretations.get(interpretation_id, ()))

    def confounded_at(self, measurement_id, at_tick):
        """Confound state as of the tick — a confound registered later must
        not invalidate a conclusion at a tick that predates it."""
        return any(c["tick"] <= at_tick for c in self._confounds.get(measurement_id, []))

    # -- effective status: dependency checking precedes intrinsic status -------
    def effective_status(self, interpretation_or_id, _seen=None, at_tick=None):
        """at_tick None evaluates against current state; with at_tick, every
        dependency check runs against the world as of that tick: registry
        versions via the event log, chain heads as of the tick, confound
        registrations as of the tick. Dependency-before-intrinsic precedence
        holds at every tick."""
        interp = (self.interpretation(interpretation_or_id)
                  if isinstance(interpretation_or_id, str) else interpretation_or_id)
        if interp is None:
            raise KeyError("unknown interpretation")
        seen = set(_seen or set())
        if interp.interpretation_id in seen:
            return STALE  # defensive; insertion-time acyclicity should prevent
        seen.add(interp.interpretation_id)
        for dep in interp.depends_on:
            if dep.dep_type == DEP_INTERPRETATION:
                versions = self._interpretations.get(dep.dep_id)
                if versions is None:
                    return STALE
                dep_interp = versions[-1]
                if at_tick is not None and dep_interp.availability_tick > at_tick:
                    continue  # basis did not exist at the tick — cannot have influenced it
                # Pins track the chain head: the dependency was written against the
                # basis chain as it was, so any revision of that chain — however
                # minor — flags dependent conclusions for re-examination.
                owner = self._hypotheses.get(dep_interp.hypothesis_id)
                if owner is None:
                    chain_head_version = 0
                elif at_tick is None:
                    chain_head_version = len(owner.interpretations)
                else:
                    chain_head_version = sum(
                        1 for x in owner.interpretations
                        if self.interpretation(x) is not None
                        and self.interpretation(x).availability_tick <= at_tick)
                if chain_head_version != dep.version:
                    return STALE  # basis was revised — dependent conclusions need re-examination
                dep_status = self.effective_status(dep_interp, seen, at_tick)
                if dep_status in (STALE, APPLICABILITY_UNKNOWN):
                    return STALE
                if dep_status in (REFUTED_CONDITIONAL, INSUFFICIENT_EVIDENCE, SUPERSEDED):
                    return INSUFFICIENT_EVIDENCE
            elif dep.dep_type == DEP_MEASUREMENT:
                m = self._measurements.get(dep.dep_id)
                if m is None:
                    return STALE
                confounded = (self.confounded_at(dep.dep_id, at_tick)
                              if at_tick is not None else m.confounded)
                if confounded:
                    return STALE  # basis evidence confounded — conclusions need re-examination
            else:
                if at_tick is None:
                    current = self.registry.version(dep.dep_type, dep.dep_id)
                else:
                    current = self.registry.version_at(dep.dep_type, dep.dep_id, at_tick)
                if current is None:
                    return APPLICABILITY_UNKNOWN  # entity absent (transfer or unregistered) at this tick
                if current != dep.version:
                    return STALE  # newer version of the same entity exists (as of the tick)
        return interp.status

    def hypothesis_status(self, hypothesis_id):
        hyp = self._hypotheses[hypothesis_id]
        iid = hyp.current_interpretation_id()
        if iid is None:
            return PROPOSED
        return self.effective_status(iid)

    def status_at_tick(self, hypothesis_id, tick):
        """Chain-head status as of a replay tick: interpretation revision
        history is replayed, not just availability, and every dependency
        check runs against the world as of the tick (registry event log,
        chain heads, confound registrations). Returns None when no
        interpretation was available at the tick."""
        head = self.interpretation_head_at_tick(hypothesis_id, tick)
        if head is None:
            return None
        return self.effective_status(head, at_tick=tick)

    def interpretation_head_at_tick(self, hypothesis_id, tick):
        """Latest interpretation available at the tick (None if none was)."""
        head = None
        for iid in self._hypotheses[hypothesis_id].interpretations:
            interp = self.interpretation(iid)
            if interp is not None and interp.availability_tick <= tick:
                head = interp
        return head

    # -- reconsideration -------------------------------------------------------
    def reevaluate(self, event):
        """event: dict with kind. Supported kinds:
          dependency_changed {dep_type, dep_id}
        Scans hypotheses whose current effective status is refuted_conditional
        (or stale/insufficient through dependency invalidation) and records
        reconsideration eligibility when a machine-checkable condition matches.
        Eligibility is recorded as an append-only event — never a silent status
        rewrite."""
        if event.get("kind") != "dependency_changed":
            raise ValueError("unsupported reevaluation event")
        out = []
        for hyp in self.hypotheses():
            iid = hyp.current_interpretation_id()
            if iid is None:
                continue
            status = self.effective_status(iid)
            intrinsic = self.interpretation(iid).status
            # Reconsideration reopens closed questions; it does not re-examine
            # hypotheses in good standing — that way lies over-invalidation.
            if intrinsic in GOOD_BASIS and status in GOOD_BASIS:
                continue
            for cond in hyp.reconsider_machine:
                cond = thaw(cond)
                if (cond.get("kind") == "dependency_changed"
                        and cond.get("dep_type") == event.get("dep_type")
                        and cond.get("dep_id", event.get("dep_id")) == event.get("dep_id")):
                    record = {"hypothesis_id": hyp.hypothesis_id,
                              "trigger": dict(event), "matched_condition": dict(cond),
                              "intrinsic_status": intrinsic, "effective_status": status,
                              "tick": self._next_tick()}
                    self._reconsideration_events.append(record)
                    out.append(record)
                    break
        return out

    def reconsideration_events(self):
        return tuple(self._reconsideration_events)

    # -- ontology revision -------------------------------------------------------
    def revise_ontology(self, effective_tick=None, **kwargs):
        """Ontology revision is itself versioned and dependency-aware: entries
        keep their original classification and pinned ontology version. The
        registry bump makes ontology-dependent interpretations stale until
        re-examined — revision is not silent. The bump is an event with a
        tick: replay at earlier ticks must see the pre-revision ontology."""
        if effective_tick is None:
            raise ValueError("ontology revisions require an explicit availability tick")
        self.registry._validate_tick(effective_tick)
        if self.registry.version(DEP_ONTOLOGY, "sim_ontology") is None:
            raise ValueError("ontology dependency must be registered before revision")
        self.ontology = self.ontology.revise(**kwargs)
        self.registry.bump(DEP_ONTOLOGY, "sim_ontology",
                           tick=effective_tick)
        return self.ontology

    def ontology_at_tick(self, tick):
        """Use only vocabulary revisions already available at the replay tick."""
        version = self.registry.version_at(DEP_ONTOLOGY, "sim_ontology", tick)
        ontology = self.ontology
        if version is None:
            return None
        while ontology is not None and ontology.version > version:
            ontology = ontology.prior
        return ontology if ontology is not None and ontology.version == version else None

    # -- structural retrieval: no LLM anywhere ---------------------------------
    def retrieve(self, target_component=None, ontology_category=None,
                 dep_type=None, dep_id=None, clearance="evolve", at_tick=None,
                 include_measurements=True):
        """Structural candidate retrieval. Similarity produces candidates for
        comparison — never merges hypotheses, never establishes equivalence.
        Results carry the keys they matched so callers can judge."""
        max_clearance = CLEARANCE[clearance]
        ontology = self.ontology if at_tick is None else self.ontology_at_tick(at_tick)
        if (ontology_category is not None and at_tick is not None
                and (ontology is None or (
                    ontology.category(ontology_category) is None
                    and ontology_category not in ontology.renames))):
            return ()

        def visible(tick):
            return at_tick is None or tick <= at_tick

        def measurement_visible(m):
            if not visible(m.availability_tick):
                return False
            if m.partition == "sealed":
                return False  # sealed never enters any retrieval context
            return CLEARANCE[m.partition] <= max_clearance

        out = []
        for hyp in self.hypotheses():
            if not visible(hyp.availability_tick):
                continue
            keys = []
            if ontology_category is not None:
                if hyp.ontology_category == ontology_category:
                    keys.append("ontology_category")
                elif ontology is not None and ontology.resolve(ontology_category) == hyp.ontology_category:
                    keys.append("ontology_category_renamed")
            if target_component is not None:
                if any(a.target == target_component and visible(a.availability_tick)
                       for a in hyp.attempts):
                    keys.append("target_component")
            if dep_type is not None:
                if at_tick is None:
                    iid = hyp.current_interpretation_id()
                    interp = self.interpretation(iid) if iid else None
                else:
                    interp = self.interpretation_head_at_tick(hyp.hypothesis_id, at_tick)
                deps = interp.depends_on if interp is not None else ()
                if any(d.dep_type == dep_type and (dep_id is None or d.dep_id == dep_id) for d in deps):
                    keys.append("dependency_overlap")
            if not keys:
                continue
            measurements = []
            if include_measurements:
                seen_ids = []
                for att in hyp.attempts:
                    if visible(att.availability_tick):
                        for mid in att.consequences:
                            if mid not in seen_ids:
                                seen_ids.append(mid)
                for mid in seen_ids:
                    m = self._measurements.get(mid)
                    if m is not None and measurement_visible(m):
                        measurements.append(mid)
            out.append({"hypothesis_id": hyp.hypothesis_id, "match_keys": tuple(keys),
                        "ontology_category": hyp.ontology_category,
                        "status": (self.status_at_tick(hyp.hypothesis_id, at_tick)
                                   if at_tick is not None
                                   else self.hypothesis_status(hyp.hypothesis_id)),
                        "measurements": tuple(measurements),
                        "candidate_not_equivalent": True})
        return tuple(out)

    # -- snapshot / serialization ------------------------------------------------
    def snapshot(self):
        ontology_history = []
        ontology = self.ontology
        while ontology is not None:
            ontology_history.append(ontology.as_dict())
            ontology = ontology.prior
        return {
            "tick": self._tick,
            "ontology": self.ontology.as_dict(),
            "ontology_history": list(reversed(ontology_history)),
            "n_hypotheses": len(self._hypotheses),
            "n_measurements": len(self._measurements),
            "n_lineage_nodes": len(self.lineage.nodes()),
            "lineage": {
                "current": self.lineage.current,
                "nodes": [
                    {"node_id": node.node_id, "kind": node.kind,
                     "label": node.label, "parents": [list(p) for p in node.parents],
                     "created_tick": node.created_tick}
                    for node in self.lineage.nodes()
                ],
            },
            "hypotheses": [self._hypotheses[k].as_dict() for k in sorted(self._hypotheses)],
            "measurements": [self._measurements[k].as_dict() for k in sorted(self._measurements)],
            "interpretations": {iid: [v.as_dict() for v in versions]
                                for iid, versions in sorted(self._interpretations.items())},
            "confounds": {mid: list(records)
                          for mid, records in sorted(self._confounds.items())},
            "registry_events": [dict(e) for e in self._events_snapshot()],
            "reconsideration_events": list(self._reconsideration_events),
        }

    def _events_snapshot(self):
        return self.registry.events()

    def digest(self):
        return sha256_text(canonical_json(self.snapshot()))
