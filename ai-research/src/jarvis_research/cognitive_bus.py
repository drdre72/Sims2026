"""Minimal typed cognitive substrate prototype.

The bus stores versioned evidence objects, executes deterministic primitives over
eligible typed state, and binds results structurally. It is an experimental
runtime reference, not a security boundary or a general agent framework.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import itertools
import json

VALID = "valid"
UNKNOWN = "unknown"
INVALID = "invalid"
AMBIGUOUS = "ambiguous"
RESOLVED = "resolved"
STALE = "stale"
STATUSES = (VALID, UNKNOWN, INVALID, AMBIGUOUS, RESOLVED, STALE)
EXECUTABLE_INPUT_STATUSES = (VALID, RESOLVED)


class FrozenMap(Mapping):
    def __init__(self, data=()):
        items = data.items() if isinstance(data, Mapping) else data
        self._items = tuple((str(k), freeze(v)) for k, v in items)
        self._data = dict(self._items)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return "FrozenMap(" + repr(self._data) + ")"

    def __eq__(self, other):
        if isinstance(other, FrozenMap):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return False

    def __hash__(self):
        return hash(canonical_json(self))


def freeze(value):
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(x) for x in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported evidence value type: {type(value).__name__}")


def thaw(value):
    if isinstance(value, FrozenMap):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(x) for x in value]
    if isinstance(value, list):
        return [thaw(x) for x in value]
    return value


def canonical_json(value):
    return json.dumps(thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceRef:
    object_id: str
    version: int

    def as_dict(self):
        return {"object_id": self.object_id, "version": self.version}


@dataclass(frozen=True)
class EvidenceObject:
    object_id: str
    kind: str
    value: object
    status: str
    provenance: object
    version: int
    depends_on: tuple
    context_id: str
    created_tick: int

    def as_dict(self):
        return {
            "object_id": self.object_id,
            "kind": self.kind,
            "value": thaw(self.value),
            "status": self.status,
            "provenance": thaw(self.provenance),
            "version": self.version,
            "depends_on": [d.as_dict() for d in self.depends_on],
            "context_id": self.context_id,
            "created_tick": self.created_tick,
        }


def _status_rank(status):
    order = {INVALID: 4, AMBIGUOUS: 3, UNKNOWN: 2, VALID: 1}
    return order.get(status, 0)


def worst_status(statuses):
    statuses = list(statuses)
    if not statuses:
        return VALID
    return max(statuses, key=_status_rank)


def _is_bit(value):
    return isinstance(value, bool) or (isinstance(value, int) and value in (0, 1))


def _bit(value):
    return bool(value)


def validate_binary_row_set(value):
    value = thaw(value)
    if not isinstance(value, dict) or "rows" not in value:
        return UNKNOWN
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        return UNKNOWN
    expected = value.get("expected_rows")
    statuses = []
    if expected is not None and len(rows) != expected:
        statuses.append(UNKNOWN)
    labels = []
    for row in rows:
        if not isinstance(row, dict):
            statuses.append(INVALID)
            continue
        missing = {"label", "h1", "h2"} - set(row)
        if missing:
            statuses.append(UNKNOWN)
            continue
        label = row.get("label")
        if not isinstance(label, str) or not label:
            statuses.append(INVALID)
        else:
            labels.append(label)
        for key in ("h1", "h2"):
            bit = row.get(key)
            if bit is None:
                statuses.append(UNKNOWN)
            elif not _is_bit(bit):
                statuses.append(INVALID)
    if len(labels) != len(set(labels)):
        statuses.append(AMBIGUOUS)
    return worst_status(statuses)


def validate_epistemic_matrix_problem(value):
    value = thaw(value)
    if not isinstance(value, dict) or "consistent" not in value or "queries" not in value:
        return UNKNOWN
    consistent = value.get("consistent")
    queries = value.get("queries")
    if not isinstance(consistent, list) or len(consistent) != 2 or not isinstance(queries, list) or not queries:
        return UNKNOWN
    statuses = []
    if any(not isinstance(x, bool) for x in consistent):
        statuses.append(INVALID)
    labels = []
    for query in queries:
        if not isinstance(query, dict):
            statuses.append(INVALID)
            continue
        missing = {"label", "cost", "h1", "h2"} - set(query)
        if missing:
            statuses.append(UNKNOWN)
            continue
        if not isinstance(query["label"], str) or not query["label"]:
            statuses.append(INVALID)
        else:
            labels.append(query["label"])
        if not isinstance(query["cost"], (int, float)) or query["cost"] < 0:
            statuses.append(INVALID)
        for key in ("h1", "h2"):
            if query.get(key) is None:
                statuses.append(UNKNOWN)
            elif not _is_bit(query.get(key)):
                statuses.append(INVALID)
    if len(labels) != len(set(labels)):
        statuses.append(AMBIGUOUS)
    return worst_status(statuses)


RULE_KINDS = {"conj", "sum", "either", "neg"}


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_rule_epistemic_problem(value):
    value = thaw(value)
    required = {"rules", "observations", "queries"}
    if not isinstance(value, dict) or required - set(value):
        return UNKNOWN
    rules, observations, queries = value["rules"], value["observations"], value["queries"]
    if not isinstance(rules, list) or len(rules) != 2 or not isinstance(observations, list) or not observations or not isinstance(queries, list) or not queries:
        return UNKNOWN
    statuses = []
    for rule in rules:
        if not isinstance(rule, dict) or "kind" not in rule:
            statuses.append(UNKNOWN)
            continue
        if rule["kind"] not in RULE_KINDS:
            statuses.append(INVALID)
            continue
        needed = {"conj": ("a", "b"), "sum": ("a", "c"), "either": ("a", "b"), "neg": ("a", "b")}[rule["kind"]]
        if any(k not in rule for k in needed):
            statuses.append(UNKNOWN)
        elif any(not _numeric(rule[k]) for k in needed):
            statuses.append(INVALID)
    for obs in observations:
        if not isinstance(obs, dict):
            statuses.append(INVALID)
            continue
        if {"x", "y", "outcome"} - set(obs):
            statuses.append(UNKNOWN)
        elif not _numeric(obs["x"]) or not _numeric(obs["y"]) or not isinstance(obs["outcome"], bool):
            statuses.append(INVALID)
    labels = []
    for query in queries:
        if not isinstance(query, dict):
            statuses.append(INVALID)
            continue
        if {"label", "x", "y"} - set(query):
            statuses.append(UNKNOWN)
            continue
        if not isinstance(query["label"], str) or not query["label"]:
            statuses.append(INVALID)
        else:
            labels.append(query["label"])
        if not _numeric(query["x"]) or not _numeric(query["y"]):
            statuses.append(INVALID)
    if len(labels) != len(set(labels)):
        statuses.append(AMBIGUOUS)
    return worst_status(statuses)


KIND_VALIDATORS = {
    "binary_row_set": validate_binary_row_set,
    "epistemic_matrix_problem": validate_epistemic_matrix_problem,
    "rule_epistemic_problem": validate_rule_epistemic_problem,
}


def validate_value(kind, value):
    validator = KIND_VALIDATORS.get(kind)
    return validator(value) if validator else VALID


def stable_object_id(kind, value, context_id="default"):
    return "obj_" + sha256_text(context_id + "|" + kind + "|" + canonical_json(value))[:16]


def derived_object_id(primitive_name, input_refs, context_id="default"):
    key = context_id + "|" + primitive_name + "|" + ",".join(r.object_id for r in input_refs)
    return "drv_" + sha256_text(key)[:16]


class CognitiveBus:
    def __init__(self):
        self._current = {}
        self._history = {}
        self._bindings = {}
        self._tick = 0

    def _next_tick(self):
        self._tick += 1
        return self._tick

    def put(self, kind, value, object_id=None, provenance=None, context_id="default"):
        frozen = freeze(value)
        object_id = object_id or stable_object_id(kind, frozen, context_id)
        if object_id in self._current:
            raise ValueError(f"object already exists: {object_id}")
        status = validate_value(kind, frozen)
        obj = EvidenceObject(
            object_id=object_id,
            kind=kind,
            value=frozen,
            status=status,
            provenance=freeze(provenance or {"source": "unspecified"}),
            version=1,
            depends_on=(),
            context_id=context_id,
            created_tick=self._next_tick(),
        )
        self._current[object_id] = obj
        self._history[object_id] = [obj]
        return obj

    def _replace(self, object_id, value, status, provenance, depends_on):
        old = self._current[object_id]
        obj = EvidenceObject(
            object_id=object_id,
            kind=old.kind,
            value=freeze(value),
            status=status,
            provenance=freeze(provenance),
            version=old.version + 1,
            depends_on=tuple(depends_on),
            context_id=old.context_id,
            created_tick=self._next_tick(),
        )
        self._current[object_id] = obj
        self._history[object_id].append(obj)
        return obj

    def update(self, object_id, value, provenance=None):
        old = self._current[object_id]
        if old.depends_on:
            raise ValueError("derived objects cannot be updated as premises")
        status = validate_value(old.kind, value)
        return self._replace(object_id, value, status, provenance or {"source": "unspecified_update"}, ())

    def current(self, object_id):
        return self._current.get(object_id)

    def history(self, object_id):
        return tuple(self._history.get(object_id, ()))

    def objects(self, kind=None, context_id=None):
        rows = sorted(self._current.values(), key=lambda x: x.created_tick)
        if kind is not None:
            rows = [x for x in rows if x.kind == kind]
        if context_id is not None:
            rows = [x for x in rows if x.context_id == context_id]
        return tuple(rows)

    def effective_status(self, obj, _seen=None):
        seen = set(_seen or set())
        if obj.object_id in seen:
            return AMBIGUOUS
        seen.add(obj.object_id)
        for dep in obj.depends_on:
            cur = self._current.get(dep.object_id)
            if cur is None or cur.version != dep.version:
                return STALE
            dep_status = self.effective_status(cur, seen)
            if dep_status not in EXECUTABLE_INPUT_STATUSES and dep_status != AMBIGUOUS:
                return STALE
            if dep_status == AMBIGUOUS:
                return STALE
        return obj.status

    def executable_input(self, obj):
        return obj.status in EXECUTABLE_INPUT_STATUSES and self.effective_status(obj) == obj.status

    def bind(self, name, object_id):
        if object_id not in self._current:
            raise KeyError(object_id)
        self._bindings[name] = object_id

    def binding(self, name):
        object_id = self._bindings.get(name)
        return self._current.get(object_id) if object_id else None

    def bindings(self):
        return dict(self._bindings)

    def put_derived(self, output_id, kind, value, status, primitive_name, input_refs, context_id):
        provenance = {
            "source": "deterministic_primitive",
            "primitive": primitive_name,
            "inputs": [r.as_dict() for r in input_refs],
        }
        existing = self._current.get(output_id)
        if existing is None:
            frozen = freeze(value)
            obj = EvidenceObject(
                object_id=output_id,
                kind=kind,
                value=frozen,
                status=status,
                provenance=freeze(provenance),
                version=1,
                depends_on=tuple(input_refs),
                context_id=context_id,
                created_tick=self._next_tick(),
            )
            self._current[output_id] = obj
            self._history[output_id] = [obj]
            return obj, False
        return self._replace(output_id, value, status, provenance, input_refs), True

    def snapshot(self):
        objects = []
        for obj in self.objects():
            row = obj.as_dict()
            row["effective_status"] = self.effective_status(obj)
            objects.append(row)
        return {
            "objects": objects,
            "bindings": self.bindings(),
            "n_objects": len(objects),
            "tick": self._tick,
        }


@dataclass(frozen=True)
class Primitive:
    name: str
    input_kinds: tuple
    output_kind: str
    function: object
    binding_name: str
    applicable: object = None

    def execute(self, inputs):
        return self.function(*inputs)

    def applies_to(self, inputs):
        return self.applicable is None or self.applicable(*inputs)


def primitive_diff(rowset_obj):
    rows = thaw(rowset_obj.value)["rows"]
    flags = {}
    for row in rows:
        flags[row["label"]] = _bit(row["h1"]) != _bit(row["h2"])
    return {"flags": flags}, RESOLVED


def primitive_count(flags_obj):
    flags = thaw(flags_obj.value)["flags"]
    return {"count": sum(1 for x in flags.values() if x)}, RESOLVED


def primitive_unique(flags_obj, count_obj):
    flags = thaw(flags_obj.value)["flags"]
    count = thaw(count_obj.value)["count"]
    actual = sum(1 for x in flags.values() if x)
    if count != actual:
        return {"unique": None, "count": count, "differences": [], "reason": "count_flags_mismatch"}, INVALID
    differences = [label for label, diff in flags.items() if diff]
    unique = differences[0] if count == 1 else "none"
    return {"unique": unique, "count": count, "differences": differences}, RESOLVED


def eval_rule(rule, x, y):
    if rule["kind"] == "conj":
        return x >= rule["a"] and y >= rule["b"]
    if rule["kind"] == "sum":
        return x >= rule["a"] and x + y >= rule["c"]
    if rule["kind"] == "either":
        return x >= rule["a"] or y >= rule["b"]
    if rule["kind"] == "neg":
        return x >= rule["a"] and y <= rule["b"]
    raise ValueError(rule["kind"])


def primitive_consistent_rules(problem_obj):
    problem = thaw(problem_obj.value)
    fit = []
    for rule in problem["rules"]:
        fit.append(all(eval_rule(rule, o["x"], o["y"]) == o["outcome"] for o in problem["observations"]))
    return {"fit": fit}, RESOLVED


def _discrimination_value(choice, candidates, reason, costs=None):
    return {
        "choice": choice,
        "candidates": list(candidates),
        "costs": dict(costs or {}),
        "reason": reason,
    }


def primitive_discriminate_rules(problem_obj, consistency_obj):
    problem = thaw(problem_obj.value)
    fit = thaw(consistency_obj.value)["fit"]
    if sum(fit) != 2:
        return _discrimination_value("no_query", [], "already_resolved"), RESOLVED
    diffs = []
    for query in problem["queries"]:
        predictions = [eval_rule(rule, query["x"], query["y"]) for rule in problem["rules"]]
        if predictions[0] != predictions[1]:
            diffs.append(query["label"])
    if not diffs:
        return _discrimination_value("no_query", [], "no_discriminating_query"), RESOLVED
    if len(diffs) > 1:
        return _discrimination_value(None, diffs, "equal_cost_tie"), AMBIGUOUS
    return _discrimination_value(diffs[0], diffs, "unique_discriminator"), RESOLVED


def primitive_discriminate_matrix(problem_obj):
    problem = thaw(problem_obj.value)
    consistent = problem["consistent"]
    if sum(consistent) != 2:
        return _discrimination_value("no_query", [], "already_resolved"), RESOLVED
    diffs = [q for q in problem["queries"] if _bit(q["h1"]) != _bit(q["h2"])]
    if not diffs:
        return _discrimination_value("no_query", [], "no_discriminating_query"), RESOLVED
    cheapest = min(q["cost"] for q in diffs)
    candidates = [q["label"] for q in diffs if q["cost"] == cheapest]
    costs = {q["label"]: q["cost"] for q in diffs}
    if len(candidates) > 1:
        return _discrimination_value(None, candidates, "equal_cost_tie", costs), AMBIGUOUS
    return _discrimination_value(candidates[0], candidates, "lowest_cost_discriminator", costs), RESOLVED


def derived_from(parent_obj, derived_obj):
    return any(dep.object_id == parent_obj.object_id for dep in derived_obj.depends_on)


def unique_inputs_compatible(flags_obj, count_obj):
    return derived_from(flags_obj, count_obj)


def rule_discrimination_inputs_compatible(problem_obj, consistency_obj):
    return derived_from(problem_obj, consistency_obj)


def scoped_binding_name(name, context_id):
    return name if context_id == "default" else context_id + ":" + name


PRIMITIVES = (
    Primitive("DIFF", ("binary_row_set",), "comparison_flags", primitive_diff, "comparison.flags"),
    Primitive("COUNT", ("comparison_flags",), "cardinality", primitive_count, "comparison.count"),
    Primitive("UNIQUE", ("comparison_flags", "cardinality"), "unique_selection", primitive_unique, "decision.unique", unique_inputs_compatible),
    Primitive("CONSISTENT", ("rule_epistemic_problem",), "hypothesis_consistency", primitive_consistent_rules, "epistemic.consistency"),
    Primitive("DISCRIMINATE_RULES", ("rule_epistemic_problem", "hypothesis_consistency"), "query_decision", primitive_discriminate_rules, "epistemic.choice", rule_discrimination_inputs_compatible),
    Primitive("DISCRIMINATE_MATRIX", ("epistemic_matrix_problem",), "query_decision", primitive_discriminate_matrix, "epistemic.choice"),
)


def auto_dispatch(bus, primitives=PRIMITIVES, max_rounds=20):
    events = []
    for round_index in range(max_rounds):
        progressed = False
        for primitive in primitives:
            groups = []
            for kind in primitive.input_kinds:
                groups.append([obj for obj in bus.objects(kind=kind) if bus.executable_input(obj)])
            if any(not group for group in groups):
                continue
            for combo in itertools.product(*groups):
                contexts = {obj.context_id for obj in combo}
                if len(contexts) != 1:
                    continue
                context_id = combo[0].context_id
                if not primitive.applies_to(combo):
                    continue
                refs = tuple(EvidenceRef(obj.object_id, obj.version) for obj in combo)
                output_id = derived_object_id(primitive.name, refs, context_id)
                existing = bus.current(output_id)
                if existing is not None and existing.depends_on == refs:
                    continue
                value, status = primitive.execute(combo)
                obj, recomputed = bus.put_derived(output_id, primitive.output_kind, value, status, primitive.name, refs, context_id)
                scoped_binding = scoped_binding_name(primitive.binding_name, context_id)
                bus.bind(scoped_binding, output_id)
                if primitive.binding_name not in bus.bindings():
                    bus.bind(primitive.binding_name, output_id)
                events.append({
                    "primitive": primitive.name,
                    "round": round_index,
                    "inputs": [r.as_dict() for r in refs],
                    "output_id": obj.object_id,
                    "output_version": obj.version,
                    "output_status": obj.status,
                    "recomputed": recomputed,
                    "binding": scoped_binding,
                })
                progressed = True
        if not progressed:
            break
    return events


@dataclass(frozen=True)
class UpdateProposal:
    object_id: str
    value: object
    provenance: object
    valid: bool
    errors: tuple


def validate_update_proposal(bus, proposal):
    errors = list(proposal.errors)
    obj = bus.current(proposal.object_id)
    if obj is None:
        errors.append("missing_object")
    elif obj.depends_on:
        errors.append("derived_object_not_a_premise")
    provenance = thaw(proposal.provenance)
    if not provenance.get("proposer"):
        errors.append("missing_proposer")
    if not provenance.get("rationale"):
        errors.append("missing_rationale")
    try:
        validate_value(obj.kind, proposal.value) if obj is not None else None
    except Exception as exc:
        errors.append("unvalidatable_value:" + type(exc).__name__)
    return tuple(dict.fromkeys(errors))


def propose_premise_update(bus, object_id, value, provenance):
    provenance = dict(provenance or {})
    try:
        frozen_value = freeze(value)
        freeze_errors = ()
    except TypeError as exc:
        frozen_value = None
        freeze_errors = (str(exc),)
    proposal = UpdateProposal(object_id, frozen_value, freeze(provenance), False, freeze_errors)
    errors = validate_update_proposal(bus, proposal)
    return UpdateProposal(object_id, frozen_value, freeze(provenance), not errors, errors)


def apply_update_proposal(bus, proposal):
    errors = validate_update_proposal(bus, proposal)
    if errors or not proposal.valid:
        raise ValueError("invalid update proposal: " + ",".join(errors))
    return bus.update(proposal.object_id, thaw(proposal.value), thaw(proposal.provenance))
