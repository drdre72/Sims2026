"""HZN-IR-10 D2 — execution-level address validation.

The gate is the ONLY path to execution: raw model-generated ids never
bypass it. An address is executable iff it is well-formed, bound to the
store's exact catalog fingerprint and epoch, and resolves to exactly
one canonical row in the COMPLETE catalog. Everything else is a typed
refusal — there is no silent fallback to arbitrary candidates.

Verdict taxonomy:
  executable          unique resolution, correctly bound, in-view
  refused_malformed   unparseable / wrong type / missing fields
  refused_unbound     no catalog/epoch binding presented
  refused_mismatched  binding names a different catalog fingerprint
  refused_stale       correct catalog, wrong observation epoch
  refused_missing     well-formed, correctly bound, but absent
  refused_ambiguous   descriptor resolves to multiple rows
  refused_outside_view  valid catalog address, but NOT a member of the
                        authorized candidate view it was presented in

HZN-IR-11 D1: catalog validity, candidate membership and semantic
correctness are DISTINCT properties. An id can be a perfectly valid
catalog address yet lie outside the candidate view the model was
authorized to select from — the validator must reject it, otherwise a
model can bypass retrieval scoping by inventing addresses it never saw.
Semantic correctness (the RIGHT answer, not merely AN answer) is not
gate-detectable and is measured separately in runner taxonomies.
"""

import hashlib
import json
import re

REFUSALS = ("refused_malformed", "refused_unbound", "refused_mismatched",
            "refused_stale", "refused_missing", "refused_ambiguous",
            "refused_outside_view")


def candidate_view_sha(member_ids) -> str:
    """Identity of an authorized candidate view — the exact set of
    catalog ids the proposer was shown and may select from."""
    return hashlib.sha256(
        json.dumps(sorted(int(i) for i in member_ids)).encode()
    ).hexdigest()


def validate_for_execution(store, address, binding=None, view=None):
    """Validate an execution address against a store.

    address: int id, numeric-string id, or {"family":..., "layer":...}
    binding: {"catalog_sha256":..., "epoch":...} the caller asserts it
             is addressing — required, verified before resolution.
    view:    optional {"view_sha256":..., "member_ids": [...]} — the
             authorized candidate presentation. When supplied, BOTH
             the hash must match the member list AND the resolved id
             must be a member; otherwise refused_outside_view. The
             member list is the authority — the hash exists to bind
             the presentation identity into the audit record.
    Returns a typed verdict dict; "executable" carries param_id +
    binding. Never raises for model-controllable inputs.
    """
    # 1. well-formed?
    kind, value = None, None
    if isinstance(address, bool):
        pass  # bool is an int subclass — explicitly not an id
    elif isinstance(address, int):
        kind, value = "id", address
    elif isinstance(address, str) and re.fullmatch(r"[+-]?\d+",
                                                   address.strip()):
        kind, value = "id", int(address.strip())
    elif isinstance(address, dict):
        fam, lay = address.get("family"), address.get("layer")
        if isinstance(fam, str) and isinstance(lay, int) \
                and not isinstance(lay, bool):
            kind, value = "descriptor", (fam, lay)
    if kind is None:
        return {"status": "refused_malformed",
                "catalog_sha256": store.catalog_sha256,
                "epoch": store.epoch}

    # 2. bound?
    if not isinstance(binding, dict) or \
            "catalog_sha256" not in binding or "epoch" not in binding:
        return {"status": "refused_unbound",
                "catalog_sha256": store.catalog_sha256,
                "epoch": store.epoch}

    # 3. correctly bound?
    if binding["catalog_sha256"] != store.catalog_sha256:
        return {"status": "refused_mismatched",
                "catalog_sha256": store.catalog_sha256,
                "epoch": store.epoch}
    if binding["epoch"] != store.epoch:
        return {"status": "refused_stale",
                "catalog_sha256": store.catalog_sha256,
                "epoch": store.epoch}

    # 4. candidate-view integrity (checked before resolution so a
    # tampered or malformed view can never authorize anything)
    def view_member(pid):
        if view is None:
            return True
        if not isinstance(view, dict) or \
                not isinstance(view.get("member_ids"), list):
            return False
        if "view_sha256" in view and \
                view["view_sha256"] != candidate_view_sha(
                    view["member_ids"]):
            return False
        return pid in view["member_ids"]

    def executable_or_outside(pid):
        if not view_member(pid):
            return {"status": "refused_outside_view",
                    "param_id": pid,
                    "catalog_sha256": store.catalog_sha256,
                    "epoch": store.epoch}
        return {"status": "executable", "param_id": pid,
                "catalog_sha256": store.catalog_sha256,
                "epoch": store.epoch}

    # 5. resolvable in the complete catalog?
    if kind == "id":
        if store.get(value) is None:
            return {"status": "refused_missing",
                    "catalog_sha256": store.catalog_sha256,
                    "epoch": store.epoch}
        return executable_or_outside(value)
    res = store.resolve(*value)
    res_status = res["status"]
    if res_status == "unique":
        return executable_or_outside(res["param_id"])
    return {"status": f"refused_{res_status}",
            **({"param_ids": res["param_ids"]}
               if res_status == "ambiguous" else {}),
            "catalog_sha256": store.catalog_sha256,
            "epoch": store.epoch}


class MockExecutor:
    """D2 acceptance harness: performs NO parameter modification.
    Records every verdict and every address that reaches the execution
    stage. Acceptance requires zero refused addresses in `executed`.
    """

    def __init__(self, store):
        self.store = store
        self.attempts = []   # every validation verdict
        self.executed = []   # param_ids that reached mock execution

    def submit(self, address, binding=None, view=None):
        verdict = validate_for_execution(self.store, address, binding,
                                         view)
        self.attempts.append({"address_repr": repr(address)[:80],
                              "verdict": verdict})
        if verdict["status"] == "executable":
            # mock execution: record only, never modify anything
            self.executed.append(verdict["param_id"])
        return verdict
