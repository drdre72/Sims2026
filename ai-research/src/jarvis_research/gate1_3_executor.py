"""HZN-IR-10 D2 — execution-level address validation.

The gate is the ONLY path to execution: raw model-generated ids never
bypass it. An address is executable iff it is well-formed, bound to the
store's exact catalog fingerprint and epoch, and resolves to exactly
one canonical row in the COMPLETE catalog. Everything else is a typed
refusal — there is no silent fallback to arbitrary candidates.

Verdict taxonomy:
  executable          unique resolution, correctly bound
  refused_malformed   unparseable / wrong type / missing fields
  refused_unbound     no catalog/epoch binding presented
  refused_mismatched  binding names a different catalog fingerprint
  refused_stale       correct catalog, wrong observation epoch
  refused_missing     well-formed, correctly bound, but absent
  refused_ambiguous   descriptor resolves to multiple rows

Note what this gate does NOT catch: a valid-but-semantically-incorrect
selection (the model meant row A but validly addressed row B) is
executable by construction — the gate guarantees address integrity,
not selection quality. Selection quality is measured separately in the
runner taxonomy, per directive.
"""

import re

REFUSALS = ("refused_malformed", "refused_unbound", "refused_mismatched",
            "refused_stale", "refused_missing", "refused_ambiguous")


def validate_for_execution(store, address, binding=None):
    """Validate an execution address against a store.

    address: int id, numeric-string id, or {"family":..., "layer":...}
    binding: {"catalog_sha256":..., "epoch":...} the caller asserts it
             is addressing — required, verified before resolution.
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

    # 4. resolvable in the complete catalog?
    if kind == "id":
        if store.get(value) is None:
            return {"status": "refused_missing",
                    "catalog_sha256": store.catalog_sha256,
                    "epoch": store.epoch}
        return {"status": "executable", "param_id": value,
                "catalog_sha256": store.catalog_sha256,
                "epoch": store.epoch}
    res = store.resolve(*value)
    res_status = res["status"]
    if res_status == "unique":
        return {"status": "executable", "param_id": res["param_id"],
                "catalog_sha256": store.catalog_sha256,
                "epoch": store.epoch}
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

    def submit(self, address, binding=None):
        verdict = validate_for_execution(self.store, address, binding)
        self.attempts.append({"address_repr": repr(address)[:80],
                              "verdict": verdict})
        if verdict["status"] == "executable":
            # mock execution: record only, never modify anything
            self.executed.append(verdict["param_id"])
        return verdict
