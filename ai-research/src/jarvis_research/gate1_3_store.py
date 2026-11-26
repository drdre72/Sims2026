"""HZN-IR-09 §3 — deterministic, read-only addressable observation store.

The store is STRICTLY mechanical: exact filtering, sorting, retrieval
and identity resolution over canonical candidate records. It contains
no intelligent intervention policy, no learned retriever, no embedding
model, no LLM — Python dicts and exact predicates only, per directive.

Design contract:
- read-only: built once from the canonical catalog; no mutation API
- exact: all predicates are equality / interval tests on stored fields
- total functions: every query returns a typed result
  (unique | ambiguous | missing | list), never an exception for
  well-formed queries and never a guessed answer
- auditable: resolves to stored canonical records whose provenance is
  carried through from the observation layer
"""

import hashlib
import json


def catalog_sha256(records) -> str:
    """Content identity of a stored catalog (for catalog-independence
    verification in comprehension coverage, IR-09 F5)."""
    canon = [{k: r.get(k) for k in (
        "param_id", "layer", "family", "numel", "mag_min", "mag_max",
        "count_max", "obs_available", "ar", "lc")} for r in records]
    canon.sort(key=lambda r: r["param_id"])
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True).encode()).hexdigest()


class ObservationStore:
    """Read-only canonical catalog with exact mechanical retrieval.

    HZN-IR-10 D1 hardening:
    - duplicate param_ids are rejected at construction (an address
      space with collisions is not an address space)
    - every accessor returns COPIES — callers can never mutate the
      stored canonical records
    - the store carries its catalog fingerprint and observation epoch;
      resolution verdicts are bound to both, so an address resolved
      against one catalog/epoch can never be silently replayed against
      another
    - resolve() always evaluates against the COMPLETE catalog, never a
      truncated retrieval view — retrieval narrows presentation, never
      identity
    """

    def __init__(self, canonical_records, epoch=None, provenance=None):
        self._rows = [dict(r) for r in canonical_records]
        ids = [r["param_id"] for r in self._rows]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate param_ids rejected: {dupes[:5]}")
        self._by_id = {r["param_id"]: r for r in self._rows}
        self._by_pair = {}
        self._by_family = {}
        for r in self._rows:
            self._by_pair.setdefault((r["family"], r["layer"]), []).append(r)
            self._by_family.setdefault(r["family"], []).append(r)
        self.catalog_sha256 = catalog_sha256(self._rows)
        self.epoch = epoch
        self.provenance = dict(provenance) if provenance else None

    def __len__(self):
        return len(self._rows)

    def _binding(self):
        return {"catalog_sha256": self.catalog_sha256, "epoch": self.epoch}

    # ---- retrieval -------------------------------------------------
    def get(self, param_id):
        """Exact id lookup; COPY or None when absent (never fabricated,
        never a mutable reference)."""
        r = self._by_id.get(param_id)
        return dict(r) if r is not None else None

    def find(self, family=None, layer=None, obs_available=None,
             min_abs_lc=None, max_abs_lc=None,
             min_ar=None, max_ar=None, limit=None):
        """Exact conjunctive filter; sorted by param_id; copies."""
        out = []
        for r in self._rows:
            if family is not None and r["family"] != family:
                continue
            if layer is not None and r["layer"] != layer:
                continue
            if obs_available is not None and r["obs_available"] != obs_available:
                continue
            if min_abs_lc is not None and (
                    r["lc"] is None or abs(r["lc"]) < min_abs_lc):
                continue
            if max_abs_lc is not None and (
                    r["lc"] is None or abs(r["lc"]) > max_abs_lc):
                continue
            if min_ar is not None and (r["ar"] is None or r["ar"] < min_ar):
                continue
            if max_ar is not None and (r["ar"] is None or r["ar"] > max_ar):
                continue
            out.append(dict(r))
        out.sort(key=lambda r: r["param_id"])
        return out[:limit] if limit else out

    def resolve(self, family, layer):
        """Identity resolution for a (family, layer) descriptor over the
        COMPLETE catalog.

        Returns a typed verdict bound to the catalog fingerprint and
        epoch — the store never picks an arbitrary member of an
        ambiguous set and never invents a missing row:
          {"status": "unique",    "param_id": int,     + binding}
          {"status": "ambiguous", "param_ids": [...],  + binding}
          {"status": "missing",                        + binding}
        """
        hits = self._by_pair.get((family, layer), [])
        if not hits:
            return {"status": "missing", **self._binding()}
        if len(hits) > 1:
            return {"status": "ambiguous",
                    "param_ids": sorted(h["param_id"] for h in hits),
                    **self._binding()}
        return {"status": "unique", "param_id": hits[0]["param_id"],
                **self._binding()}

    def top_abs_lc(self, k, obs_only=True):
        """The k rows with largest |lc| — exact sort, ties broken by
        param_id for determinism; copies."""
        rows = [r for r in self._rows
                if (r["obs_available"] and r["lc"] is not None)
                or not obs_only]
        rows.sort(key=lambda r: (-abs(r["lc"] or 0.0), r["param_id"]))
        return [dict(r) for r in rows[:k]]

    # ---- rendering --------------------------------------------------
    def render_subset(self, records, fam_code, uniform_mag=True,
                      obs=True, level="linguistic"):
        """Render a retrieved subset as catalog text in a channel level —
        same surfaces the proposer would see, just fewer rows."""
        from jarvis_research.gate1_3_channel import encode_catalog_text
        entries = [self._as_entry(r) for r in records]
        return encode_catalog_text(entries, fam_code, uniform_mag, obs,
                                   level)

    @staticmethod
    def _as_entry(r):
        return {"param_id": r["param_id"], "layer": r["layer"],
                "family": r["family"], "numel": r["numel"],
                "mag_min": r["mag_min"], "mag_max": r["mag_max"],
                "count_max": r["count_max"],
                "obs_available": r["obs_available"],
                "obs_act_rank": r["ar"],
                "obs_log_contrast": r["lc"]}
