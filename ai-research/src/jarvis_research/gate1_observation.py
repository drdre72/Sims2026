"""Gate 1.2 observation layer (HZN-IR-03 R1/R2/R4).

Forward-only, practice-derived activation statistics attached to catalog
rows. Everything here is correlational description, not causal evidence.
Statistics are computed ONCE on the unmodified checkpoint and held fixed
for all candidates/conditions; the block is labeled with the checkpoint
revision it was measured on.

Key rules from the REV 1 spec:
- Module mapping is resolved by walking the module tree to each
  parameter's direct owning nn.Module, NEVER by name-prefix matching.
- Parameters sharing a hooked parent module share an obs_group; identical
  values across a group are one measurement, not independent evidence.
- Unexecuted modules are marked unavailable, never written as zero.
- The contrast ratio is eps-stabilized; unstable rows are flagged.
- The shuffled placebo permutes WHOLE observation vectors at obs_group
  level within (family, layer-band) strata, preserving the joint feature
  relationship; every row sharing an obs_group receives the same
  reassigned vector, the permutation is returned for audit, and
  effective coverage is measured on the exact features the proposer
  sees (HZN-IR-04 R2).
- The conventional evidence interpreter (B1) is a FROZEN prespecified
  rule; it is deliberately direction-agnostic and consumes the SAME
  proposer-visible field shown to the model (HZN-IR-04 R1).
"""

import hashlib
import json
import math
import random
from typing import Dict, List, Optional

import torch

OBSERVATION_VERSION = "obs-v2"
# Stabilization: eps = this fraction of the MEDIAN positive group RMS.
# Frozen; changing it is a protocol change.
EPS_FRACTION = 0.01
# Layer bands for shuffle strata (layers grouped into bands of this size
# so a stratum usually contains >1 observation group).
SHUFFLE_LAYER_BAND = 8
# Precision of the proposer-visible signed log-contrast column (lc).
# The conventional interpreter consumes THIS rounded value, never the
# raw statistic, so the evidence channel is bit-identical across
# conditions (HZN-IR-04 R1).
LOG_CONTRAST_DECIMALS = 2

# Frozen conventional interpreter (HZN-IR-03 R1, parity-corrected per
# HZN-IR-04 R1). interp-v2 weights each catalog row by the STRENGTH of
# its proposer-visible contrast: s_i = minmax(|obs_log_contrast_i|)
# within family, sampling weight w_i = FLOOR + s_i — the identical
# 2-decimal lc value rendered in the model's catalog rows. The floor
# keeps every row reachable; min-max keeps the rule scale-free. The
# interpreter has ACCESS to every obs_* field on the row (same rows the
# proposer's catalog is built from); the frozen rule uses only the
# |lc| channel. This rule is prespecified and must not be tuned on
# outcomes — any change bumps INTERPRETER_VERSION.
INTERPRETER_VERSION = "interp-v2"
INTERPRETER_FLOOR = 0.1


def resolve_hook_targets(model, catalog: List[dict]) -> Dict[int, str]:
    """Map each catalog param_id to the path of its DIRECT owning module.

    Walks the module tree rather than matching name prefixes: strip the
    trailing attribute component from the parameter name, then verify the
    module actually owns that parameter locally (named_parameters with
    recurse=False). Raises on any unresolved parameter — hook coverage
    must be complete before execution (HZN-IR-03 R4).
    """
    modules = dict(model.named_modules())
    unresolved = []
    targets: Dict[int, str] = {}
    for entry in catalog:
        path = entry["param"].rsplit(".", 1)[0]
        mod = modules.get(path)
        local = (dict(mod.named_parameters(recurse=False))
                 if mod is not None else {})
        leaf = entry["param"].rsplit(".", 1)[1]
        if mod is None or leaf not in local:
            unresolved.append(entry["param"])
        else:
            targets[entry["param_id"]] = path
    if unresolved:
        raise ValueError(
            f"hook coverage incomplete: {len(unresolved)} catalog params "
            f"have no resolvable owning module: {unresolved[:5]}")
    return targets


class _ModuleStats:
    """Scalar-only hook accumulator: running sums, no retained tensors."""
    __slots__ = ("n_hi", "n_lo", "rms_hi", "rms_lo", "max_abs",
                 "per_passage_rms", "hi_rms_list", "lo_rms_list")

    def __init__(self):
        self.n_hi = 0
        self.n_lo = 0
        self.rms_hi = 0.0
        self.rms_lo = 0.0
        self.max_abs = 0.0
        self.per_passage_rms = []
        self.hi_rms_list = []
        self.lo_rms_list = []

    def add(self, output: torch.Tensor, hi: bool):
        x = output[0] if isinstance(output, tuple) else output
        if not isinstance(x, torch.Tensor) or x.numel() == 0:
            return
        f = x.detach().float()
        rms = float(f.pow(2).mean().sqrt().item())
        mx = float(f.abs().max().item())
        if hi:
            self.n_hi += 1
            self.rms_hi += rms
            self.hi_rms_list.append(rms)
        else:
            self.n_lo += 1
            self.rms_lo += rms
            self.lo_rms_list.append(rms)
        self.max_abs = max(self.max_abs, mx)
        self.per_passage_rms.append(rms)


@torch.no_grad()
def collect_observations(model, tok, practice_texts: Dict[str, str],
                         targets: Dict[int, str], checkpoint_revision: str,
                         passage_losses: Optional[Dict[str, float]] = None
                         ) -> dict:
    """Measure per-module activation on stratified hi/lo practice splits.

    `targets` maps catalog keys to owning-module paths (the union across
    all arms, from resolve_hook_targets). `passage_losses` (mean CE per
    passage name) may be supplied if already computed; otherwise it is
    measured here with one extra forward pass. Split is WITHIN domain
    (domain = key prefix before first "_"): the higher-loss half of each
    domain's passages is the hi set, so subject matter cannot drive the
    contrast.
    """
    if passage_losses is None:
        passage_losses = {}
        for name, text in practice_texts.items():
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            if ids.shape[1] < 2:
                continue
            logits = model(ids).logits.float()
            sl, lab = logits[:, :-1, :], ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.size(-1)), lab.reshape(-1))
            passage_losses[name] = float(nll.item())
    hi_set, lo_set = stratified_split(practice_texts, passage_losses)

    groups: Dict[str, _ModuleStats] = {}
    handles = []
    for path in set(targets.values()):
        stats = groups.setdefault(path, _ModuleStats())
        mod = model.get_submodule(path)
        which = {"hi": False}

        def _hook(_m, _in, out, s=stats, w=which):
            s.add(out, w["hi"])

        handles.append((mod.register_forward_hook(_hook), which, stats))

    try:
        # One forward per practice passage; classify into hi/lo via the
        # shared flag and accumulate.
        for name, text in practice_texts.items():
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            if ids.shape[1] < 2:
                continue
            is_hi = name in hi_set
            for _, which, _ in handles:
                which["hi"] = is_hi
            model(ids)
    finally:
        for h, _, _ in handles:
            h.remove()

    # Stabilization eps from the distribution of positive group RMS.
    overall = [(g.rms_hi + g.rms_lo) / max(g.n_hi + g.n_lo, 1)
               for g in groups.values() if g.n_hi + g.n_lo > 0]
    pos = [v for v in overall if v > 0]
    eps = EPS_FRACTION * (sorted(pos)[len(pos) // 2] if pos else 0.0)
    eps = max(eps, 1e-8)

    group_stats = {}
    for path, g in groups.items():
        n = g.n_hi + g.n_lo
        if n == 0:
            group_stats[path] = {"available": False}
            continue
        act = (g.rms_hi + g.rms_lo) / n
        hi_m = g.rms_hi / g.n_hi if g.n_hi else 0.0
        lo_m = g.rms_lo / g.n_lo if g.n_lo else 0.0
        mean = act
        var = (sum((r - mean) ** 2 for r in g.per_passage_rms) / n) if n else 0.0
        # Separate high/low sample counts and variance (HZN-IR-04 §5):
        # a pooled count/variance can hide a one-sided measurement.
        var_hi = (sum((r - hi_m) ** 2 for r in g.hi_rms_list) / g.n_hi
                  if g.n_hi else 0.0)
        var_lo = (sum((r - lo_m) ** 2 for r in g.lo_rms_list) / g.n_lo
                  if g.n_lo else 0.0)
        group_stats[path] = {
            "available": True,
            "act": act, "hi": hi_m, "lo": lo_m,
            "hilo": (hi_m + eps) / (lo_m + eps),
            "max_abs": g.max_abs,
            "variance": var, "n_samples": n,
            "n_hi": g.n_hi, "n_lo": g.n_lo,
            "var_hi": var_hi, "var_lo": var_lo,
            "hilo_unstable": lo_m <= eps,
        }

    return {
        "version": OBSERVATION_VERSION,
        "checkpoint_sha256": checkpoint_revision,
        "eps": eps,
        "hi_passages": sorted(hi_set),
        "lo_passages": sorted(lo_set),
        "passage_losses": {k: round(v, 6)
                           for k, v in passage_losses.items()},
        "groups": group_stats,
    }


def stratified_split(texts: Dict[str, str], losses: Dict[str, float]):
    """Within-domain median split: hi = higher-loss half per domain."""
    domains: Dict[str, List[str]] = {}
    for name in texts:
        if name in losses:
            domains.setdefault(name.split("_")[0], []).append(name)
    hi, lo = set(), set()
    for names in domains.values():
        names.sort(key=lambda n: losses[n])
        half = len(names) // 2
        lo.update(names[:half])
        hi.update(names[half:])
        if len(names) % 2:  # odd: middle goes to hi (higher-median side)
            pass
    return hi, lo


def _family_ranks(rows: List[dict], key: str):
    """In-place tie-aware [0,1] midranks of `key` within each family.

    Equal values receive equal ranks (the mean of the positions the tie
    group occupies, normalized to [0,1]); a single member gets 0.5.
    Breaking ties by param_id would inject artificial distinctions into
    evidence the proposer sees (HZN-IR-04 R2).
    """
    fams: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("obs_available"):
            fams.setdefault(r["family"], []).append(r)
    for members in fams.values():
        members.sort(key=lambda r: r[key])
        n = len(members)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and members[j + 1][key] == members[i][key]:
                j += 1
            # positions i..j (0-based) share the value; midrank position
            rank = 0.5 if n == 1 else round((i + j) / 2 / (n - 1), 4)
            for k in range(i, j + 1):
                members[k][f"{key}_rank"] = rank
            i = j + 1


def _apply_group(row: dict, stats: dict):
    if not stats.get("available"):
        row["obs_available"] = False
        return
    row["obs_available"] = True
    for k in ("act", "hi", "lo", "hilo", "variance", "n_samples",
              "n_hi", "n_lo", "var_hi", "var_lo", "hilo_unstable"):
        row[f"obs_{k}"] = round(stats[k], 6) if isinstance(
            stats[k], float) else stats[k]
    # The proposer-visible contrast channel: signed, eps-stabilized log
    # of the hi/lo activation ratio at fixed 2-decimal precision. Both
    # the prompt renderer and interp-v2 consume THIS field, so the
    # evidence channel is identical across model and algorithm (R1).
    row["obs_log_contrast"] = round(math.log(stats["hilo"]),
                                    LOG_CONTRAST_DECIMALS)


def annotate_catalog(catalog: List[dict], obs: dict,
                     targets: Dict[int, str]) -> List[dict]:
    """Return a NEW catalog list with obs fields merged per row."""
    out = []
    for e in catalog:
        r = dict(e)
        path = targets[e["param_id"]]
        r["obs_module"] = path
        r["obs_group"] = path  # group identity = hooked module path
        _apply_group(r, obs["groups"].get(path, {"available": False}))
        out.append(r)
    _family_ranks(out, "obs_act")
    _family_ranks(out, "obs_hilo")
    return out


def _visible_signature(row: dict) -> tuple:
    """The exact evidence features rendered to the proposer (ar, lc).

    Effective placebo coverage is measured on THIS tuple, not on the
    raw statistics: an association that moved but left the visible
    features identical is not a real change for the A1-vs-A2
    comparison (HZN-IR-04 R2).
    """
    return (row.get("obs_act_rank"), row.get("obs_log_contrast"))


def shuffle_observations(annotated: List[dict], seed: int,
                         band: int = SHUFFLE_LAYER_BAND) -> dict:
    """Seeded placebo: permute whole obs vectors across unique obs_groups.

    Groups (not rows) are the permutation unit: rows sharing an
    obs_group are one deduplicated measurement, so every member row
    receives the SAME reassigned vector — a shared hooked module can
    never show its parameters conflicting observations (HZN-IR-04 R2).
    The joint (act, hi, lo, hilo, ...) feature relationship moves
    intact. Strata are (family, layer-band). Tie-aware ranks are
    recomputed after permutation so they stay internally consistent.

    Returns the shuffled catalog, the audit permutation
    {obs_group: source_obs_group}, the group count, and the effective
    coverage fraction measured on proposer-visible features (ar, lc).
    A stratum of one group cannot be permuted and counts as unchanged.
    """
    rows = [dict(r) for r in annotated]
    obs_keys = sorted({k for r in rows for k in r
                       if k.startswith("obs_")
                       and k not in ("obs_module", "obs_group")})
    # Unique observation groups -> member row indices; stratum from the
    # group's first row (a group is one hooked module: same family/layer).
    group_rows: Dict[str, List[int]] = {}
    group_sig_pre: Dict[str, tuple] = {}
    for i, r in enumerate(rows):
        if r.get("obs_available"):
            group_rows.setdefault(r["obs_group"], []).append(i)
            group_sig_pre[r["obs_group"]] = _visible_signature(r)
    strata: Dict[tuple, List[str]] = {}
    for g, idxs in group_rows.items():
        r0 = rows[idxs[0]]
        strata.setdefault((r0["family"], r0["layer"] // band), []).append(g)

    rng = random.Random(seed)
    permutation: Dict[str, str] = {}
    vectors: Dict[str, dict] = {}
    for key, groups in strata.items():
        order = list(range(len(groups)))
        rng.shuffle(order)
        for g, j in zip(groups, order):
            src = groups[j]
            permutation[g] = src
            vectors[g] = {k: rows[group_rows[src][0]].get(k)
                          for k in obs_keys}
    for g, idxs in group_rows.items():
        for i in idxs:
            for k in obs_keys:
                rows[i][k] = vectors[g][k]
    # Recompute ranks on the permuted assignment for internal consistency.
    for r in rows:
        r.pop("obs_act_rank", None)
        r.pop("obs_hilo_rank", None)
    _family_ranks(rows, "obs_act")
    _family_ranks(rows, "obs_hilo")

    changed = 0
    for g, idxs in group_rows.items():
        if _visible_signature(rows[idxs[0]]) != group_sig_pre[g]:
            changed += 1
    total = len(group_rows)
    return {"catalog": rows,
            "effective_coverage": (changed / total if total else 0.0),
            "n_groups": total,
            "n_groups_shuffled": total,
            "permutation": permutation,
            "seed": seed}


def interpreter_weights(catalog: List[dict]) -> Dict[int, float]:
    """FROZEN conventional evidence rule (interp-v2) for B1/B2.

    w_i = FLOOR + minmax(|obs_log_contrast_i|) within family — the
    identical 2-decimal signed log-contrast shown to the model in the
    lc column, so the evidence channel is bit-identical across the
    model and algorithm conditions (HZN-IR-04 R1). Direction-agnostic
    by design; unavailable observations get the floor weight only.
    """
    fams: Dict[str, List[dict]] = {}
    for e in catalog:
        fams.setdefault(e["family"], []).append(e)
    weights: Dict[int, float] = {}
    for members in fams.values():
        s = [abs(m["obs_log_contrast"]) if m.get("obs_available")
             and m.get("obs_log_contrast") is not None
             else 0.0 for m in members]
        lo, hi = min(s), max(s)
        for m, v in zip(members, s):
            norm = (v - lo) / (hi - lo) if hi > lo else 0.0
            weights[m["param_id"]] = INTERPRETER_FLOOR + norm
    return weights


def obs_sha256(obs: dict) -> str:
    """Hash the observation block into the run manifest."""
    return hashlib.sha256(
        json.dumps(obs, sort_keys=True, default=str).encode()).hexdigest()
