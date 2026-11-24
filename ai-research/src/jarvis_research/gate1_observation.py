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
  relationship; effective coverage is recorded.
- The conventional evidence interpreter (B1) is a FROZEN prespecified
  rule; it is deliberately direction-agnostic.
"""

import hashlib
import json
import math
import random
from typing import Dict, List, Optional

import torch

OBSERVATION_VERSION = "obs-v1"
# Stabilization: eps = this fraction of the MEDIAN positive group RMS.
# Frozen; changing it is a protocol change.
EPS_FRACTION = 0.01
# Layer bands for shuffle strata (layers grouped into bands of this size
# so a stratum usually contains >1 observation group).
SHUFFLE_LAYER_BAND = 8

# Frozen conventional interpreter (HZN-IR-03 R1). B1 weights each catalog
# row by the STRENGTH of its high/low-loss association, direction-
# agnostic: s_i = minmax(|ln hilo_i|) within family, sampling weight
# w_i = FLOOR + s_i. The floor keeps every row reachable; min-max keeps
# the rule scale-free. Ties fall through to param_id order in the
# weighted sampler. This rule is prespecified and must not be tuned on
# outcomes — any change bumps INTERPRETER_VERSION.
INTERPRETER_VERSION = "interp-v1"
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
                 "per_passage_rms")

    def __init__(self):
        self.n_hi = 0
        self.n_lo = 0
        self.rms_hi = 0.0
        self.rms_lo = 0.0
        self.max_abs = 0.0
        self.per_passage_rms = []

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
        else:
            self.n_lo += 1
            self.rms_lo += rms
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
        group_stats[path] = {
            "available": True,
            "act": act, "hi": hi_m, "lo": lo_m,
            "hilo": (hi_m + eps) / (lo_m + eps),
            "max_abs": g.max_abs,
            "variance": var, "n_samples": n,
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
    """In-place [0,1] ranks of `key` within each family; single -> 0.5."""
    fams: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("obs_available"):
            fams.setdefault(r["family"], []).append(r)
    for members in fams.values():
        members.sort(key=lambda r: (r[key], r["param_id"]))
        n = len(members)
        for i, r in enumerate(members):
            r[f"{key}_rank"] = 0.5 if n == 1 else round(i / (n - 1), 4)


def _apply_group(row: dict, stats: dict):
    if not stats.get("available"):
        row["obs_available"] = False
        return
    row["obs_available"] = True
    for k in ("act", "hi", "lo", "hilo", "variance", "n_samples",
              "hilo_unstable"):
        row[f"obs_{k}"] = round(stats[k], 6) if isinstance(
            stats[k], float) else stats[k]


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


def shuffle_observations(annotated: List[dict], seed: int,
                         band: int = SHUFFLE_LAYER_BAND) -> dict:
    """Seeded placebo: permute whole obs vectors within (family, band).

    The joint (act, hi, lo, hilo, variance) relationship is preserved —
    only its parameter correspondence moves. Ranks are recomputed after
    permutation so they stay internally consistent. Returns a dict with
    the shuffled catalog and coverage provenance: the fraction of
    obs_groups whose association actually changed (HZN-IR-03 R2). A
    stratum of one group cannot be permuted and counts as unchanged.
    """
    obs_keys = ("obs_act", "obs_hi", "obs_lo", "obs_hilo", "obs_variance",
                "obs_n_samples", "obs_hilo_unstable", "obs_available",
                "obs_act_rank", "obs_hilo_rank")
    rows = [dict(r) for r in annotated]
    strata: Dict[tuple, List[int]] = {}
    for i, r in enumerate(rows):
        if r.get("obs_available"):
            strata.setdefault((r["family"], r["layer"] // band), []).append(i)
    rng = random.Random(seed)
    changed = 0
    total = 0
    permutation = {}
    for key, idxs in strata.items():
        vals = [{k: rows[i].get(k) for k in obs_keys} for i in idxs]
        order = list(range(len(idxs)))
        rng.shuffle(order)
        total += len(idxs)
        for i, j in zip(idxs, order):
            original = {k: rows[i].get(k) for k in obs_keys}
            if vals[j] != original:
                changed += 1
            permutation[rows[i]["obs_group"]] = rows[idxs[j]]["obs_group"]
            for k in obs_keys:
                rows[i][k] = vals[j][k]
    # Recompute ranks on the permuted assignment for internal consistency.
    for r in rows:
        r.pop("obs_act_rank", None)
        r.pop("obs_hilo_rank", None)
    _family_ranks(rows, "obs_act")
    _family_ranks(rows, "obs_hilo")
    return {"catalog": rows,
            "effective_coverage": (changed / total if total else 0.0),
            "n_groups_shuffled": total,
            "seed": seed}


def interpreter_weights(catalog: List[dict]) -> Dict[int, float]:
    """FROZEN conventional evidence rule (interp-v1) for B1/B2.

    w_i = FLOOR + minmax(|ln hilo_i|) within family. Direction-agnostic
    by design; unavailable observations get the floor weight only.
    """
    fams: Dict[str, List[dict]] = {}
    for e in catalog:
        fams.setdefault(e["family"], []).append(e)
    weights: Dict[int, float] = {}
    for members in fams.values():
        s = [abs(math.log(m["obs_hilo"])) if m.get("obs_available")
             and m["obs_hilo"] > 0 else 0.0 for m in members]
        lo, hi = min(s), max(s)
        for m, v in zip(members, s):
            norm = (v - lo) / (hi - lo) if hi > lo else 0.0
            weights[m["param_id"]] = INTERPRETER_FLOOR + norm
    return weights


def obs_sha256(obs: dict) -> str:
    """Hash the observation block into the run manifest."""
    return hashlib.sha256(
        json.dumps(obs, sort_keys=True, default=str).encode()).hexdigest()
