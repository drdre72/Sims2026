#!/usr/bin/env python3
"""gate1_pilot.py — Gate 1 exploratory engineering pilot.

This script runs an EXPLORATORY Gate 1 engineering loop on a small scale to:
  1. Verify every component works end-to-end
  2. Estimate variance and total compute cost
  3. Identify remaining failures before freezing the confirmatory protocol

It is EXPLORATORY, not confirmatory. Results from this pilot MUST NOT be
used for promotion decisions. The confirmatory experiment requires a
separately frozen protocol, independent evaluator approval, and a
genuinely untouched final holdout.

Pipeline (per reviewer's acceptance logic):
  1. PROPOSE: model emits a structured edit proposal (or control generates one)
  2. VALIDATE: patch manager validates against arm schema (hard bounds)
  3. APPLY: in-place under no_grad, track touched tensors
  4. QUALITY CHECK: NaN/Inf + degenerate + perplexity explosion -> rollback if collapsed
  5. EVALUATE PRACTICE: feedback signal for proposer (PRACTICE partition only)
  6. (After N steps) EVALUATE SELECTION: primary promotion metric (hidden from proposer)
  7. EVALUATE REGRESSION: per-domain capability check
  8. COMPARE CONTROLS: random, fixed heuristic, adaptive practice search;
     actual proposal counts and GPU-wall seconds are reported, not assumed equal
  9. AUDIT: full trail (pinned revision/shard hashes, proposals, seeds,
     scores, synchronized GPU-operation wall seconds)

Arms:
  A: sparse additive (mid/late layers, 11-31)
  B: bounded attenuation (0.70-0.99, all layers)

Controls (per arm):
  random: within-arm random selection. NOT exactly matched by tensor family,
          parameter count, perturbation norm, or computational budget.
  fixed_heuristic: fixed algorithm (not feedback-adaptive).
  adaptive: compare random candidates using only practice feedback.

Usage:
    .venv-opt\\Scripts\\python.exe src/jarvis_research/gate1_pilot.py
        [--seeds 1 2 3] [--steps 8] [--smoke] [--out <filename>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

_THIS = Path(__file__).resolve()
AI_RESEARCH = _THIS.parents[2]
RESULTS = AI_RESEARCH / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3.5-4B"
DTYPE = torch.bfloat16

# Representative late-trajectory feedback for prompt-length preflight.
_SAMPLE_HISTORY = [
    {"step": i, "practice_delta_ce": -0.0013 + 0.0004 * i,
     "predicted_delta_ce": -0.002}
    for i in range(4)]


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=AI_RESEARCH, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Control generators. Each produces a sequence of proposals for a given arm,
# matched to the arm's allowed surface and the proposer's step count.
# ---------------------------------------------------------------------------

def random_control_proposals(
    catalog: List[dict],
    arm_config: dict,
    n_steps: int,
    seed: int,
    weights: Optional[Dict[int, float]] = None,
) -> List[dict]:
    """Generate n_steps random proposals from the SHARED legal catalog.

    Gate 1.1: draws from the identical catalog the proposer sees, so the
    random control searches the same parameter space. Magnitude and count
    are drawn uniformly from each entry's legal range — the same bounds
    shown to the model. This matches the SEARCH SPACE and proposal budget;
    it does not match the model's parameter-selection distribution (that is
    the quantity under test).

    `weights` (param_id -> weight) implements the Gate 1.2 evidence-guided
    conventional interpreter (interp-v1): biased parameter SELECTION with
    identical magnitude/count sampling, so any A-vs-B difference isolates
    the evidence channel rather than the proposal format.
    """
    import random as _random
    g = _random.Random(seed * 31 + 17)
    proposals = []
    if not catalog:
        return proposals
    w_list = ([max(weights.get(e["param_id"], 0.0), 0.0) for e in catalog]
              if weights else None)

    for step in range(n_steps):
        entry = (g.choices(catalog, weights=w_list, k=1)[0]
                 if w_list else catalog[g.randrange(len(catalog))])
        if arm_config["name"] == "sparse_additive":
            magnitude = g.uniform(entry["mag_min"], entry["mag_max"])
            count = g.randint(entry["count_min"], entry["count_max"])
            proposals.append({
                "edits": [{"param_id": entry["param_id"], "operation": "add_sparse",
                           "magnitude": magnitude, "count": count}],
                "rationale": "random control",
                "predicted_effect": "random",
            })
        elif arm_config["name"] == "bounded_attenuation":
            magnitude = g.uniform(entry["mag_min"], entry["mag_max"])
            proposals.append({
                "edits": [{"param_id": entry["param_id"], "operation": "scale_dense",
                           "magnitude": magnitude}],
                "rationale": "random control",
                "predicted_effect": "random",
            })
    return proposals


def guided_control_proposals(
    arm_config: dict,
    n_steps: int,
) -> List[dict]:
    """Generate n_steps FIXED-HEURISTIC proposals for the given arm.

    For Arm A (sparse additive): we cannot compute gradients without a
    labeled target, so the guided control uses a fixed pattern — apply
    small positive deltas to mid-layer MLP gate projections, which is
    a reasonable heuristic based on Phase A (MLP is robust, mid-layers
    are safe).

    For Arm B (bounded attenuation): uniform attenuation at the midpoint
    factor (0.85) applied to one random mid-layer projection per step.
    This is a simple, fixed strategy that matches the seed-303 pattern
    in spirit but uses a single layer and the midpoint factor.
    """
    proposals = []
    if arm_config["name"] == "sparse_additive":
        # Fixed: small positive delta on mid-layer MLP, count=256
        target_layer = 16  # mid
        delta = 0.001  # small, within bounds
        count = 256
        for step in range(n_steps):
            # Rotate through MLP types
            mlp_types = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
            pname = f"model.layers.{target_layer}.{mlp_types[step % 3]}"
            proposals.append({
                "edits": [{"param": pname, "operation": "add_sparse",
                           "magnitude": delta, "count": count}],
                "rationale": "fixed mid-layer MLP heuristic",
                "predicted_effect": "fixed heuristic",
            })
    elif arm_config["name"] == "bounded_attenuation":
        # Fixed: midpoint factor 0.85 on one mid-layer projection per step
        target_layer = 16
        factor = 0.85
        proj_types = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
        for step in range(n_steps):
            pname = f"model.layers.{target_layer}.{proj_types[step % 3]}"
            proposals.append({
                "edits": [{"param": pname, "operation": "scale_dense",
                           "magnitude": factor}],
                "rationale": "fixed midpoint attenuation on mid-layer",
                "predicted_effect": "fixed heuristic",
            })
    return proposals


def _layer_idx(name: str) -> Optional[int]:
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        if i + 1 < len(parts):
            return int(parts[i + 1])
    return None


def local_provenance(model_id, revision=None):
    """Hash the complete cached safetensors shards; never download."""
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id=model_id, revision=revision,
                                  local_files_only=True))
    revision = path.name
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise RuntimeError("cached checkpoint is not a pinned commit snapshot")
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("no local checkpoint shards to hash")
    hashes = {}
    for shard in shards:
        digest = hashlib.sha256()
        with shard.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        hashes[shard.name] = digest.hexdigest()
    return {"revision": revision, "shard_sha256": hashes}


def adaptive_choice(model, tok, ev, patch_manager, seed, step, catalog,
                    pool: int = 2,
                    weights: Optional[Dict[int, float]] = None,
                    current_ce: float = float("inf")):
    """Compare trial edits on practice; restore after each trial.

    Evaluates `pool` candidates per iteration. The caller sets the iteration
    count so total practice evaluations match the autonomous trajectory's
    budget. `weights` (Gate 1.2 interp-v2) biases candidate parameter
    selection toward high |log-contrast| rows without changing magnitude
    sampling or the practice-evaluation budget.

    `current_ce` is the practice CE of the checkpoint BEFORE this
    iteration; the best trial is committed only when it strictly
    improves on it (HZN-IR-05 §3 — the unchanged checkpoint is a
    legitimate choice, matching the model's abstain affordance).
    """
    from jarvis_research.gate1_patch_manager import PatchManager
    arm_config = patch_manager.arm
    choices = random_control_proposals(catalog, arm_config, pool,
                                       seed * 1009 + step, weights=weights)
    best, best_ce = None, float("inf")
    trial_seconds = 0.0
    for idx, choice in enumerate(choices):
        if patch_manager.validate(choice)[1]:
            continue
        trial = PatchManager(model, tok, arm_config,
                             seed * 1009 + step * 3 + idx,
                             quality_baseline_ppl=patch_manager.quality_baseline_ppl)
        edits, error = trial.validate(choice)
        if error:
            continue
        try:
            _, apply_s = trial.apply(edits)
            collapsed, _, _, _, quality_s = trial.quality_check()
            trial_seconds += apply_s + quality_s
            if not collapsed:
                practice = ev.evaluate_practice(model, tok)
                trial_seconds += practice.gpu_seconds
                ce = math.log(practice.aggregate_ppl)
                if ce < best_ce:
                    best, best_ce = choice, ce
        finally:
            ok, bad, rollback_s = trial.rollback_all()
            trial_seconds += rollback_s
            if not ok:
                raise RuntimeError(f"adaptive trial rollback failed: {bad}")
    if not _adaptive_commit(best, best_ce, current_ce):
        return None, trial_seconds, len(choices)
    return best, trial_seconds, len(choices)


def _adaptive_commit(best, best_ce: float, current_ce: float) -> bool:
    """Commit only when a trial STRICTLY improves current practice CE.

    Kept as a named predicate so the keep-checkpoint boundary is
    directly unit-testable (HZN-IR-05 §3).
    """
    return best is not None and best_ce < current_ce


def _budget_exceeded(stage_totals: dict, compute_budget) -> bool:
    """Measured-compute stopping rule, checked before EVERY step so all
    outcome classes consume the same allocation (HZN-IR-05 §3)."""
    return (compute_budget is not None
            and sum(stage_totals.values()) >= compute_budget)


# ---------------------------------------------------------------------------
# Main pilot runner
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--arms", nargs="+", choices=("A", "B"), default=["A", "B"])
    ap.add_argument("--conditions", nargs="+",
                    choices=("autonomous", "random", "fixed_heuristic",
                             "adaptive",
                             # Gate 1.2 six-condition architecture (REV 1 §4):
                             # A=model, B=conventional; 0=blind, 1=real obs,
                             # 2=shuffled placebo.
                             "A0", "A1", "A2", "B0", "B1", "B2"),
                    default=["autonomous", "random", "fixed_heuristic", "adaptive"])
    ap.add_argument("--adaptive-pool", type=int, default=2,
                    help="candidate edits evaluated per adaptive step")
    ap.add_argument("--adaptive-step-divisor", type=int, default=1,
                    help="adaptive iterations = ceil(steps/divisor); use 2 for "
                         "practice-evaluation-budget matching vs autonomous")
    ap.add_argument("--budget-mode", choices=("interventions", "seconds"),
                    default="interventions",
                    help="interventions: fixed proposal allocation per "
                         "condition (intervention-matched comparison). "
                         "seconds: every condition gets the same "
                         "--compute-budget-seconds of measured stage "
                         "time (compute-matched comparison; enables "
                         "H-OBS.3)")
    ap.add_argument("--compute-budget-seconds", type=float, default=None,
                    help="per-condition measured-compute ceiling in "
                         "seconds mode")
    ap.add_argument("--max-steps", type=int, default=64,
                    help="safety cap on proposals per condition in "
                         "seconds mode")
    ap.add_argument("--data-dir", default="gate1_eval_v4",
                    help="development partition directory under data/ "
                         "(Gate 1.2 uses fresh v4 partitions; v3 stays "
                         "as the examined Gate 1.1 comparison set)")
    ap.add_argument("--out", default="gate1_pilot_v2.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip candidates already present in --out")
    ap.add_argument("--preflight", action="store_true",
                    help="Gate 1.2 observation-layer preflight ONLY: collect "
                         "observations, verify hook coverage, measure peak "
                         "VRAM and prompt tokens, write --out, run NO "
                         "candidates (HZN-IR-03 R4)")
    args = ap.parse_args()

    # Gate 1.2 condition codes; legacy names alias to their blind variants.
    _ALIAS = {"autonomous": "A0", "adaptive": "B0"}
    args.conditions = [_ALIAS.get(c, c) for c in args.conditions]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jarvis_research.gate1_evaluator import Evaluator, check_regression
    from jarvis_research.gate1_patch_manager import (
        PatchManager, make_arm_config, build_catalog, probe_ppl)
    from jarvis_research.gate1_proposer import SelfProposer, PROMPT_VERSION
    from jarvis_research.gate1_capability import (
        evaluate_capability, capability_regression, suite_fingerprint,
        transition_ledger)
    from jarvis_research.gate1_observation import (
        OBSERVATION_VERSION, INTERPRETER_VERSION, resolve_hook_targets,
        collect_observations, annotate_catalog, shuffle_observations,
        interpreter_weights, obs_sha256)
    from jarvis_research.gate1_proposer import PROMPT_VERSION_OBS

    if args.steps < 1:
        ap.error("--steps must be positive")
    if args.adaptive_pool < 1 or args.adaptive_step_divisor < 1:
        ap.error("adaptive knobs must be positive")
    if args.max_steps < 1:
        ap.error("--max-steps must be positive")
    if args.budget_mode == "seconds" and (
            args.compute_budget_seconds is None
            or args.compute_budget_seconds <= 0):
        ap.error("--budget-mode seconds requires "
                 "--compute-budget-seconds > 0")
    data_dir = AI_RESEARCH / "data" / args.data_dir
    ev = Evaluator.load(
        data_dir / "practice.json",
        data_dir / "selection.json",
        data_dir / "regression.json",
    )

    t0 = time.perf_counter()
    print(f"[load] {MODEL_ID} ...", flush=True)
    provenance = local_provenance(MODEL_ID)
    revision = provenance["revision"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision,
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=DTYPE, device_map={"": "cuda"},
        local_files_only=True)
    model.eval()
    torch.cuda.synchronize()
    params = {n: p for n, p in model.named_parameters()}
    print(f"[load] done in {time.perf_counter()-t0:.1f}s, {len(params)} params",
          flush=True)

    # Run manifest: hash every protocol element that must match for
    # --resume to be valid. A changed prompt, partition file, arm config,
    # catalog or capability suite produces a different manifest hash and
    # the resume refuses rather than silently mixing protocol versions.
    catalogs = {a: build_catalog(make_arm_config(a), params)
                for a in args.arms}

    # Gate 1.2 observation layer: collected ONCE on the unmodified
    # checkpoint, before any candidate edit, and hashed into the manifest.
    # Needed for A1/A2/B1/B2; skipped for blind-only runs.
    obs_needed = any(c in ("A1", "A2", "B1", "B2") for c in args.conditions)
    obs_block = None
    obs_targets = {}
    obs_seconds = 0.0
    if obs_needed:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_obs = time.perf_counter()
        for a in args.arms:
            obs_targets[a] = resolve_hook_targets(model, catalogs[a])
        # Union of all arm targets -> every hooked path, not just arm[0]'s.
        all_targets = {}
        for t in obs_targets.values():
            for pid, path in t.items():
                all_targets[path] = path
        obs_block = collect_observations(
            model, tok, ev.practice, all_targets,
            checkpoint_revision=revision)
        obs_seconds = time.perf_counter() - t_obs
        n_groups = len(obs_block["groups"])
        n_unavail = sum(1 for g in obs_block["groups"].values()
                        if not g.get("available"))
        print(f"[obs] {n_groups} hooked modules ({n_unavail} unexecuted), "
              f"hi={len(obs_block['hi_passages'])} "
              f"lo={len(obs_block['lo_passages'])} passages, "
              f"eps={obs_block['eps']:.4g}, {obs_seconds:.1f}s",
              flush=True)

    manifest = {
        "model_revision": revision,
        # Bind the exact weight bytes and implementation revision to the
        # protocol identity: same revision string on different bytes, or
        # a source edit between resume points, must refuse (HZN-IR-02 §6).
        "shard_sha256": provenance["shard_sha256"],
        "git_sha": _git_sha(),
        "source_sha256": {
            f.name: _file_sha256(f)
            for f in sorted(
                (Path(__file__).resolve().parent).glob("gate1_*.py"))},
        "data_dir": args.data_dir,
        "partition_sha256": {
            k: _file_sha256(data_dir / f"{k}.json")
            for k in ("practice", "selection", "regression")},
        "proposer_prompt_sha256": PROMPT_VERSION,
        "arm_config_sha256": {a: _sha256_obj(make_arm_config(a))
                              for a in catalogs},
        # Full descriptors (name, family, layer, shape, numel, std, bounds),
        # not just names — a bound change must invalidate a resume.
        "catalog_sha256": {a: _sha256_obj(c)
                           for a, c in catalogs.items()},
        "capability_suite_sha256": suite_fingerprint(),
        "observation": ({
            "sha256": obs_sha256(obs_block),
            "version": OBSERVATION_VERSION,
            "interpreter_version": INTERPRETER_VERSION,
        } if obs_block else None),
        "prompt_sha256_by_condition": {
            c: (PROMPT_VERSION_OBS if c in ("A1", "A2") else PROMPT_VERSION)
            for c in args.conditions if c in ("A0", "A1", "A2")},
        "seeds": sorted(args.seeds),
        "steps": args.steps,
        "max_new": args.max_new,
        "adaptive_pool": args.adaptive_pool,
        "adaptive_step_divisor": args.adaptive_step_divisor,
        # Budget is protocol: the matching regime belongs in the
        # manifest hash so intervention-matched and compute-matched
        # records can never be silently mixed by --resume (HZN-IR-04 R3).
        "budget_mode": args.budget_mode,
        "compute_budget_seconds": args.compute_budget_seconds,
        "max_steps": args.max_steps if args.budget_mode == "seconds" else None,
        "conditions": sorted(args.conditions),
    }
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    if args.preflight:
        # Observation-layer preflight (HZN-IR-03 R4): verify hook coverage
        # (resolve_hook_targets already raises on misses), record group
        # counts/unavailable rows, prompt token cost, placebo coverage and
        # peak VRAM. Runs NO candidates.
        pre = {"gate": "gate1_2_observation_preflight",
               "manifest_sha256": manifest["sha256"],
               "observation_seconds": round(obs_seconds, 2),
               "peak_vram_bytes": (torch.cuda.max_memory_allocated()
                                   if torch.cuda.is_available() else None)}
        # HZN-IR-04 reconciliation: mtp.* checkpoint tensors are never
        # instantiated in the CausalLM named_parameters — record the
        # live count so "excluded from the edit surface" is verified,
        # not assumed (the earlier 236-row figure was miscounted; the
        # guard never had tensors to remove).
        pre["named_parameters"] = len(params)
        pre["mtp_params_in_named_parameters"] = sum(
            "mtp" in n for n in params)
        if obs_block:
            avail = [g for g in obs_block["groups"].values()
                     if g.get("available")]
            pre["observation"] = {
                "n_groups": len(obs_block["groups"]),
                "n_unexecuted": sum(1 for g in obs_block["groups"].values()
                                    if not g.get("available")),
                "eps": obs_block["eps"],
                "hi_passages": obs_block["hi_passages"],
                "lo_passages": obs_block["lo_passages"],
                # Separate hi/lo sample counts and variance (R4 audit).
                "min_n_hi": min((g["n_hi"] for g in avail), default=0),
                "min_n_lo": min((g["n_lo"] for g in avail), default=0),
                "max_var_hi": max((g["var_hi"] for g in avail), default=0),
                "max_var_lo": max((g["var_lo"] for g in avail), default=0),
                "checkpoint": obs_block["checkpoint_sha256"],
                # Extended provenance tuple (HZN-IR-07 d7): binds the
                # measurement to the actual model state + context.
                "provenance": obs_block.get("provenance"),
            }
            pre["arms"] = {}
            for a in args.arms:
                ann = annotate_catalog(catalogs[a], obs_block,
                                       obs_targets[a])
                n_un = sum(1 for r in ann if not r.get("obs_available"))
                sh = shuffle_observations(ann, seed=0)
                pm = PatchManager(model, tok, make_arm_config(a), seed=0)
                pr = SelfProposer(model, tok, pm, args.max_new,
                                  catalog_override=ann, obs=True)
                pr_blind = SelfProposer(model, tok, pm, args.max_new,
                                        catalog_override=catalogs[a],
                                        obs=False)
                # Parameter -> hook attribution: catalog params sharing
                # one hooked module receive ONE shared observation, not
                # independent evidence (HZN-IR-04 §5).
                per_group: Dict[str, int] = {}
                for r in ann:
                    per_group[r["obs_group"]] = (
                        per_group.get(r["obs_group"], 0) + 1)
                pre["arms"][a] = {
                    "catalog_rows": len(ann),
                    "obs_unavailable_rows": n_un,
                    "obs_groups": len(per_group),
                    "shared_hook_groups": sum(
                        1 for n in per_group.values() if n > 1),
                    "max_params_per_group": (max(per_group.values())
                                             if per_group else 0),
                    # FULL serialized prompt (system + user + template),
                    # not catalog text alone; the _with_feedback variant
                    # includes 4 accumulated history entries as produced
                    # late in a trajectory (HZN-IR-05 §6).
                    "serialized_prompt_tokens": {
                        "A0": pr_blind.serialized_prompt_tokens(),
                        "A1": pr.serialized_prompt_tokens(),
                        "A0_with_feedback":
                            pr_blind.serialized_prompt_tokens(
                                step=3, practice_ce=2.5,
                                history=_SAMPLE_HISTORY),
                        "A1_with_feedback":
                            pr.serialized_prompt_tokens(
                                step=3, practice_ce=2.5,
                                history=_SAMPLE_HISTORY),
                    },
                    "shuffle_effective_coverage_seed0":
                        sh["effective_coverage"],
                    "n_groups_shuffled": sh["n_groups_shuffled"],
                    "shuffle_permutation_recorded": bool(
                        sh.get("permutation")),
                }
        out = RESULTS / args.out
        out.write_text(json.dumps(pre, indent=2), encoding="utf-8")
        print(f"[preflight] wrote {out}", flush=True)
        return 0

    # Baseline evaluation on ALL partitions + measured quality-guard baseline
    # + baseline capability suite (all on the UNMODIFIED pinned checkpoint).
    print("\n[baseline] evaluating all partitions ...", flush=True)
    base_practice = ev.evaluate_practice(model, tok)
    base_selection = ev.evaluate_selection(model, tok)
    base_regression = ev.evaluate_regression(model, tok)
    quality_baseline_ppl = probe_ppl(model, tok)
    print(f"[baseline] practice={base_practice.aggregate_ppl:.3f} "
          f"selection={base_selection.aggregate_ppl:.3f} "
          f"quality_probe_ppl={quality_baseline_ppl:.3f} "
          f"({base_regression.n_domains} domains)", flush=True)
    for d, r in base_regression.per_domain.items():
        print(f"  {d:15s} {r.aggregate_ppl:.3f}", flush=True)
    print("[baseline] capability suite ...", flush=True)
    base_capability = evaluate_capability(model, tok)
    print(f"[baseline] capability: {base_capability.per_family}", flush=True)

    report = {
        "gate": "gate1_exploratory_model_pilot_v3",
        "model": MODEL_ID, "git_sha": _git_sha(), "provenance": provenance,
        "proposer_prompt_sha256": PROMPT_VERSION,
        "proposal_max_new": args.max_new,
        "shared_catalog": "all conditions draw from build_catalog()",
        "adaptive_config": {
            "pool": args.adaptive_pool,
            "step_divisor": args.adaptive_step_divisor,
            "note": ("adaptive iterations = ceil(steps/divisor); pool "
                     "candidates evaluated per iteration on practice only"),
        },
        "timing_units": "synchronized GPU-operation wall seconds, not hardware utilization",
        "isolation": "prompt-only; no filesystem security boundary",
        "purpose": ("EXPLORATORY engineering pilot. NOT confirmatory. "
                    "Verify pipeline, estimate variance + cost, identify failures."),
        "n_seeds": len(args.seeds),
        "n_steps": args.steps,
        "baseline": {
            "practice": asdict(base_practice),
            "selection": asdict(base_selection),
            "quality_probe_ppl": quality_baseline_ppl,
            "capability": base_capability.to_dict(),
            "regression": {
                "per_domain": {d: asdict(r) for d, r in base_regression.per_domain.items()},
                "gpu_seconds": base_regression.gpu_seconds,
            },
        },
        "manifest": manifest,
        "observation_seconds": round(obs_seconds, 2),
        # R3: whether this run ENFORCED a compute-matched allocation or
        # only matched intervention counts. H-OBS.3 (interpretive
        # advantage) is declared unresolved unless seconds mode ran —
        # a descriptive flag, not a hypothesis test.
        "compute_matching": ("enforced" if args.budget_mode == "seconds"
                             else "not_enforced_intervention_matched_only"),
        "h_obs_3_status": (
            "addressable: compute-matched allocation enforced"
            if args.budget_mode == "seconds"
            else "unresolved in this run: intervention-matched only"),
        "compute_accounting": ("observation_seconds is the shared "
                               "collection cost, amortized over the "
                               "benefiting candidate records at report "
                               "finalization (sum of per-record "
                               "obs_amortized_seconds == "
                               "observation_seconds); per-candidate "
                               "stage_seconds record the IN-LOOP stages "
                               "(proposal/validation/application/quality/"
                               "practice/rollback) that the compute "
                               "budget bounds; final_eval_seconds is the "
                               "separate fixed evaluation allowance "
                               "(HZN-IR-08 finding 4) and "
                               "unattributed_wall_seconds is wall time "
                               "outside both"),
        "candidates": [],
    }
    out = RESULTS / args.out
    completed = set()
    if args.resume and out.exists():
        prior = json.loads(out.read_text(encoding="utf-8"))
        prior_sha = (prior.get("manifest") or {}).get("sha256")
        if prior_sha != manifest["sha256"]:
            raise SystemExit(
                f"--resume manifest mismatch: prior={prior_sha} != "
                f"current={manifest['sha256']}; refusing to mix protocol "
                f"versions. Use a different --out or rerun without --resume.")
        report["baseline"] = prior.get("baseline", report["baseline"])
        report["candidates"] = prior.get("candidates", [])
        completed = {c.get("label") for c in report["candidates"]}
        print(f"[resume] {len(completed)} candidates already complete",
              flush=True)

    # Regression tolerance (preliminary, for pilot only — will be set
    # formally when confirmatory protocol is frozen). Using 0.05 nats
    # CE per domain as a placeholder.
    pilot_tolerance = {d: 0.05 for d in base_regression.per_domain}

    for arm_name in args.arms:
        arm_config = make_arm_config(arm_name)
        catalog = catalogs[arm_name]
        print(f"[catalog] arm {arm_name}: {len(catalog)} legal entries",
              flush=True)
        for condition in args.conditions:
            for seed in args.seeds:
                label = f"arm{arm_name}_{condition}_s{seed}"
                if label in completed:
                    print(f"\n[{label}] already complete, skipping", flush=True)
                    continue
                print(f"\n[{label}] starting ...", flush=True)
                candidate_start = time.perf_counter()
                pm = PatchManager(model, tok, arm_config, seed=seed,
                                  quality_baseline_ppl=quality_baseline_ppl)
                pm.candidate.parent_checkpoint_sha = revision
                # Gate 1.2 information tiers. Shuffled variants are
                # re-permuted per seed and their coverage is recorded.
                annotated = None
                shuffle_info = None
                if condition in ("A1", "B1"):
                    annotated = annotate_catalog(catalog, obs_block,
                                                 obs_targets[arm_name])
                elif condition in ("A2", "B2"):
                    sh = shuffle_observations(
                        annotate_catalog(catalog, obs_block,
                                         obs_targets[arm_name]),
                        seed=seed)
                    annotated = sh["catalog"]
                    shuffle_info = {
                        "seed": seed,
                        "effective_coverage": sh["effective_coverage"],
                        "n_groups_shuffled": sh["n_groups_shuffled"],
                        "permutation": sh.get("permutation"),
                    }
                cond_catalog = annotated or catalog
                proposer = (SelfProposer(
                    model, tok, pm, args.max_new,
                    catalog_override=cond_catalog,
                    obs=condition in ("A1", "A2"))
                    if condition in ("A0", "A1", "A2") else None)
                interp_weights = (interpreter_weights(cond_catalog)
                                  if condition in ("B1", "B2") else None)
                history = []
                practice_ce = math.log(base_practice.aggregate_ppl)
                # In-loop stage pool only (HZN-IR-08 finding 4): the
                # final independent evaluation is a fixed allowance
                # accounted SEPARATELY, so eval-cost variance cannot
                # distort the in-loop cognitive-efficiency comparison.
                stage_totals = {stage: 0.0 for stage in (
                    "proposal", "validation", "application", "quality",
                    "practice", "rollback")}
                control_attempts = 0
                failed = False
                # Adaptive budget matching: pool candidates per iteration,
                # iterations = ceil(steps/divisor) so total practice
                # evaluations can match the autonomous trajectory's budget.
                n_steps = args.steps
                if condition in ("B0", "B1", "B2"):
                    n_steps = max(1, math.ceil(args.steps / args.adaptive_step_divisor))
                # R3 compute-matched mode: the condition runs until its
                # MEASURED stage-seconds reach the shared budget (or the
                # safety cap / collapse rule fires). Intervention counts
                # are then an outcome, not an allocation.
                compute_budget = (args.compute_budget_seconds
                                  if args.budget_mode == "seconds" else None)
                step_limit = (args.max_steps if compute_budget is not None
                              else n_steps)
                for step in range(step_limit):
                    # HZN-IR-05 §3: enforce the budget BEFORE each step
                    # so every outcome class — applied, rejected,
                    # abstained — counts against the same allocation.
                    if _budget_exceeded(stage_totals, compute_budget):
                        break
                    step_times = {stage: 0.0 for stage in stage_totals}
                    if condition in ("A0", "A1", "A2"):
                        raw, proposal, step_times["proposal"] = proposer.propose(
                            seed, step, practice_ce, history)
                    elif condition in ("B0", "B1", "B2"):
                        start = time.perf_counter()
                        proposal, trial_s, attempts = adaptive_choice(
                            model, tok, ev, pm, seed, step, cond_catalog,
                            pool=args.adaptive_pool, weights=interp_weights,
                            current_ce=practice_ce)
                        step_times["proposal"] = time.perf_counter() - start
                        control_attempts += attempts
                        if proposal is None:
                            # No trial beat the unchanged checkpoint —
                            # the control abstains rather than commit to
                            # a harmful intervention (HZN-IR-05 §3).
                            proposal = {"edits": [], "abstain": True,
                                        "rationale": ("adaptive control: "
                                                      "no trial improved "
                                                      "practice CE")}
                        raw = json.dumps(proposal)
                    else:
                        start = time.perf_counter()
                        proposals = (random_control_proposals(catalog, arm_config, 1,
                                     seed * 1009 + step) if condition == "random"
                                     else guided_control_proposals(arm_config, args.steps)[step:step + 1])
                        proposal = proposals[0]
                        raw = json.dumps(proposal)
                        step_times["proposal"] = time.perf_counter() - start
                        control_attempts += 1
                    # Operational abstention (HZN-IR-03 R4): a deliberate
                    # no-change proposal consumes the step, leaves the
                    # checkpoint untouched and is recorded as an outcome.
                    if (proposal and proposal.get("abstain")
                            and not proposal.get("edits")):
                        pm.record_edit(_make_record(
                            arm_name, step, raw, proposal, "abstained",
                            seed=seed, gpu_seconds=sum(step_times.values()),
                            timings=step_times))
                        for stage, value in step_times.items():
                            stage_totals[stage] += value
                        history.append({"step": step, "abstained": True})
                        print(f"  step {step}: ABSTAIN", flush=True)
                        continue
                    before = practice_ce
                    start = time.perf_counter()
                    edits, err = pm.validate(proposal)
                    step_times["validation"] = time.perf_counter() - start
                    if not err and not edits:
                        err = "proposal has no valid edits"
                    if err:
                        pm.record_edit(_make_record(
                            arm_name, step, raw, proposal, "rejected",
                            rejection_reason=err, seed=seed,
                            gpu_seconds=sum(step_times.values()), timings=step_times))
                        for stage, value in step_times.items():
                            stage_totals[stage] += value
                        history.append({"step": step, "error": err})
                        print(f"  step {step}: REJECT {err}", flush=True)
                        continue

                    try:
                        descs, step_times["application"] = pm.apply(edits)
                        collapsed, mechs, qc_ppl, nan, step_times["quality"] = pm.quality_check()
                    except Exception:
                        ok, bad, _ = pm.rollback_all()
                        if not ok:
                            raise RuntimeError(f"rollback failed after exception: {bad}")
                        raise

                    if collapsed:
                        ok, bad, rb_s = pm.rollback_all()
                        step_times["rollback"] += rb_s
                        if not ok:
                            raise RuntimeError(f"collapse rollback failed: {bad}")
                        failed = True
                        pm.record_edit(_make_record(
                            arm_name, step, raw, proposal, "collapsed_rolled_back",
                            applied_edits=descs, practice_ppl_before=math.exp(before),
                            practice_ppl_after=None,
                            collapsed=True, collapse_mechanism=mechs,
                            gpu_seconds=sum(step_times.values()),
                            timings=step_times, seed=seed))
                        print(f"  step {step}: COLLAPSE {mechs} ppl={qc_ppl} "
                              f"rolled_back={ok}", flush=True)
                    else:
                        pr = ev.evaluate_practice(model, tok)
                        step_times["practice"] = pr.gpu_seconds
                        practice_ce = math.log(pr.aggregate_ppl)
                        prediction = proposal.get("predicted_practice_delta_ce")
                        history.append({"step": step, "practice_delta_ce": round(practice_ce - before, 6),
                                        "predicted_delta_ce": prediction})
                        pm.record_edit(_make_record(
                            arm_name, step, raw, proposal, "applied", applied_edits=descs,
                            practice_ppl_before=math.exp(before),
                            practice_ppl_after=pr.aggregate_ppl,
                            gpu_seconds=sum(step_times.values()),
                            timings=step_times, seed=seed))
                        print(f"  step {step}: applied {len(descs)} edits "
                              f"practice_ppl={pr.aggregate_ppl:.3f} "
                              f"qc_ppl={qc_ppl}", flush=True)
                    for stage, value in step_times.items():
                        stage_totals[stage] += value
                    if failed:
                        break

                n_steps_run = step + 1 if step_limit else 0
                pm.finalize()
                sel = ev.evaluate_selection(model, tok)
                reg = ev.evaluate_regression(model, tok)
                cap = evaluate_capability(model, tok)
                # Final-eval allowance — outside the in-loop budget pool
                # (HZN-IR-08 finding 4): recorded independently so the
                # comparison reports in-loop vs eval vs overshoot vs wall.
                final_eval_seconds = {
                    "selection": sel.gpu_seconds,
                    "regression": reg.gpu_seconds,
                    "capability": cap.gpu_seconds}

                reg_passed, reg_details = check_regression(
                    base_regression, reg, pilot_tolerance)
                cap_check = capability_regression(base_capability, cap)
                # GUARD-style paired-transition ledger: acquisition /
                # regression / retained / unresolved, per capability task
                # and per regression domain. Net deltas hide the two-rate
                # distinction this ledger preserves.
                ledger = transition_ledger(base_capability, cap)
                domain_ledger = {}
                for d, rr in reg.per_domain.items():
                    bce = math.log(base_regression.per_domain[d].aggregate_ppl)
                    dce = math.log(rr.aggregate_ppl) - bce
                    domain_ledger[d] = (
                        "acquisition" if dce < -0.005 else
                        "regression" if dce > 0.005 else "retained")
                ledger["regression_domains"] = domain_ledger
                sel_delta = sel.aggregate_ppl - base_selection.aggregate_ppl
                ok, bad, rb_s = pm.rollback_all()
                stage_totals["rollback"] += rb_s
                if not ok:
                    raise RuntimeError(f"final rollback failed: {bad}")
                elapsed = time.perf_counter() - candidate_start
                pm.candidate.total_gpu_seconds = sum(stage_totals.values())
                candidate_dict = pm.candidate_dict()
                candidate_dict.update({
                    "label": label,
                    "condition": condition,
                    "proposer_prompt_sha256": (proposer.prompt_version
                                               if proposer else None),
                    "selection_aggregate_ppl": sel.aggregate_ppl,
                    "selection_delta": round(sel_delta, 4),
                    "selection_delta_ce": sel.cross_entropy - base_selection.cross_entropy,
                    "selection_improved": sel_delta < 0,
                    "regression_passed": reg_passed,
                    "regression_details": reg_details,
                    "capability": cap.to_dict(),
                    "capability_check": cap_check,
                    "transition_ledger": ledger,
                    "n_steps_run": n_steps_run,
                    "budget": {
                        "mode": args.budget_mode,
                        "intervention_allocation": n_steps,
                        "compute_budget_seconds": compute_budget,
                        # In-loop measured compute — what the budget
                        # actually bounds (HZN-IR-08 finding 4).
                        "compute_seconds_used": round(
                            sum(stage_totals.values()), 2),
                        # Final independent evaluation — fixed allowance,
                        # excluded from the in-loop total by construction.
                        "final_eval_seconds": {
                            k: round(v, 2)
                            for k, v in final_eval_seconds.items()},
                        # A step always completes once started, so a
                        # modest overshoot above the budget is expected
                        # and recorded, never hidden (HZN-IR-05 §3).
                        "overshoot_seconds": (
                            round(max(0.0, sum(stage_totals.values())
                                      - compute_budget), 2)
                            if compute_budget is not None else None),
                        "pool": (args.adaptive_pool
                                 if condition in ("B0", "B1", "B2") else 1),
                        "stopping_rule": (
                            "collapse -> terminate; else "
                            + ("measured compute >= budget OR "
                               f"step=={args.max_steps}"
                               if compute_budget is not None
                               else "fixed step count")),
                    },
                    # Filled at report finalization: obs_seconds divided
                    # across ALL benefiting records so the per-record
                    # shares sum to the true shared cost (HZN-IR-04 R3).
                    "obs_amortized_seconds": None,
                    "shuffle_info": shuffle_info,
                    "stage_seconds": stage_totals,
                    "candidate_wall_seconds": elapsed,
                    "unattributed_wall_seconds": (
                        elapsed - sum(stage_totals.values())
                        - sum(final_eval_seconds.values())),
                    "control_proposal_attempts": control_attempts,
                    "rollback_verified": ok,
                    "failed_quality_guard": failed,
                })
                report["candidates"].append(candidate_dict)
                report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
                out.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"[{label}] DONE: selection={sel.aggregate_ppl:.3f} "
                      f"(delta {sel_delta:+.4f}) reg_passed={reg_passed} "
                      f"measured_s={pm.candidate.total_gpu_seconds:.1f} "
                      f"wall_s={elapsed:.1f}", flush=True)

    # R3 amortization (HZN-IR-04): divide the shared collection cost
    # across every candidate RECORD that benefits from it, so per-record
    # obs_amortized_seconds sums to exactly observation_seconds — the
    # previous per-candidate share overcounted by arms x seeds x steps.
    _OBS_FED = ("A1", "A2", "B1", "B2")
    benefiting = [c for c in report["candidates"]
                  if c.get("condition") in _OBS_FED]
    share = obs_seconds / len(benefiting) if benefiting else 0.0
    for c in report["candidates"]:
        c["obs_amortized_seconds"] = (round(share, 3)
                                    if c.get("condition") in _OBS_FED
                                    else 0.0)

    report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    out = RESULTS / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {out} ({out.stat().st_size} bytes) in "
          f"{report['elapsed_seconds']}s", flush=True)

    # Summary
    n_cand = len(report["candidates"])
    n_improved = sum(1 for c in report["candidates"] if c.get("selection_improved"))
    n_reg_pass = sum(1 for c in report["candidates"] if c.get("regression_passed"))
    print(f"[summary] {n_cand} candidates, {n_improved} improved selection, "
          f"{n_reg_pass} passed regression", flush=True)
    return 0


def _make_record(arm, step, raw, parsed, status, **kwargs):
    from jarvis_research.gate1_patch_manager import EditRecord
    return EditRecord(
        arm=arm, step=step, proposal_raw=raw,
        proposal_parsed=parsed, status=status, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
