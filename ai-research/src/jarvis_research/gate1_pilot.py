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


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=AI_RESEARCH, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Control generators. Each produces a sequence of proposals for a given arm,
# matched to the arm's allowed surface and the proposer's step count.
# ---------------------------------------------------------------------------

def random_control_proposals(
    catalog: List[dict],
    arm_config: dict,
    n_steps: int,
    seed: int,
) -> List[dict]:
    """Generate n_steps random proposals from the SHARED legal catalog.

    Gate 1.1: draws from the identical catalog the proposer sees, so the
    random control searches the same parameter space. Magnitude and count
    are drawn uniformly from each entry's legal range — the same bounds
    shown to the model. This matches the SEARCH SPACE and proposal budget;
    it does not match the model's parameter-selection distribution (that is
    the quantity under test).
    """
    import random as _random
    g = _random.Random(seed * 31 + 17)
    proposals = []
    if not catalog:
        return proposals

    for step in range(n_steps):
        entry = catalog[g.randrange(len(catalog))]
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
                    pool: int = 2):
    """Compare trial edits on practice; restore after each trial.

    Evaluates `pool` candidates per iteration. The caller sets the iteration
    count so total practice evaluations match the autonomous trajectory's
    budget.
    """
    from jarvis_research.gate1_patch_manager import PatchManager
    arm_config = patch_manager.arm
    choices = random_control_proposals(catalog, arm_config, pool,
                                       seed * 1009 + step)
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
    return best, trial_seconds, len(choices)


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
                    choices=("autonomous", "random", "fixed_heuristic", "adaptive"),
                    default=["autonomous", "random", "fixed_heuristic", "adaptive"])
    ap.add_argument("--adaptive-pool", type=int, default=2,
                    help="candidate edits evaluated per adaptive step")
    ap.add_argument("--adaptive-step-divisor", type=int, default=1,
                    help="adaptive iterations = ceil(steps/divisor); use 2 for "
                         "practice-evaluation-budget matching vs autonomous")
    ap.add_argument("--out", default="gate1_pilot_v2.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip candidates already present in --out")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jarvis_research.gate1_evaluator import Evaluator, check_regression
    from jarvis_research.gate1_patch_manager import (
        PatchManager, make_arm_config, build_catalog, probe_ppl)
    from jarvis_research.gate1_proposer import SelfProposer, PROMPT_VERSION
    from jarvis_research.gate1_capability import (
        evaluate_capability, capability_regression)

    if args.steps < 1:
        ap.error("--steps must be positive")
    if args.adaptive_pool < 1 or args.adaptive_step_divisor < 1:
        ap.error("adaptive knobs must be positive")
    data_dir = AI_RESEARCH / "data" / "gate1_eval_v3"
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
        "candidates": [],
    }
    out = RESULTS / args.out
    completed = set()
    if args.resume and out.exists():
        prior = json.loads(out.read_text(encoding="utf-8"))
        report["baseline"] = prior.get("baseline", report["baseline"])
        report["candidates"] = prior.get("candidates", [])
        completed = {c.get("label") for c in report["candidates"]}
        print(f"[resume] {len(completed)} candidates already complete", flush=True)

    # Regression tolerance (preliminary, for pilot only — will be set
    # formally when confirmatory protocol is frozen). Using 0.05 nats
    # CE per domain as a placeholder.
    pilot_tolerance = {d: 0.05 for d in base_regression.per_domain}

    for arm_name in args.arms:
        arm_config = make_arm_config(arm_name)
        catalog = build_catalog(arm_config, params)
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
                proposer = (SelfProposer(model, tok, pm, args.max_new)
                            if condition == "autonomous" else None)
                history = []
                practice_ce = math.log(base_practice.aggregate_ppl)
                stage_totals = {stage: 0.0 for stage in (
                    "proposal", "validation", "application", "quality",
                    "practice", "selection", "regression", "capability",
                    "rollback")}
                control_attempts = 0
                failed = False
                # Adaptive budget matching: pool candidates per iteration,
                # iterations = ceil(steps/divisor) so total practice
                # evaluations can match the autonomous trajectory's budget.
                n_steps = args.steps
                if condition == "adaptive":
                    n_steps = max(1, math.ceil(args.steps / args.adaptive_step_divisor))
                for step in range(n_steps):
                    step_times = {stage: 0.0 for stage in stage_totals}
                    if condition == "autonomous":
                        raw, proposal, step_times["proposal"] = proposer.propose(
                            seed, step, practice_ce, history)
                    elif condition == "adaptive":
                        start = time.perf_counter()
                        proposal, trial_s, attempts = adaptive_choice(
                            model, tok, ev, pm, seed, step, catalog,
                            pool=args.adaptive_pool)
                        step_times["proposal"] = time.perf_counter() - start
                        control_attempts += attempts
                        raw = json.dumps(proposal) if proposal else "null"
                    else:
                        start = time.perf_counter()
                        proposals = (random_control_proposals(catalog, arm_config, 1,
                                     seed * 1009 + step) if condition == "random"
                                     else guided_control_proposals(arm_config, args.steps)[step:step + 1])
                        proposal = proposals[0]
                        raw = json.dumps(proposal)
                        step_times["proposal"] = time.perf_counter() - start
                        control_attempts += 1
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

                pm.finalize()
                sel = ev.evaluate_selection(model, tok)
                reg = ev.evaluate_regression(model, tok)
                cap = evaluate_capability(model, tok)
                stage_totals["selection"] += sel.gpu_seconds
                stage_totals["regression"] += reg.gpu_seconds
                stage_totals["capability"] += cap.gpu_seconds

                reg_passed, reg_details = check_regression(
                    base_regression, reg, pilot_tolerance)
                cap_check = capability_regression(base_capability, cap)
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
                    "proposer_prompt_sha256": (PROMPT_VERSION
                                               if condition == "autonomous"
                                               else None),
                    "selection_aggregate_ppl": sel.aggregate_ppl,
                    "selection_delta": round(sel_delta, 4),
                    "selection_delta_ce": sel.cross_entropy - base_selection.cross_entropy,
                    "selection_improved": sel_delta < 0,
                    "regression_passed": reg_passed,
                    "regression_details": reg_details,
                    "capability": cap.to_dict(),
                    "capability_check": cap_check,
                    "n_steps_run": n_steps,
                    "stage_seconds": stage_totals,
                    "candidate_wall_seconds": elapsed,
                    "unattributed_wall_seconds": elapsed - sum(stage_totals.values()),
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
