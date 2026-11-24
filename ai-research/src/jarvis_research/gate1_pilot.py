#!/usr/bin/env python3
"""gate1_pilot.py — Gate 1 exploratory engineering pilot.

This script runs the FULL Gate 1 acceptance pipeline on a small scale to:
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
  8. COMPARE CONTROLS: matched random + guided search at equal budget
  9. AUDIT: full trail (parent sha, proposals, seeds, scores, GPU-seconds)

Arms:
  A: sparse additive (mid/late layers, 11-31)
  B: bounded attenuation (0.70-0.99, all layers)

Controls (per arm):
  random: matched random selection within the same arm's allowed surface,
          same number of edit steps, same evaluation budget. Matched by
          tensor family and parameter count where possible.
  guided: fixed algorithm using the same candidate and compute budget.
          For Arm A: gradient sign guided (compute loss gradient on practice
          set, apply small step in the negative direction). For Arm B:
          uniform attenuation at the midpoint factor (0.85).

Usage:
    .venv-opt\\Scripts\\python.exe src/jarvis_research/gate1_pilot.py
        [--seeds 1 2 3] [--steps 8] [--smoke] [--out <filename>]
"""

from __future__ import annotations

import argparse
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
    arm_config: dict,
    params: Dict[str, torch.Tensor],
    n_steps: int,
    seed: int,
) -> List[dict]:
    """Generate n_steps random valid proposals for the given arm.

    Matched by: same operation, same allowed families, same layer range,
    same magnitude bounds. Random selection of tensor + magnitude within
    the arm's hard bounds. This is the matched random control (reviewer
    v2: "separately matched random-search control for each edit arm").
    """
    g = torch.Generator().manual_seed(seed * 31 + 17)
    proposals = []

    # Build the list of allowed params for this arm.
    allowed = []
    for name in sorted(params.keys()):
        li = _layer_idx(name)
        if li is None:
            continue
        if li < arm_config["min_layer"] or li > arm_config["max_layer"]:
            continue
        # Family check
        forbidden = any(fs in name for fs in arm_config["forbidden_substrings"])
        if forbidden:
            continue
        family_match = any(fam in name for fam in arm_config["allowed_families"])
        if not family_match:
            continue
        allowed.append(name)

    for step in range(n_steps):
        if arm_config["name"] == "sparse_additive":
            # Random tensor, random delta within bounds, random count
            pname = allowed[torch.randint(len(allowed), (1,), generator=g).item()]
            p = params[pname]
            std = float(p.float().std().item())
            # Delta bounded by min(0.5 abs, 0.5*std)
            max_abs = arm_config["max_delta_abs"]
            max_std = arm_config["max_delta_std_rel"] * std if std > 0 else max_abs
            max_delta = min(max_abs, max_std)
            # uniform in [-max_delta, max_delta]: 2*rand - 1 then scale
            r = float(torch.rand(1, generator=g).item())
            magnitude = (2.0 * r - 1.0) * max_delta
            max_count = min(arm_config["max_count"], p.numel())
            count = int(torch.randint(1, max_count + 1, (1,), generator=g).item())
            proposals.append({
                "edits": [{"param": pname, "operation": "add_sparse",
                           "magnitude": magnitude, "count": count}],
                "rationale": "random control",
                "predicted_effect": "random",
            })
        elif arm_config["name"] == "bounded_attenuation":
            # Random tensor, random factor within [0.70, 0.99]
            pname = allowed[torch.randint(len(allowed), (1,), generator=g).item()]
            lo, hi = arm_config["min_factor"], arm_config["max_factor"]
            r = float(torch.rand(1, generator=g).item())
            magnitude = lo + r * (hi - lo)
            proposals.append({
                "edits": [{"param": pname, "operation": "scale_dense",
                           "magnitude": magnitude}],
                "rationale": "random control",
                "predicted_effect": "random",
            })
    return proposals


def guided_control_proposals(
    arm_config: dict,
    n_steps: int,
) -> List[dict]:
    """Generate n_steps guided (fixed-algorithm) proposals for the given arm.

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
                "rationale": "guided: fixed mid-layer MLP heuristic",
                "predicted_effect": "guided",
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
                "rationale": "guided: fixed midpoint attenuation on mid-layer",
                "predicted_effect": "guided",
            })
    return proposals


def _layer_idx(name: str) -> Optional[int]:
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        if i + 1 < len(parts):
            return int(parts[i + 1])
    return None


# ---------------------------------------------------------------------------
# Main pilot runner
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=8,
                    help="edit steps per candidate (small for pilot)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 2 steps, skip controls")
    ap.add_argument("--out", default="gate1_pilot.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jarvis_research.gate1_evaluator import Evaluator, check_regression
    from jarvis_research.gate1_patch_manager import PatchManager, make_arm_config

    data_dir = AI_RESEARCH / "data" / "gate1_eval"
    ev = Evaluator.load(
        data_dir / "practice.json",
        data_dir / "selection.json",
        data_dir / "regression.json",
    )

    t0 = time.perf_counter()
    print(f"[load] {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map={"": "cuda"})
    model.eval()
    torch.cuda.synchronize()
    params = {n: p for n, p in model.named_parameters()}
    print(f"[load] done in {time.perf_counter()-t0:.1f}s, {len(params)} params",
          flush=True)

    # Baseline evaluation on ALL partitions
    print("\n[baseline] evaluating all partitions ...", flush=True)
    base_practice = ev.evaluate_practice(model, tok)
    base_selection = ev.evaluate_selection(model, tok)
    base_regression = ev.evaluate_regression(model, tok)
    print(f"[baseline] practice={base_practice.aggregate_ppl:.3f} "
          f"selection={base_selection.aggregate_ppl:.3f} "
          f"({base_regression.n_domains} domains)", flush=True)
    for d, r in base_regression.per_domain.items():
        print(f"  {d:15s} {r.aggregate_ppl:.3f}", flush=True)

    report = {
        "gate": "gate1_engineering_pilot",
        "model": MODEL_ID, "git_sha": _git_sha(),
        "purpose": ("EXPLORATORY engineering pilot. NOT confirmatory. "
                    "Verify pipeline, estimate variance + cost, identify failures."),
        "n_seeds": len(args.seeds),
        "n_steps": args.steps,
        "baseline": {
            "practice": asdict(base_practice),
            "selection": asdict(base_selection),
            "regression": {
                "per_domain": {d: asdict(r) for d, r in base_regression.per_domain.items()},
                "gpu_seconds": base_regression.gpu_seconds,
            },
        },
        "candidates": [],
    }

    # Regression tolerance (preliminary, for pilot only — will be set
    # formally when confirmatory protocol is frozen). Using 0.05 nats
    # CE per domain as a placeholder.
    pilot_tolerance = {d: 0.05 for d in base_regression.per_domain}

    arms = ["A", "B"] if not args.smoke else ["A"]
    conditions = ["autonomous"] if args.smoke else ["autonomous", "random", "guided"]
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    steps = 2 if args.smoke else args.steps

    for arm_name in arms:
        arm_config = make_arm_config(arm_name)
        for condition in conditions:
            for seed in seeds:
                label = f"arm{arm_name}_{condition}_s{seed}"
                print(f"\n[{label}] starting ...", flush=True)
                pm = PatchManager(model, tok, arm_config, seed=seed)
                step_gpu_s = 0.0

                # Generate proposals for this condition
                if condition == "random":
                    proposals = random_control_proposals(
                        arm_config, params, steps, seed)
                elif condition == "guided":
                    proposals = guided_control_proposals(arm_config, steps)
                else:  # autonomous — use a simple heuristic proposer for the pilot
                    # For the engineering pilot, the autonomous proposer is a
                    # placeholder that generates valid proposals within bounds.
                    # The real autonomous proposer (gate1_proposer.py) will be
                    # built after the pipeline is verified.
                    proposals = _pilot_autonomous_proposals(
                        arm_config, params, steps, seed)

                for step in range(steps):
                    if step >= len(proposals):
                        break
                    proposal = proposals[step]

                    # Validate
                    edits, err = pm.validate(proposal)
                    if err:
                        pm.record_edit(_make_record(
                            arm_name, step, json.dumps(proposal), proposal,
                            "rejected", rejection_reason=err))
                        print(f"  step {step}: REJECT {err}", flush=True)
                        continue

                    # Apply
                    descs, apply_s = pm.apply(edits)
                    step_gpu_s += apply_s

                    # Quality check
                    collapsed, mechs, qc_ppl, nan, qc_s = pm.quality_check()
                    step_gpu_s += qc_s

                    if collapsed:
                        # Rollback
                        ok, bad, rb_s = pm.rollback_all()
                        step_gpu_s += rb_s
                        pm.record_edit(_make_record(
                            arm_name, step, json.dumps(proposal), proposal,
                            "collapsed_rolled_back",
                            applied_edits=descs,
                            practice_ppl_before=None,
                            practice_ppl_after=qc_ppl,
                            collapsed=True, collapse_mechanism=mechs,
                            gpu_seconds=step_gpu_s, seed=seed))
                        print(f"  step {step}: COLLAPSE {mechs} ppl={qc_ppl} "
                              f"rolled_back={ok}", flush=True)
                        break
                    else:
                        # Evaluate practice (feedback signal)
                        pr = ev.evaluate_practice(model, tok)
                        step_gpu_s += pr.gpu_seconds
                        pm.record_edit(_make_record(
                            arm_name, step, json.dumps(proposal), proposal,
                            "applied",
                            applied_edits=descs,
                            practice_ppl_before=None,
                            practice_ppl_after=pr.aggregate_ppl,
                            gpu_seconds=step_gpu_s, seed=seed))
                        print(f"  step {step}: applied {len(descs)} edits "
                              f"practice_ppl={pr.aggregate_ppl:.3f} "
                              f"qc_ppl={qc_ppl}", flush=True)

                # Final evaluation: selection + regression (after all steps)
                pm.finalize()
                sel = ev.evaluate_selection(model, tok)
                reg = ev.evaluate_regression(model, tok)
                pm.candidate.total_gpu_seconds += sel.gpu_seconds + reg.gpu_seconds

                # Regression check
                reg_passed, reg_details = check_regression(
                    base_regression, reg, pilot_tolerance)

                # Selection delta
                sel_delta = sel.aggregate_ppl - base_selection.aggregate_ppl

                candidate_dict = pm.candidate_dict()
                candidate_dict.update({
                    "condition": condition,
                    "selection_aggregate_ppl": sel.aggregate_ppl,
                    "selection_delta": round(sel_delta, 4),
                    "selection_improved": sel_delta < 0,
                    "regression_passed": reg_passed,
                    "regression_details": reg_details,
                    "selection_gpu_seconds": sel.gpu_seconds,
                    "regression_gpu_seconds": reg.gpu_seconds,
                })
                report["candidates"].append(candidate_dict)
                print(f"[{label}] DONE: selection={sel.aggregate_ppl:.3f} "
                      f"(delta {sel_delta:+.4f}) reg_passed={reg_passed} "
                      f"gpu_s={pm.candidate.total_gpu_seconds:.1f}", flush=True)

                # Rollback for next candidate
                ok, bad, rb_s = pm.rollback_all()
                if not ok:
                    print(f"  WARNING: final rollback mismatch on {bad}", flush=True)

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


def _pilot_autonomous_proposals(arm_config, params, n_steps, seed):
    """Placeholder autonomous proposer for the engineering pilot.

    This generates valid proposals within bounds using a simple heuristic:
    for Arm A, small positive deltas on mid-layer MLP; for Arm B, midpoint
    attenuation on mid-layer projections. The REAL autonomous proposer
    (gate1_proposer.py, to be built) will use the model itself to generate
    proposals from practice feedback.
    """
    g = torch.Generator().manual_seed(seed * 101 + 23)
    proposals = []
    target_layer = 16
    for step in range(n_steps):
        if arm_config["name"] == "sparse_additive":
            mlp_types = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
            pname = f"model.layers.{target_layer}.{mlp_types[step % 3]}"
            proposals.append({
                "edits": [{"param": pname, "operation": "add_sparse",
                           "magnitude": 0.001, "count": 256}],
                "rationale": "pilot autonomous: small mid-layer MLP delta",
                "predicted_effect": "small improvement",
            })
        elif arm_config["name"] == "bounded_attenuation":
            proj_types = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
            pname = f"model.layers.{target_layer}.{proj_types[step % 3]}"
            r = float(torch.rand(1, generator=g).item())
            factor = 0.80 + r * 0.15  # uniform in [0.80, 0.95]
            proposals.append({
                "edits": [{"param": pname, "operation": "scale_dense",
                           "magnitude": factor}],
                "rationale": "pilot autonomous: mid-layer attenuation",
                "predicted_effect": "stabilization",
            })
    return proposals


if __name__ == "__main__":
    raise SystemExit(main())
