#!/usr/bin/env python3
"""gate1_patch_manager.py — reversible weight-edit manager for Gate 1.

The patch manager is the trust boundary between the proposer's text
output and the actual model weights. It:

  - Validates edit proposals against the arm-specific schema (hard bounds)
  - Applies edits in-place under torch.no_grad()
  - Tracks every touched tensor for bit-exact rollback
  - Enforces cumulative-drift limits (max total perturbation per tensor)
  - Provides GPU-seconds accounting for every apply/restore operation
  - Records a full audit trail (parent checkpoint, proposal, seed, scores)

Two edit arms, both preregistered:

  Arm A — Sparse additive (mid/late layers only)
    Operation: param += delta (applied to `count` randomly selected elements)
    Layers: 11-31 only (Phase A: early layers are lethal)
    Families: recurrence, mlp, deltanet_proj, attention (no embed/lm_head/norm)
    |delta| <= 0.5 absolute AND <= 0.5 * tensor_std (whichever binds)
    count: 1-512 scalars per edit
    Cumulative drift: max |total_delta| per element <= 0.5

  Arm B — Bounded multiplicative attenuation
    Operation: param *= factor (dense, all elements)
    Layers: 0-31 (attenuation sweep showed this is safe within bounds)
    Families: mlp, deltanet_proj, attention (no embed/lm_head/norm)
    factor: 0.70-0.99 (attenuation sweep: x<0.70 risky on multi-layer; x>=0.99 trivial)
    Cumulative drift: cumulative factor product per tensor >= 0.50

Both arms enforce:
  - Post-edit NaN/Inf check on a forward pass
  - Post-edit perplexity + degeneracy check (the quality guard from Phase A)
  - Automatic rollback on collapse
  - torch.equal() verification of every restored tensor
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

MODEL_ID = "Qwen/Qwen3.5-4B"
DTYPE = torch.bfloat16

# Phase A collapse thresholds (carried forward for the quality guard).
PPL_RATIO_COLLAPSE = 5.0
PPL_ABS_COLLAPSE = 200.0
GEN_CHECK_TOKENS = 16
GEN_CHECK_PROMPT = "The capital of France is"
QUALITY_PROBE_TEXT = (
    "The capital of France is Paris. Water is made of hydrogen and oxygen. "
    "A week has seven days. The Earth orbits the Sun once per year."
)


@torch.no_grad()
def probe_ppl(model, tok, text: str = QUALITY_PROBE_TEXT) -> float:
    """Perplexity of a fixed probe on the current model state."""
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    out = model(ids)
    logits = out.logits.float()
    sl, lab = logits[:, :-1, :], ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        sl.reshape(-1, sl.size(-1)), lab.reshape(-1))
    return float(torch.exp(loss).item()) if torch.isfinite(loss) else float("inf")


@torch.no_grad()
def _tensor_std(p: torch.Tensor) -> float:
    """Std without materializing an fp32 copy of the tensor.

    ``.float().std()`` allocates 2x the bf16 size; for ~1 GB Arm B tensors
    that transient alone can exhaust VRAM. Accumulate sum/sumsq over
    chunks in fp32 so auxiliary memory stays O(chunk), not O(numel).
    """
    d = p.detach().reshape(-1)
    total = 0.0
    sumsq = 0.0
    chunk = 1 << 22  # ~4M elements per chunk
    n = d.numel()
    for start in range(0, n, chunk):
        part = d[start:start + chunk].float()
        total += float(part.sum().item())
        sumsq += float(part.pow(2).sum().item())
    mean = total / n
    var = max(sumsq / n - mean * mean, 0.0)
    return math.sqrt(var)


# Arm A bounds (sparse additive, mid/late layers).
ARM_A = {
    "name": "sparse_additive",
    "operation": "add_sparse",
    "min_layer": 11,
    "max_layer": 31,
    "max_delta_abs": 0.5,
    "max_delta_std_rel": 0.5,
    "min_count": 1,
    "max_count": 512,
    "max_cumulative_abs": 0.5,
    # Allowed family prefixes (matched against param name substrings).
    "allowed_families": [
        "linear_attn.dt_bias", "linear_attn.A_log",
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        "linear_attn.in_proj_b", "linear_attn.in_proj_a",
        "linear_attn.out_proj",
        "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ],
    "forbidden_substrings": [
        "embed_tokens", "lm_head", "layernorm", "model.norm",
        "conv1d",  # conv1d is lethal at L0; exclude from Arm A
    ],
}

# Arm B bounds (bounded multiplicative attenuation).
ARM_B = {
    "name": "bounded_attenuation",
    "operation": "scale_dense",
    "min_layer": 0,
    "max_layer": 31,
    "min_factor": 0.70,
    "max_factor": 0.99,
    "min_cumulative_product": 0.50,
    "allowed_families": [
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        "linear_attn.out_proj", "linear_attn.in_proj_a",
        "linear_attn.in_proj_b", "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.o_proj",
    ],
    "forbidden_substrings": [
        "embed_tokens", "lm_head", "layernorm", "model.norm",
        "conv1d", "dt_bias", "A_log",  # recurrence params: no clear attenuation behavior
    ],
}


@dataclass
class EditRecord:
    """Audit record for a single applied edit."""
    arm: str
    step: int
    proposal_raw: str  # the proposer's raw text output
    proposal_parsed: Optional[dict]  # parsed JSON, or None if unparseable
    status: str  # "applied" | "rejected" | "collapsed_rolled_back"
    rejection_reason: Optional[str] = None
    applied_edits: List[dict] = field(default_factory=list)
    practice_ppl_before: Optional[float] = None
    practice_ppl_after: Optional[float] = None
    collapsed: bool = False
    collapse_mechanism: Optional[List[str]] = None
    gpu_seconds: float = 0.0
    timings: dict = field(default_factory=dict)
    seed: Optional[int] = None


@dataclass
class CandidateRecord:
    """Full audit trail for one candidate (one proposer trajectory)."""
    arm: str
    seed: int
    parent_checkpoint_sha: str  # hash of model state before edits
    edits: List[EditRecord] = field(default_factory=list)
    n_applied: int = 0
    n_rejected: int = 0
    n_collapsed: int = 0
    final_applied_tensors: List[str] = field(default_factory=list)
    total_gpu_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "arm": self.arm, "seed": self.seed,
            "parent_checkpoint_sha": self.parent_checkpoint_sha,
            "n_applied": self.n_applied, "n_rejected": self.n_rejected,
            "n_collapsed": self.n_collapsed,
            "final_applied_tensors": self.final_applied_tensors,
            "total_gpu_seconds": round(self.total_gpu_seconds, 2),
            "edits": [asdict(e) for e in self.edits],
        }


def build_catalog(arm_config: dict, params: Dict[str, torch.Tensor]) -> List[dict]:
    """Build the complete legal edit catalog for an arm.

    Every proposer and control MUST draw from this identical list so all
    conditions search the same parameter space. Each entry carries enough
    information to make a schema-valid proposal without trial-and-error:
    the exact parameter name, layer, shape, element count, tensor std, and
    the legal magnitude/count ranges for this arm.
    """
    arm = arm_config
    catalog = []
    for name in sorted(params.keys()):
        parts = name.split(".")
        li = None
        if "layers" in parts:
            i = parts.index("layers")
            if i + 1 < len(parts):
                try:
                    li = int(parts[i + 1])
                except ValueError:
                    li = None
        if li is None:
            continue
        if li < arm["min_layer"] or li > arm["max_layer"]:
            continue
        forbidden = any(fs in name for fs in arm["forbidden_substrings"])
        if forbidden:
            continue
        family = next((fam for fam in arm["allowed_families"] if fam in name), None)
        if family is None:
            continue
        p = params[name]
        std = _tensor_std(p)
        entry = {
            "param_id": len(catalog),
            "param": name,
            "family": family,
            "layer": li,
            "shape": list(p.shape),
            "numel": int(p.numel()),
            "std": round(std, 6),
        }
        if arm["name"] == "sparse_additive":
            max_abs = arm["max_delta_abs"]
            max_std = arm["max_delta_std_rel"] * std if std > 0 else max_abs
            entry["mag_min"] = -min(max_abs, max_std)
            entry["mag_max"] = min(max_abs, max_std)
            entry["count_min"] = arm["min_count"]
            entry["count_max"] = min(arm["max_count"], int(p.numel()))
        elif arm["name"] == "bounded_attenuation":
            entry["mag_min"] = arm["min_factor"]
            entry["mag_max"] = arm["max_factor"]
        catalog.append(entry)
    return catalog


class PatchManager:
    """Manages reversible weight edits for a Gate 1 candidate.

    The PatchManager holds the model, tokenizer, and all bookkeeping needed
    to apply, evaluate, and roll back edits for a single candidate trajectory.
    It is constructed once per candidate and discarded after the candidate's
    final evaluation.
    """

    def __init__(self, model, tok, arm_config: dict, seed: int,
                 quality_baseline_ppl: Optional[float] = None):
        self.model = model
        self.tok = tok
        self.arm = arm_config
        self.seed = seed
        self.params = {n: p for n, p in model.named_parameters()}
        self.catalog = build_catalog(arm_config, self.params)
        self._catalog_by_id = {e["param_id"]: e for e in self.catalog}
        self._catalog_by_name = {e["param"]: e for e in self.catalog}
        # Measured perplexity of the fixed quality-check probe on the
        # UNMODIFIED pinned checkpoint. Required: the collapse guard must be
        # calibrated to the actual model, not a hard-coded constant.
        self.quality_baseline_ppl = quality_baseline_ppl
        # Originals: snapshot of every tensor BEFORE any edit, for final
        # rollback. Kept on CPU: GPU-side clones of large Arm B tensors can
        # exhaust VRAM within a single candidate.
        self.originals: Dict[str, torch.Tensor] = {}
        # Touched: names of tensors that have been edited (for restore + drift tracking).
        self.touched: set = set()
        # Cumulative drift tracking: per-tensor sum of absolute deltas (Arm A)
        # or product of factors (Arm B).
        self.cumulative_delta: Dict[str, float] = {}  # Arm A: sum |delta|; Arm B: product factors
        self.candidate = CandidateRecord(
            arm=arm_config["name"], seed=seed,
            parent_checkpoint_sha=self._checkpoint_sha(),
        )
        self._cpu_gen = torch.Generator().manual_seed(seed)

    def _checkpoint_sha(self) -> str:
        """Use pinned upstream revision; pilot records full local shard hashes."""
        revision = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        return revision or "unverified-local-revision"

    def _classify_family(self, name: str) -> Optional[str]:
        """Check if a param name matches an allowed family. Returns family or None."""
        for forbidden in self.arm["forbidden_substrings"]:
            if forbidden in name:
                return None
        for fam in self.arm["allowed_families"]:
            if fam in name:
                return fam
        return None

    def _layer_idx(self, name: str) -> Optional[int]:
        parts = name.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            if i + 1 < len(parts):
                return int(parts[i + 1])
        return None

    def validate(self, proposal: dict) -> Tuple[Optional[List[dict]], Optional[str]]:
        """Validate a proposal against the arm-specific schema. Returns (edits, error)."""
        if not isinstance(proposal, dict):
            return None, "proposal not a dict"
        edits_list = proposal.get("edits")
        if not isinstance(edits_list, list) or not edits_list:
            return None, "missing or empty 'edits' list"
        if len(edits_list) > 8:
            return None, "batch too large (>8)"

        validated = []
        prospective = self.cumulative_delta.copy()
        batch_scalars = 0
        sparse_names = set()
        for i, e in enumerate(edits_list):
            if not isinstance(e, dict):
                return None, f"edit {i}: not a dict"

            # Resolve the target: accept either an exact param name or a
            # catalog param_id. Both refer to the SAME legal catalog that
            # every proposer and control draws from.
            param_name = e.get("param")
            param_id = e.get("param_id")
            if param_id is not None:
                entry = self._catalog_by_id.get(param_id)
                if entry is None:
                    return None, f"edit {i}: unknown param_id {param_id!r}"
                if param_name is not None and param_name != entry["param"]:
                    return None, (f"edit {i}: param_id {param_id} resolves to "
                                  f"{entry['param']!r}, not {param_name!r}")
                param_name = entry["param"]
            if not isinstance(param_name, str) or param_name not in self.params:
                return None, f"edit {i}: unknown param {param_name!r}"

            # Family check
            family = self._classify_family(param_name)
            if family is None:
                return None, (f"edit {i}: param {param_name!r} is not on the "
                              f"allowed family list for arm {self.arm['name']}")

            # Layer check
            li = self._layer_idx(param_name)
            if li is not None:
                if li < self.arm["min_layer"] or li > self.arm["max_layer"]:
                    return None, (f"edit {i}: layer {li} out of bounds "
                                  f"[{self.arm['min_layer']}, {self.arm['max_layer']}]")

            op = e.get("operation")
            expected_op = self.arm["operation"]
            if op != expected_op:
                return None, (f"edit {i}: operation {op!r} != arm operation "
                              f"{expected_op!r}")

            # Magnitude check
            try:
                magnitude = float(e.get("magnitude"))
            except (TypeError, ValueError):
                return None, f"edit {i}: magnitude not numeric"

            if not math.isfinite(magnitude):
                return None, f"edit {i}: magnitude not finite"

            if self.arm["name"] == "sparse_additive":
                if abs(magnitude) > self.arm["max_delta_abs"]:
                    return None, (f"edit {i}: |delta|={abs(magnitude):.4f} > "
                                  f"max {self.arm['max_delta_abs']}")
                std = _tensor_std(self.params[param_name])
                if std > 0 and abs(magnitude) / std > self.arm["max_delta_std_rel"]:
                    return None, (f"edit {i}: |delta|/std={abs(magnitude)/std:.4f} > "
                                  f"max {self.arm['max_delta_std_rel']}")

                count = e.get("count")
                if not isinstance(count, int) or isinstance(count, bool):
                    return None, f"edit {i}: count not an integer"
                if count < self.arm["min_count"] or count > self.arm["max_count"]:
                    return None, (f"edit {i}: count {count} out of bounds "
                                  f"[{self.arm['min_count']}, {self.arm['max_count']}]")
                if count > self.params[param_name].numel():
                    return None, (f"edit {i}: count {count} > numel "
                                  f"{self.params[param_name].numel()}")

                # Duplicate tensor edits could hit overlapping sampled indices.
                # Reject rather than silently accepting an unknown aggregate.
                if param_name in sparse_names:
                    return None, f"edit {i}: duplicate sparse tensor in batch"
                sparse_names.add(param_name)
                batch_scalars += count
                if batch_scalars > self.arm["max_count"]:
                    return None, f"edit {i}: aggregate sparse count exceeds 512"
                current_drift = prospective.get(param_name, 0.0)
                if current_drift + abs(magnitude) > self.arm["max_cumulative_abs"]:
                    return None, (f"edit {i}: cumulative |delta| would exceed "
                                  f"{self.arm['max_cumulative_abs']} for {param_name}")
                prospective[param_name] = current_drift + abs(magnitude)

                validated.append({
                    "name": param_name, "operation": op,
                    "magnitude": magnitude, "count": count,
                    "family": family, "layer": li,
                })

            elif self.arm["name"] == "bounded_attenuation":
                if magnitude < self.arm["min_factor"] or magnitude > self.arm["max_factor"]:
                    return None, (f"edit {i}: factor {magnitude} out of bounds "
                                  f"[{self.arm['min_factor']}, {self.arm['max_factor']}]")

                # Cumulative product check
                current_product = prospective.get(param_name, 1.0)
                new_product = current_product * magnitude
                if new_product < self.arm["min_cumulative_product"]:
                    return None, (f"edit {i}: cumulative factor product {new_product:.4f} < "
                                  f"min {self.arm['min_cumulative_product']} for {param_name}")
                prospective[param_name] = new_product

                validated.append({
                    "name": param_name, "operation": op,
                    "magnitude": magnitude, "count": None,
                    "family": family, "layer": li,
                })
            else:
                return None, f"unknown arm {self.arm['name']}"

        return validated, None

    def apply(self, edits: List[dict]) -> Tuple[List[dict], float]:
        """Apply validated edits in-place. Returns (descriptions, gpu_seconds)."""
        t0 = time.perf_counter()
        descs = []
        for edit in edits:
            name = edit["name"]
            # Snapshot at first touch
            if name not in self.touched:
                self.touched.add(name)
                self.originals[name] = self.params[name].detach().to("cpu")
                if self.arm["name"] == "bounded_attenuation":
                    self.cumulative_delta[name] = 1.0
                else:
                    self.cumulative_delta[name] = 0.0

            p = self.params[name]
            with torch.no_grad():
                std = _tensor_std(p)
                if self.arm["name"] == "sparse_additive":
                    idx = torch.randperm(p.numel(), generator=self._cpu_gen)[:edit["count"]].to(p.device)
                    flat = p.view(-1)
                    flat[idx] += edit["magnitude"]
                    self.cumulative_delta[name] += abs(edit["magnitude"])
                    elements = edit["count"]
                elif self.arm["name"] == "bounded_attenuation":
                    p.view(-1).mul_(edit["magnitude"])
                    self.cumulative_delta[name] *= edit["magnitude"]
                    elements = p.numel()

            descs.append({
                "param": name,
                "family": edit["family"],
                "layer": edit["layer"],
                "operation": edit["operation"],
                "magnitude": edit["magnitude"],
                "scope": "sparse" if self.arm["name"] == "sparse_additive" else "dense",
                "count": edit["count"],
                "tensor_std": round(std, 6),
                "elements_touched": elements,
                "cumulative_drift": round(self.cumulative_delta[name], 4),
            })
        gpu_s = time.perf_counter() - t0
        return descs, gpu_s

    def quality_check(self) -> Tuple[bool, List[str], float, float, float]:
        """Post-edit quality guard. Returns (collapsed, mechanisms, ppl, nan, gpu_s).

        Uses the Phase A collapse criteria: NaN/Inf, degenerate generation,
        or perplexity explosion (>5x or >200 absolute).
        """
        t0 = time.perf_counter()
        ids = self.tok(QUALITY_PROBE_TEXT, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = self.model(ids)
            logits = out.logits.float()
            nan = bool(torch.isnan(logits).any() or torch.isinf(logits).any())
            sl, lab = logits[:, :-1, :], ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.size(-1)), lab.reshape(-1))
            ppl = float(torch.exp(loss).item()) if torch.isfinite(loss) else float("inf")

        # Degenerate generation check
        deg = False
        if math.isfinite(ppl):
            with torch.no_grad():
                gen_ids = self.tok(GEN_CHECK_PROMPT, return_tensors="pt").input_ids.to("cuda")
                gen_out = self.model.generate(gen_ids, max_new_tokens=GEN_CHECK_TOKENS, do_sample=False)
                toks = gen_out[0, gen_ids.shape[1]:].tolist()
                deg = (not toks) or len(set(toks)) <= 1

        # Collapse criteria: ratio against the MEASURED baseline perplexity
        # of this pinned checkpoint on this fixed probe — never a constant
        # carried over from a different experiment.
        if self.quality_baseline_ppl is None or self.quality_baseline_ppl <= 0:
            raise RuntimeError(
                "quality_baseline_ppl not set: measure it on the unmodified "
                "checkpoint before running candidates")
        base_ppl = self.quality_baseline_ppl
        ratio = (ppl / base_ppl) if math.isfinite(ppl) and base_ppl > 0 else float("inf")
        collapsed = nan or deg or ratio > PPL_RATIO_COLLAPSE or ppl > PPL_ABS_COLLAPSE
        mechanisms = [m for m, c in
                      (("nan_inf", nan), ("degenerate", deg),
                       ("ppl_explosion", ratio > PPL_RATIO_COLLAPSE or ppl > PPL_ABS_COLLAPSE))
                      if c]
        gpu_s = time.perf_counter() - t0
        return collapsed, mechanisms, round(ppl, 3) if math.isfinite(ppl) else "inf", nan, gpu_s

    def rollback_all(self) -> Tuple[bool, Optional[str], float]:
        """Restore ALL touched tensors to original. Returns (ok, bad_name, gpu_s).

        Verified by torch.equal() on every tensor (reviewer v2 fix 3).
        """
        t0 = time.perf_counter()
        for name, orig in self.originals.items():
            with torch.no_grad():
                self.params[name].view(-1).copy_(orig.view(-1))
        torch.cuda.synchronize()
        # Verify
        for name, orig in self.originals.items():
            if not torch.equal(self.params[name].detach().cpu(), orig):
                gpu_s = time.perf_counter() - t0
                return False, name, gpu_s
        self.touched.clear()
        self.originals.clear()
        self.cumulative_delta.clear()
        gpu_s = time.perf_counter() - t0
        return True, None, gpu_s

    def record_edit(self, record: EditRecord):
        """Add an edit record to the candidate audit trail."""
        self.candidate.edits.append(record)
        if record.status == "applied":
            self.candidate.n_applied += 1
        elif record.status == "rejected":
            self.candidate.n_rejected += 1
        elif record.status == "collapsed_rolled_back":
            self.candidate.n_collapsed += 1
        self.candidate.total_gpu_seconds += record.gpu_seconds

    def finalize(self):
        """Mark the candidate as complete. Update final applied tensors list."""
        self.candidate.final_applied_tensors = sorted(self.touched)

    def candidate_dict(self) -> dict:
        return self.candidate.to_dict()


def make_arm_config(arm: str) -> dict:
    """Return the arm configuration dict for 'A' or 'B'."""
    if arm == "A":
        return ARM_A
    elif arm == "B":
        return ARM_B
    else:
        raise ValueError(f"unknown arm {arm!r}")
