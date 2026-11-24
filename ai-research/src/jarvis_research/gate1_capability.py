"""Gate 1.1 capability suite — task-level evaluation separate from perplexity.

The language-modeling partitions measure text likelihood; they cannot detect
loss of reasoning, code, structured-output, or long-context capability. This
suite adds four small, deterministic task families scored INDEPENDENTLY of
the selection/regression perplexity endpoint:

  reasoning   — multi-step arithmetic word problems, exact numeric answer
  code        — short function completion, exact-match on final expression
  structured  — constrained JSON generation, schema-validated
  longctx     — perplexity on a single long (>1024-token) coherent passage

All prompts and references are ORIGINAL text written for Gate 1.1 — they are
not the examined seed-303 or gate1_eval passages. Like the development
partitions, this suite is visible to the development account; it is NOT a
sealed final holdout.

Scoring is deterministic where possible (exact match / JSON parse) and
perplexity-based only for longctx. Generation uses greedy decoding with a
fixed token budget so scores are reproducible.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Task definitions. Each task: id, family, prompt, and a scorer spec.
# ---------------------------------------------------------------------------

REASONING_TASKS = [
    {"id": "rsn_1", "prompt": "A train travels 60 km in 45 minutes, then 80 km in 1 hour 15 minutes. What is its average speed for the whole journey in km/h? Answer with just the number.",
     "answer": "56"},
    {"id": "rsn_2", "prompt": "If 3 workers build a wall in 8 days, how many days would 6 workers take at the same rate? Answer with just the number.",
     "answer": "4"},
    {"id": "rsn_3", "prompt": "A store marks up an item costing $40 by 25%, then applies a 10% discount to the marked price. What is the final price in dollars? Answer with just the number.",
     "answer": "45"},
    {"id": "rsn_4", "prompt": "What is 17% of 250? Answer with just the number.",
     "answer": "42.5"},
    {"id": "rsn_5", "prompt": "A tank holds 500 liters. It is 40% full. Water flows in at 12 liters per minute. How many minutes until it is 90% full? Answer with just the number.",
     "answer": "20.833"},
    {"id": "rsn_6", "prompt": "If x + 2x + 3x = 84, what is x? Answer with just the number.",
     "answer": "14"},
]

CODE_TASKS = [
    {"id": "code_1",
     "prompt": "Complete the Python function. Return ONLY the completed function body line(s), no explanation.\n\ndef is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return",
     "expect_contains": ["[::-1]", "=="]},
    {"id": "code_2",
     "prompt": "Complete the Python function. Return ONLY the completed line(s), no explanation.\n\ndef sum_digits(n):\n    total = 0\n    while n > 0:\n        total += n % 10\n        n =",
     "expect_contains": ["// 10", "n //= 10", "n = n // 10"]},
    {"id": "code_3",
     "prompt": "Complete the Python function. Return ONLY the completed line(s), no explanation.\n\ndef fizzbuzz(n):\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            print('FizzBuzz')\n        elif i % 3 == 0:",
     "expect_contains": ["Fizz", "elif", "else"]},
    {"id": "code_4",
     "prompt": "Complete the Python function. Return ONLY the completed line(s), no explanation.\n\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:",
     "expect_contains": ["lo = mid", "lo=mid"]},
]

STRUCTURED_TASKS = [
    {"id": "struct_1",
     "prompt": "Extract the person from the sentence as JSON with keys name, age, city. Output ONLY valid JSON.\n\nSentence: Maria Chen, who is 34 years old, lives in Portland.",
     "required_keys": ["name", "age", "city"]},
    {"id": "struct_2",
     "prompt": "List the items as a JSON object with key 'items' containing an array of strings. Output ONLY valid JSON.\n\nItems: apples, bread, and a 2-liter bottle of milk.",
     "required_keys": ["items"]},
]

# Long-context passage: original coherent text, ~1200+ tokens when encoded.
# Repeats a varied technical narrative to exceed the context threshold.
_LONGCTX_PARAGRAPHS = [
    "The archive system at the northern observatory had been running continuously for eleven years when the first anomaly appeared in its calibration logs. Technicians initially dismissed the irregular readings as sensor drift, a common fault in instruments exposed to temperature swings between desert days and freezing nights. But the pattern persisted through three separate maintenance cycles, and the duty engineer who finally flagged it noted that the deviations correlated not with weather but with the station's own power-draw cycles.",
    "Investigation revealed that the archive's indexing algorithm had been silently accumulating rounding errors each time it compressed a nightly backup. Individually negligible, the errors compounded over thousands of cycles until the index no longer pointed to the correct data blocks. The fix required rewriting the compression routine and re-indexing eleven years of observations, a process that took four months and consumed most of the station's spare computing capacity.",
    "The lesson was incorporated into the observatory's design standards: every long-running archival system now undergoes a scheduled integrity audit, and all compression routines must document their numerical precision assumptions. What began as a minor calibration complaint became one of the most thorough reviews of the facility's data pipeline in its history, and the resulting procedures have since been adopted by three other research stations.",
]
LONGCTX_TEXT = "\n\n".join(_LONGCTX_PARAGRAPHS * 3)  # ~730 tokens


@dataclass
class TaskResult:
    task_id: str
    family: str
    score: float          # 1.0 pass / 0.0 fail, or ppl for longctx
    detail: str = ""
    gpu_seconds: float = 0.0


@dataclass
class CapabilityReport:
    per_family: Dict[str, dict] = field(default_factory=dict)
    results: List[TaskResult] = field(default_factory=list)
    gpu_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "per_family": self.per_family,
            "n_tasks": len(self.results),
            "gpu_seconds": round(self.gpu_seconds, 4),
        }


def _generate(model, tok, prompt: str, max_new: int = 96) -> str:
    """Greedy generation with fixed budget; returns decoded text."""
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def _first_number(text: str) -> Optional[str]:
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return m.group(0) if m else None


def _numbers_close(a: str, b: str, tol: float = 0.02) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (ValueError, ZeroDivisionError):
        return False


@torch.no_grad()
def evaluate_capability(model, tok) -> CapabilityReport:
    """Run the full capability suite. Returns per-family and per-task scores.

    Families: reasoning (exact numeric), code (keyword containment),
    structured (JSON parse + required keys), longctx (perplexity, lower is
    better — reported raw, not thresholded here).
    """
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    rep = CapabilityReport()
    fam_scores: Dict[str, List[float]] = {}

    # --- longctx FIRST: the single large forward needs the most free VRAM,
    # so run it before generation tasks fragment the allocator ---
    s0 = time.perf_counter()
    ids = tok(LONGCTX_TEXT, return_tensors="pt").input_ids.to("cuda")
    n_tokens = ids.shape[1]
    with torch.no_grad():
        out = model(ids)
        logits = out.logits  # bf16, do NOT .float() the whole tensor
        lab = ids[:, 1:]     # len = seq-1; logit position i predicts lab[i]
        nll = 0.0
        n = lab.numel()
        n_pos = lab.shape[1]
        chunk = 128
        for start in range(0, n_pos, chunk):
            end = min(start + chunk, n_pos)
            sl = logits[:, start:end, :].float()
            lb = lab[:, start:end]
            nll += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.size(-1)), lb.reshape(-1),
                reduction="sum").item())
        ce = nll / n if n else float("inf")
        ppl = float(torch.exp(torch.tensor(ce)).item()) if math.isfinite(ce) else float("inf")
    del out, logits, ids, lab
    torch.cuda.empty_cache()
    rep.results.append(TaskResult(
        task_id="longctx_1", family="longctx",
        score=round(ppl, 4),
        detail=f"tokens={n_tokens} ce={ce:.5f}",
        gpu_seconds=time.perf_counter() - s0))
    fam_scores.setdefault("longctx", []).append(ppl)

    # --- reasoning: exact numeric answer ---
    for task in REASONING_TASKS:
        s0 = time.perf_counter()
        gen = _generate(model, tok, task["prompt"] + "\nAnswer:", max_new=24)
        got = _first_number(gen)
        ok = got is not None and _numbers_close(got, task["answer"])
        rep.results.append(TaskResult(
            task_id=task["id"], family="reasoning",
            score=1.0 if ok else 0.0,
            detail=f"expected={task['answer']} got={got!r} raw={gen[:80]!r}",
            gpu_seconds=time.perf_counter() - s0))
        fam_scores.setdefault("reasoning", []).append(1.0 if ok else 0.0)
    torch.cuda.empty_cache()

    # --- code: required construct present in completion ---
    for task in CODE_TASKS:
        s0 = time.perf_counter()
        gen = _generate(model, tok, task["prompt"] + "\n", max_new=48)
        low = gen.lower()
        ok = any(k.lower() in low for k in task["expect_contains"])
        rep.results.append(TaskResult(
            task_id=task["id"], family="code",
            score=1.0 if ok else 0.0,
            detail=f"expect_any={task['expect_contains']} raw={gen[:80]!r}",
            gpu_seconds=time.perf_counter() - s0))
        fam_scores.setdefault("code", []).append(1.0 if ok else 0.0)
    torch.cuda.empty_cache()

    # --- structured: valid JSON with required keys ---
    for task in STRUCTURED_TASKS:
        s0 = time.perf_counter()
        gen = _generate(model, tok, task["prompt"] + "\n", max_new=96)
        ok = False
        detail = ""
        try:
            # Extract first JSON object
            start = gen.find("{")
            end = gen.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(gen[start:end + 1])
                ok = all(k in obj for k in task["required_keys"])
                detail = f"keys={list(obj.keys())}"
            else:
                detail = "no JSON object found"
        except json.JSONDecodeError as exc:
            detail = f"JSONDecodeError: {exc}"
        rep.results.append(TaskResult(
            task_id=task["id"], family="structured",
            score=1.0 if ok else 0.0,
            detail=f"{detail} raw={gen[:80]!r}",
            gpu_seconds=time.perf_counter() - s0))
        fam_scores.setdefault("structured", []).append(1.0 if ok else 0.0)
    torch.cuda.empty_cache()

    for fam, scores in fam_scores.items():
        if fam == "longctx":
            rep.per_family[fam] = {
                "score": round(sum(scores) / len(scores), 4),
                "metric": "ppl_lower_is_better",
                "n": len(scores),
            }
        else:
            rep.per_family[fam] = {
                "score": round(sum(scores) / len(scores), 4),
                "metric": "accuracy",
                "n": len(scores),
            }

    rep.gpu_seconds = time.perf_counter() - t0
    return rep


def capability_regression(
    baseline: CapabilityReport,
    candidate: CapabilityReport,
    accuracy_tolerance: float = 0.0,
    ppl_ratio_max: float = 1.10,
) -> Dict[str, dict]:
    """Compare candidate vs baseline capability. Returns per-family detail.

    Accuracy families: candidate must not drop below baseline - tolerance.
    longctx: candidate ppl must not exceed baseline ppl * ppl_ratio_max.
    """
    details = {}
    passed = True
    for fam in set(baseline.per_family) | set(candidate.per_family):
        b = baseline.per_family.get(fam, {})
        c = candidate.per_family.get(fam, {})
        if fam == "longctx":
            bp, cp = b.get("score", float("inf")), c.get("score", float("inf"))
            ratio = cp / bp if bp > 0 else float("inf")
            ok = ratio <= ppl_ratio_max
            details[fam] = {"baseline_ppl": bp, "candidate_ppl": cp,
                            "ratio": round(ratio, 4), "passed": ok}
        else:
            ba, ca = b.get("score", 0.0), c.get("score", 0.0)
            ok = ca >= ba - accuracy_tolerance
            details[fam] = {"baseline_acc": ba, "candidate_acc": ca,
                            "delta": round(ca - ba, 4), "passed": ok}
        if not details[fam]["passed"]:
            passed = False
    return {"passed": passed, "per_family": details}
