"""Gate 1.1 capability suite — task-level evaluation separate from perplexity.

The language-modeling partitions measure text likelihood; they cannot detect
loss of reasoning, code, structured-output, or long-context capability. This
suite adds four small, deterministic task families scored INDEPENDENTLY of
the selection/regression perplexity endpoint:

  reasoning   — multi-step arithmetic word problems, exact numeric answer
  code        — short function completion, sandboxed exec on test inputs
  structured  — constrained JSON generation, schema + semantic value checks
  longctx     — perplexity on a ~730-token coherent passage (fluency only)
  retrieval   — needle question requiring information from deep in LONGCTX

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
     "tests": [("is_palindrome('Racecar')", True),
               ("is_palindrome('A man a plan a canal Panama')", True),
               ("is_palindrome('banana')", False)]},
    {"id": "code_2",
     "prompt": "Complete the Python function. Return ONLY the completed line(s), no explanation.\n\ndef sum_digits(n):\n    total = 0\n    while n > 0:\n        total += n % 10\n        n =",
     "tests": [("sum_digits(1234)", 10), ("sum_digits(999)", 27),
               ("sum_digits(5)", 5)]},
    {"id": "code_3",
     "prompt": "Complete the Python function. Return ONLY the completed line(s), no explanation.\n\ndef fizzbuzz_lines(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:",
     "tests": [("fizzbuzz_lines(16)", ["1", "2", "Fizz", "4", "Buzz", "Fizz",
                                      "7", "8", "Fizz", "Buzz", "11", "Fizz",
                                      "13", "14", "FizzBuzz", "16"])]},
    {"id": "code_4",
     "prompt": "Complete the Python function. Return ONLY the completed line(s), no explanation.\n\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:",
     "tests": [("binary_search([1,3,5,7,9], 7)", 3),
               ("binary_search([1,3,5,7,9], 4)", -1),
               ("binary_search([2], 2)", 0)]},
]

STRUCTURED_TASKS = [
    {"id": "struct_1",
     "prompt": "Extract the person from the sentence as JSON with keys name, age, city. Output ONLY valid JSON.\n\nSentence: Maria Chen, who is 34 years old, lives in Portland.",
     "required_keys": ["name", "age", "city"],
     "value_checks": {"name": {"contains_all": ["maria", "chen"]},
                      "age": {"equals": 34},
                      "city": {"contains_all": ["portland"]}}},
    {"id": "struct_2",
     "prompt": "List the items as a JSON object with key 'items' containing an array of strings. Output ONLY valid JSON.\n\nItems: apples, bread, and a 2-liter bottle of milk.",
     "required_keys": ["items"],
     "value_checks": {"items": {"list_contains_all": ["apple", "bread", "milk"]}}},
]

# Long-context passage: original coherent text, ~1200+ tokens when encoded.
# Repeats a varied technical narrative to exceed the context threshold.
_LONGCTX_PARAGRAPHS = [
    "The archive system at the northern observatory had been running continuously for eleven years when the first anomaly appeared in its calibration logs. Technicians initially dismissed the irregular readings as sensor drift, a common fault in instruments exposed to temperature swings between desert days and freezing nights. But the pattern persisted through three separate maintenance cycles, and the duty engineer who finally flagged it noted that the deviations correlated not with weather but with the station's own power-draw cycles.",
    "Investigation revealed that the archive's indexing algorithm had been silently accumulating rounding errors each time it compressed a nightly backup. Individually negligible, the errors compounded over thousands of cycles until the index no longer pointed to the correct data blocks. The duty engineer's calibration threshold, recorded in the maintenance log as exactly 47.3 microkelvin, turned out to be the decisive clue isolating the faulty routine. The fix required rewriting the compression routine and re-indexing eleven years of observations, a process that took four months and consumed most of the station's spare computing capacity.",
    "The lesson was incorporated into the observatory's design standards: every long-running archival system now undergoes a scheduled integrity audit, and all compression routines must document their numerical precision assumptions. What began as a minor calibration complaint became one of the most thorough reviews of the facility's data pipeline in its history, and the resulting procedures have since been adopted by three other research stations.",
]
LONGCTX_TEXT = "\n\n".join(_LONGCTX_PARAGRAPHS * 3)  # ~750 tokens

# Needle task: the fact sits mid-passage inside repeated text, so correct
# answers require actual long-range retrieval — not perplexity.
RETRIEVAL_TASKS = [
    {"id": "retr_1",
     "prompt": (LONGCTX_TEXT
                + "\n\nQuestion: What was the duty engineer's calibration "
                  "threshold, in microkelvin? Answer with just the number."),
     "answer": "47.3"},
]


def suite_fingerprint() -> str:
    """SHA-256 over the exact task definitions — a resume manifest can detect
    a changed suite instead of silently mixing scoring versions."""
    import hashlib
    payload = {
        "reasoning": REASONING_TASKS, "code": CODE_TASKS,
        "structured": STRUCTURED_TASKS, "longctx": LONGCTX_TEXT,
        "retrieval": RETRIEVAL_TASKS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class TaskResult:
    task_id: str
    family: str
    score: float          # 1.0 pass / 0.0 fail, or ppl for longctx
    detail: str = ""
    gpu_seconds: float = 0.0
    # True when the task could not be fairly graded because evaluator
    # infrastructure failed (e.g. containment refusal) — never a model
    # capability data point (HZN-IR-05 §3).
    infrastructure_error: bool = False


@dataclass
class CapabilityReport:
    per_family: Dict[str, dict] = field(default_factory=dict)
    results: List[TaskResult] = field(default_factory=list)
    gpu_seconds: float = 0.0
    has_infrastructure_failures: bool = False

    def to_dict(self) -> dict:
        return {
            "per_family": self.per_family,
            "n_tasks": len(self.results),
            "gpu_seconds": round(self.gpu_seconds, 4),
            "has_infrastructure_failures": self.has_infrastructure_failures,
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


# --- Sandboxed code grading -------------------------------------------------

# CONTAINMENT BOUNDARY (HZN-IR-04 R4): generated code executes inside a
# short-lived isolated-mode subprocess confined to a kill-on-close
# Windows Job Object (512 MiB commit ceiling) that is assigned BEFORE
# any generated code runs — the driver blocks on a single stdin gate
# byte that the parent only sends after assignment succeeds. If the job
# cannot be created or the process cannot be assigned, the gate byte is
# never sent, the child is killed, and grading is REFUSED: containment
# is fail-closed, never reported-and-continued. Output is bounded
# (256 KiB stdout / 64 KiB stderr drain caps) so a spamming completion
# cannot deadlock the pipe or exhaust evaluator memory.
# The AST restrictions, restricted builtins and wall-clock timeout are
# GRADING SAFEGUARDS layered on that containment. This is still not a
# complete security sandbox: there is no separate low-privilege OS
# token, no syscall mediation and no filesystem policy — those are
# required before this path ever grades adversarial code, not
# retroactively justified by the checks above.
_SAFE_BUILTINS = (
    "abs all any bool dict divmod enumerate filter float format frozenset "
    "int isinstance len list map max min ord pow print range repr reversed "
    "round set sorted str sum tuple zip chr"
).split()

_CODE_DRIVER = r'''
import ast, builtins, json, sys

# Containment gate (HZN-IR-04 R4): block until the parent confirms the
# Job Object assignment by sending one byte. Only interpreter startup
# and these stdlib imports happen before the gate — NO generated code
# is parsed or executed until containment is established.
sys.stdin.buffer.read(1)

source = json.loads(sys.argv[1])
tests = json.loads(sys.argv[2])
tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        print(json.dumps({"ok": False, "error": "import not allowed"})); sys.exit(0)
    if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
        print(json.dumps({"ok": False, "error": "dunder access not allowed"})); sys.exit(0)
    if isinstance(node, ast.Name) and node.id in {
            "eval","exec","compile","open","input","globals","locals","vars",
            "__import__","breakpoint","exit","quit","getattr","setattr","delattr"}:
        print(json.dumps({"ok": False, "error": "forbidden name " + node.id})); sys.exit(0)

SAFE = set(%(safe)r)
ns = {"__builtins__": {k: getattr(builtins, k)
                      for k in SAFE if hasattr(builtins, k)}}
try:
    exec(compile(tree, "<completion>", "exec"), ns)
    results = []
    for call, expected in tests:
        got = eval(call, ns)
        results.append(got == expected)
    print(json.dumps({"ok": True, "results": results}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": repr(exc)[:200]}))
''' % {"safe": _SAFE_BUILTINS}


_JOB_MEMORY_LIMIT = 512 * 1024 * 1024  # 512 MiB commit ceiling


def _assign_windows_job(proc, memory_limit=_JOB_MEMORY_LIMIT):
    """Confine `proc` to a kill-on-close Job Object (HZN-IR-03 R4).

    Process-level containment: a private memory ceiling and automatic
    termination of the whole tree on timeout/close. NOTE: Windows'
    ACTIVE_PROCESS limit is deliberately NOT set — it prevents Python
    interpreter initialization (verified empirically, rc=101). Child
    spawning is instead prevented by the AST gate, which blocks all
    imports. Returns the job handle, or None on any failure — callers
    must treat None as 'unconfined' and record it.
    """
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation",
                         JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ("IoInfo", ctypes.c_byte * 48),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            0x2000 |  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            0x0200 |  # JOB_OBJECT_LIMIT_JOB_MEMORY
            0x0400    # JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        )
        info.JobMemoryLimit = memory_limit
        ok = k32.SetInformationJobObject(
            job, 9, ctypes.byref(info),  # JobObjectExtendedLimitInformation
            ctypes.sizeof(info))
        if not ok or not k32.AssignProcessToJobObject(job, int(proc._handle)):
            k32.CloseHandle(job)
            return None
        return job
    except (AttributeError, OSError):
        return None


_STDOUT_CAP = 256 * 1024
_STDERR_CAP = 64 * 1024


def _drain(stream, buf: bytearray, cap: int) -> None:
    """Fully drain a pipe into `buf` capped at `cap` bytes.

    Keeps reading past the cap so the child can never deadlock on a
    full pipe; only the retained prefix is bounded.
    """
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        room = cap - len(buf)
        if room > 0:
            buf += chunk[:room]


def _run_code_sandboxed(source: str, tests: list,
                        timeout_s: float = 15.0) -> dict:
    """Execute candidate code in a confined subprocess; fail-closed.

    The child blocks on a one-byte stdin gate until a kill-on-close Job
    Object (512 MiB ceiling) has been assigned, so generated code never
    executes uncontained. A containment failure REFUSES grading and
    returns {"ok": False, "containment": "refused"} — there is no
    confined=false execution path. Output drains are capped.
    """
    import subprocess
    import sys
    import threading
    job = None
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", _CODE_DRIVER,
         json.dumps(source), json.dumps(tests)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if proc.poll() is not None:
            return {"ok": False,
                    "error": "grader subprocess exited before containment",
                    "confined": False}
        job = _assign_windows_job(proc)
        if job is None:
            proc.kill()
            proc.wait()
            return {"ok": False,
                    "error": ("containment unavailable: job assignment "
                              "failed; grading refused (fail-closed)"),
                    "confined": False,
                    "containment": "refused"}
        # Containment established -> release the gated driver.
        proc.stdin.write(b"1")
        proc.stdin.flush()
        proc.stdin.close()
        out_buf, err_buf = bytearray(), bytearray()
        threads = [
            threading.Thread(target=_drain,
                             args=(proc.stdout, out_buf, _STDOUT_CAP),
                             daemon=True),
            threading.Thread(target=_drain,
                             args=(proc.stderr, err_buf, _STDERR_CAP),
                             daemon=True)]
        for t in threads:
            t.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return {"ok": False,
                    "error": f"timeout after {timeout_s}s",
                    "confined": True}
        for t in threads:
            t.join(timeout=5)
        out = out_buf.decode("utf-8", errors="replace")
        err = err_buf.decode("utf-8", errors="replace")
        line = out.strip().splitlines()[-1] if out.strip() else ""
        result = {"confined": True}
        if len(out_buf) >= _STDOUT_CAP or len(err_buf) >= _STDERR_CAP:
            result["output_truncated"] = True
        if not line:
            result.update(ok=False,
                          error=f"no output (stderr={err[-200:]!r})")
            return result
        result.update(json.loads(line))
        return result
    except Exception as exc:
        return {"ok": False, "error": repr(exc)[:200],
                "confined": job is not None}
    finally:
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream and not stream.closed:
                    stream.close()
            except Exception:
                pass
        if job is not None:
            import ctypes
            ctypes.windll.kernel32.CloseHandle(job)


def _check_value(value, spec: dict) -> bool:
    """Semantic checks on parsed JSON values."""
    for kind, want in spec.items():
        if kind == "contains_all":
            hay = str(value).lower()
            if not all(w.lower() in hay for w in want):
                return False
        elif kind == "equals":
            try:
                if not _numbers_close(str(value), str(want)):
                    return False
            except Exception:
                return False
        elif kind == "list_contains_all":
            if not isinstance(value, list):
                return False
            hay = " ".join(str(v).lower() for v in value)
            if not all(w.lower() in hay for w in want):
                return False
        else:
            return False
    return True


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

    # --- code: reconstruct + sandboxed execution against test inputs ---
    for task in CODE_TASKS:
        s0 = time.perf_counter()
        gen = _generate(model, tok, task["prompt"] + "\n", max_new=48)
        # Rebuild the candidate source: prompt tail (after 'def ') + the
        # model's completion. Insert a space if the continuation doesn't
        # start with whitespace (e.g. "return" + "s == ...").
        head = task["prompt"].split("def ", 1)[1]
        sep = "" if gen[:1].isspace() else " "
        source = "def " + head + sep + gen
        res = _run_code_sandboxed(source, task["tests"])
        # A containment refusal is an evaluator infrastructure failure,
        # not evidence the model produced bad code (HZN-IR-05 §3): tag
        # the result and exclude it from scored aggregates.
        infra = res.get("containment") == "refused"
        ok = res.get("ok") and all(res.get("results") or [])
        detail = (f"results={res.get('results')}"
                  if res.get("ok") else f"error={res.get('error')}")
        rep.results.append(TaskResult(
            task_id=task["id"], family="code",
            score=1.0 if ok else 0.0,
            detail=f"{detail} raw={gen[:60]!r}",
            gpu_seconds=time.perf_counter() - s0,
            infrastructure_error=infra))
        if not infra:
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
                # Semantic checks: required values must match the source,
                # not merely exist.
                for key, spec in (task.get("value_checks") or {}).items():
                    if key not in obj or not _check_value(obj[key], spec):
                        ok = False
                        detail = f"value_check_failed key={key}"
                if ok:
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

    # --- retrieval: needle fact deep inside the long passage ---
    for task in RETRIEVAL_TASKS:
        s0 = time.perf_counter()
        gen = _generate(model, tok, task["prompt"] + "\nAnswer:", max_new=24)
        got = _first_number(gen)
        ok = got is not None and _numbers_close(got, task["answer"])
        rep.results.append(TaskResult(
            task_id=task["id"], family="retrieval",
            score=1.0 if ok else 0.0,
            detail=f"expected={task['answer']} got={got!r} raw={gen[:80]!r}",
            gpu_seconds=time.perf_counter() - s0))
        fam_scores.setdefault("retrieval", []).append(1.0 if ok else 0.0)
    torch.cuda.empty_cache()

    n_infra_by_fam: Dict[str, int] = {}
    for t in rep.results:
        if t.infrastructure_error:
            n_infra_by_fam[t.family] = n_infra_by_fam.get(t.family, 0) + 1
            fam_scores.setdefault(t.family, [])
    for fam, scores in fam_scores.items():
        entry = {
            "score": (round(sum(scores) / len(scores), 4)
                      if scores else None),
            "metric": ("ppl_lower_is_better" if fam == "longctx"
                       else "accuracy"),
            "n": len(scores),
        }
        if n_infra_by_fam.get(fam):
            entry["n_infrastructure_errors"] = n_infra_by_fam[fam]
            if not scores:
                entry["unscorable_infrastructure"] = True
        rep.per_family[fam] = entry

    # Evaluator integrity (HZN-IR-03 R4): duplicate task IDs would pair
    # wrong results in the transition ledger, and a nonfinite score would
    # silently poison family means. Reject both explicitly.
    ids = [t.task_id for t in rep.results]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate capability task ids: {dupes}")
    bad = [t.task_id for t in rep.results
           if not math.isfinite(t.score)]
    if bad:
        raise ValueError(f"nonfinite capability scores: {bad}")

    rep.gpu_seconds = time.perf_counter() - t0
    rep.has_infrastructure_failures = any(
        t.infrastructure_error for t in rep.results)
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
    n_infra = 0
    for fam in set(baseline.per_family) | set(candidate.per_family):
        b = baseline.per_family.get(fam, {})
        c = candidate.per_family.get(fam, {})
        n_infra += (b.get("n_infrastructure_errors", 0)
                    + c.get("n_infrastructure_errors", 0))
        if fam == "longctx":
            bp, cp = b.get("score", float("inf")), c.get("score", float("inf"))
            ratio = cp / bp if bp > 0 else float("inf")
            ok = ratio <= ppl_ratio_max
            details[fam] = {"baseline_ppl": bp, "candidate_ppl": cp,
                            "ratio": round(ratio, 4), "passed": ok}
        elif b.get("score") is None or c.get("score") is None:
            # No scored tasks on at least one side — an infrastructure
            # failure, NOT a model regression (HZN-IR-05 §3). Conserva-
            # tive: an unscorable family cannot certify "no regression".
            details[fam] = {"passed": False,
                            "unscorable_infrastructure": True,
                            "n_infrastructure_errors":
                                b.get("n_infrastructure_errors", 0)
                                + c.get("n_infrastructure_errors", 0)}
        else:
            ba, ca = b["score"], c["score"]
            ok = ca >= ba - accuracy_tolerance
            details[fam] = {"baseline_acc": ba, "candidate_acc": ca,
                            "delta": round(ca - ba, 4), "passed": ok}
        if not details[fam]["passed"]:
            passed = False
    return {"passed": passed, "per_family": details,
            "n_infrastructure_errors": n_infra}


def transition_ledger(
    baseline: CapabilityReport,
    candidate: CapabilityReport,
    longctx_ratio: float = 1.10,
) -> Dict[str, dict]:
    """GUARD-style paired transitions per task.

    Four outcomes per (baseline, candidate) pair: retained (both pass),
    acquisition (fail→pass), regression (pass→fail), unresolved (both
    fail). A net accuracy delta hides the distinction between, e.g.,
    +3/−0 and +23/−20. longctx is scored on ppl ratio rather than pass/fail.
    """
    # Exact task-ID correspondence is required (HZN-IR-02 §6): silently
    # skipping a missing baseline result would let a changed suite compare
    # mismatched pairs. Fail loudly instead.
    base_by_id = {t.task_id: t for t in baseline.results}
    cand_ids = {t.task_id for t in candidate.results}
    missing = cand_ids - set(base_by_id)
    extra = set(base_by_id) - cand_ids
    if missing or extra:
        raise ValueError(
            f"transition_ledger task-id mismatch: missing baseline "
            f"{sorted(missing)}, missing candidate {sorted(extra)}")
    fams: Dict[str, dict] = {}
    overall = {"acquisition": 0, "regression": 0, "retained": 0,
               "unresolved": 0, "infrastructure_failure": 0}
    # Continuous (ppl-ratio) transitions are reported separately from the
    # binary acquisition/regression/retained rates — mixing a relative-
    # change metric into pass/fail counts overstates the binary screen.
    continuous = {}
    for t in candidate.results:
        b = base_by_id[t.task_id]
        if t.infrastructure_error or b.infrastructure_error:
            # Evaluator-side failure on either side: the pair is
            # unevaluable, never a regression or acquisition (HZN-IR-05).
            f = fams.setdefault(t.family, {"acquisition": 0,
                                           "regression": 0,
                                           "retained": 0,
                                           "unresolved": 0,
                                           "infrastructure_failure": 0})
            f["infrastructure_failure"] += 1
            overall["infrastructure_failure"] += 1
            continue
        if t.family == "longctx":
            ratio = t.score / b.score if b.score > 0 else float("inf")
            continuous[t.task_id] = {
                "baseline_score": b.score, "candidate_score": t.score,
                "ratio": (None if ratio == float("inf") else round(ratio, 4)),
                "transition": ("regression" if ratio > longctx_ratio else
                               "acquisition" if ratio < 1.0 / longctx_ratio
                               else "retained")}
            continue
        bp, cp = b.score >= 0.5, t.score >= 0.5
        state = ("retained" if bp and cp else
                 "acquisition" if not bp and cp else
                 "regression" if bp and not cp else "unresolved")
        f = fams.setdefault(t.family, {"acquisition": 0, "regression": 0,
                                       "retained": 0, "unresolved": 0})
        f[state] += 1
        overall[state] += 1
    return {"per_family": fams, "overall": overall,
            "continuous_transitions": continuous}
