"""Gate 0.1 — A/B kernel comparison: fallback vs accelerated Gated DeltaNet.

Run under each interpreter to produce two comparable JSON manifests:

    .venv\\Scripts\\python.exe      src/jarvis_research/kernel_ab.py --mode fallback
    .venv-opt\\Scripts\\python.exe  src/jarvis_research/kernel_ab.py --mode accelerated

The script is identical in both runs. The only variable is the interpreter
environment: ``.venv`` has no ``fla``/``causal_conv1d`` (pure-PyTorch fallback),
``.venv-opt`` has ``fla-core`` + ``triton-windows`` (accelerated delta-rule).

Measurements (identical prompts, seeds, greedy decoding):
  * which kernels engaged (fla delta-rule call count, causal_conv1d presence)
  * model load time and post-load VRAM
  * greedy-output determinism across 5 identical runs
  * throughput (tok/s) at several prompt-length classes
  * BF16 edit -> forward -> rollback bit-exactness
  * recurrent-state isolation: does running prompt A first change prompt B's
    output? (guards against DeltaNet state leakage across sequential generate())
  * peak VRAM

Writes ``ai-research/results/kernel_ab_<mode>.json``. Local-only, gitignored.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
AI_RESEARCH = _THIS.parents[2]            # .../ai-research
RESULTS = AI_RESEARCH / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3.5-4B"
DTYPE = torch.bfloat16
GEN_MAX_NEW = 64                          # decode length for throughput runs
REPEAT_RUNS = 5                           # determinism repeats
SEED = 1234

PROMPT = "Summarize the primary mechanism of the gated delta rule in one sentence."
ISOLATION_A = "Write a one-sentence fact about the planet Saturn."
ISOLATION_B = "Reply with exactly the word: anchor"

# Prompt-length classes for throughput: we pad a filler to reach the class.
def _prompt_of_len(tok, n):
    base = "The quick brown fox jumps over the lazy dog. "
    # ~6 tokens per sentence; repeat to reach ~n tokens
    reps = max(1, n // 8)
    text = base * reps + "\nQuestion: how many dogs? Answer:"
    ids = tok(text, return_tensors="pt").input_ids
    return ids, text


def _now():
    return time.perf_counter()


def _vram_gb():
    return round(torch.cuda.memory_allocated() / (1 << 30), 3)


def _peak_gb():
    return round(torch.cuda.max_memory_allocated() / (1 << 30), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fallback", "accelerated"])
    args = ap.parse_args()
    mode = args.mode

    report = {
        "gate": "gate_0_1_kernel_ab",
        "mode": mode,
        "model": MODEL_ID,
        "seed": SEED,
        "git_sha": _git_sha(),
        "env": _env_manifest(),
        "kernels": {},
        "load": {},
        "determinism": {},
        "throughput": [],
        "edit_rollback": {},
        "state_isolation": {},
        "vram": {},
    }

    if not torch.cuda.is_available():
        report["error"] = "cuda_unavailable"
        _dump(report, mode)
        return 1

    # ------------------------------------------------------------------
    # Detect/instrument kernel availability BEFORE model load binds kernels.
    # We wrap fla's delta-rule ops with a counter so we can prove engagement.
    # ------------------------------------------------------------------
    fla_counter = {"chunk": 0, "recurrent": 0}
    kernels = report["kernels"]
    kernels["triton_present"] = _try_import("triton")
    kernels["causal_conv1d_present"] = _try_import("causal_conv1d")

    try:
        import functools

        import fla.ops.gated_delta_rule as _gdr

        _orig_chunk = getattr(_gdr, "chunk_gated_delta_rule", None)
        _orig_rec = getattr(_gdr, "fused_recurrent_gated_delta_rule", None)

        # functools.wraps preserves __wrapped__ -> real signature so that
        # transformers' applicable_params filtering forwards g/beta/etc.
        @functools.wraps(_orig_chunk)
        def _count_chunk(*a, **k):
            fla_counter["chunk"] += 1
            return _orig_chunk(*a, **k)

        @functools.wraps(_orig_rec)
        def _count_rec(*a, **k):
            fla_counter["recurrent"] += 1
            return _orig_rec(*a, **k)

        if _orig_chunk is not None:
            _gdr.chunk_gated_delta_rule = _count_chunk
        if _orig_rec is not None:
            _gdr.fused_recurrent_gated_delta_rule = _count_rec
        kernels["fla_present"] = True
    except Exception as e:  # noqa: BLE001
        kernels["fla_present"] = False
        kernels["fla_import_error"] = type(e).__name__

    # ------------------------------------------------------------------
    # Load model (BF16) — timing + VRAM
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = _now()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map={"": "cuda"}
    )
    model.eval()
    torch.cuda.synchronize()
    report["load"] = {"seconds": round(_now() - t0, 2), "vram_gb": _vram_gb()}
    torch.cuda.reset_peak_memory_stats()

    def generate(text, max_new=GEN_MAX_NEW):
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        torch.cuda.synchronize()
        st = _now()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        dt = _now() - st
        new = out.shape[1] - ids.shape[1]
        return out, new, dt, tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Determinism: 5x identical greedy generation -> outputs must match
    # ------------------------------------------------------------------
    outs = []
    for _ in range(REPEAT_RUNS):
        torch.manual_seed(SEED)
        _, _, _, txt = generate(PROMPT)
        outs.append(txt)
    report["determinism"] = {
        "runs": REPEAT_RUNS,
        "all_identical": len(set(outs)) == 1,
        "sample": outs[0][:200],
    }

    # ------------------------------------------------------------------
    # Throughput across prompt-length classes
    # ------------------------------------------------------------------
    for cls in (128, 512, 1024):
        ids, text = _prompt_of_len(tok, cls)
        # warmup once (kernel JIT / cache) then time 2 runs
        generate(text, max_new=GEN_MAX_NEW)
        ts = []
        for _ in range(2):
            _, new, dt, _ = generate(text, max_new=GEN_MAX_NEW)
            ts.append(round(new / dt, 2))
        report["throughput"].append(
            {"prompt_class": cls, "prompt_tokens": int(ids.shape[1]),
             "gen_tokens": GEN_MAX_NEW, "tok_per_s": ts, "median": round(sorted(ts)[len(ts)//2], 2)}
        )

    # ------------------------------------------------------------------
    # BF16 edit -> forward -> rollback bit-exactness (same dt_bias as Gate 0)
    # ------------------------------------------------------------------
    target = None
    for name, p in model.named_parameters():
        if "dt_bias" in name and p.dtype == torch.bfloat16:
            target = (name, p)
            break
    if target is not None:
        name, p = target
        orig = p.detach().clone()
        with torch.no_grad():
            p.view(-1)[:32] += 0.01
        changed = bool((p.detach() != orig).any().item())
        # forward to ensure edited weights still compute cleanly
        ids, text = _prompt_of_len(tok, 32)
        with torch.no_grad():
            _ = model(ids.to("cuda"))
        nan_inf = bool(torch.isnan(p).any().item() or torch.isinf(p).any().item())
        with torch.no_grad():
            p.copy_(orig)
        restored = bool(torch.equal(p.detach(), orig))
        report["edit_rollback"] = {
            "param": name, "elements": 32, "delta": 0.01,
            "values_changed": changed, "nan_inf_after_edit": nan_inf,
            "restored_bit_exact": restored,
        }
    else:
        report["edit_rollback"] = {"error": "dt_bias param not found"}

    # ------------------------------------------------------------------
    # Recurrent-state isolation: A then B, vs B alone. B must be identical.
    # ------------------------------------------------------------------
    torch.manual_seed(SEED)
    _, _, _, b_alone = generate(ISOLATION_B)
    torch.manual_seed(SEED)
    _, _, _, _a_out = generate(ISOLATION_A)
    torch.manual_seed(SEED)
    _, _, _, b_after_a = generate(ISOLATION_B)
    report["state_isolation"] = {
        "b_alone": b_alone[:120],
        "b_after_a": b_after_a[:120],
        "isolated": b_alone == b_after_a,
    }

    # ------------------------------------------------------------------
    # Kernel engagement + peak VRAM
    # ------------------------------------------------------------------
    report["kernels"]["fla_delta_rule_calls"] = fla_counter
    report["kernels"]["fla_engaged"] = (fla_counter["chunk"] + fla_counter["recurrent"]) > 0
    report["vram"] = {"post_load_gb": report["load"]["vram_gb"], "peak_gb": _peak_gb()}

    _dump(report, mode)
    print(json.dumps({
        "mode": mode,
        "fla_engaged": report["kernels"]["fla_engaged"],
        "cc1d_present": kernels["causal_conv1d_present"],
        "det_identical": report["determinism"]["all_identical"],
        "restore_exact": report["edit_rollback"].get("restored_bit_exact"),
        "isolated": report["state_isolation"]["isolated"],
        "median_toks": [t["median"] for t in report["throughput"]],
        "peak_vram": report["vram"]["peak_gb"],
    }, indent=2))
    return 0


def _env_manifest():
    import transformers
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1 << 30), 2),
        "bf16": torch.cuda.is_bf16_supported(),
    }


def _try_import(name):
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _git_sha():
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=AI_RESEARCH.parent,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _dump(report, mode):
    out = RESULTS / f"kernel_ab_{mode}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[kernel_ab] wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
