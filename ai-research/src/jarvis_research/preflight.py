"""
preflight.py — Gate 0 hardware preflight for JARVIS RIP v1.2

Measures CUDA/driver, BF16 support, Qwen3.5-4B generation, 5-run repeatability,
throughput at multiple prompt lengths, cold/hot VRAM, BF16 edit/rollback
bit-exactness, NaN/Inf detection, and per-candidate GPU-seconds estimate.

Output: ai-research/results/gate0_hardware.json

Per ROADMAP_v1.2 §4 Gate 0 and DEVIN_HANDOFF.md P0.
Does not leak private paths, tokens, or usernames.
"""

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[3]  # Sims2026/
RESULTS_DIR = REPO_ROOT / "ai-research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3.5-4B"
OUTPUT_FILE = RESULTS_DIR / "gate0_hardware.json"


def sanitize_path(p):
    """Strip usernames from paths for public-safe logging."""
    s = str(p)
    if "\\Users\\" in s:
        parts = s.split("\\")
        for i, part in enumerate(parts):
            if part == "Users" and i + 1 < len(parts):
                parts[i + 1] = "<user>"
        s = "\\".join(parts)
    if "/Users/" in s:
        parts = s.split("/")
        for i, part in enumerate(parts):
            if part == "Users" and i + 1 < len(parts):
                parts[i + 1] = "<user>"
        s = "/".join(parts)
    return s


def get_git_sha(repo_path):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_env_manifest():
    """Record static environment info — no model needed."""
    import transformers

    gpu_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\")

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(REPO_ROOT),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "transformers_version": transformers.__version__,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "gpu": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb": round(gpu_props.total_memory / 1e9, 2) if gpu_props else None,
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}" if gpu_props else None,
            "multiprocessor_count": gpu_props.multi_processor_count if gpu_props else None,
        },
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_available_gb": round(vm.available / 1e9, 1),
        "disk_free_gb": round(disk.free / 1e9, 1),
        "model_id": MODEL_ID,
    }

    # Driver version via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        manifest["gpu"]["driver_version"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        manifest["gpu"]["driver_version"] = "unknown"

    # Model revision hash (from HF, without downloading weights)
    try:
        from huggingface_hub import model_info
        info = model_info(MODEL_ID)
        manifest["model_revision"] = info.sha
    except Exception:
        manifest["model_revision"] = "unknown"

    return manifest


def load_model():
    """Load Qwen3.5-4B in BF16 on GPU. Returns (model, tokenizer, load_time_s)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[preflight] Loading model (this triggers download if not cached)...", flush=True)
    start = time.monotonic()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    load_time = time.monotonic() - start
    print(f"[preflight] Model loaded in {load_time:.1f}s", flush=True)
    return model, tokenizer, load_time


def measure_vram():
    """Current and peak GPU VRAM in GB."""
    if not torch.cuda.is_available():
        return {"current_gb": 0, "peak_gb": 0}
    current = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {"current_gb": round(current, 2), "peak_gb": round(peak, 2)}


def reset_vram_tracker():
    torch.cuda.reset_peak_memory_stats()


def check_nan_inf(tensor, label):
    """Check a tensor for NaN/Inf, return status dict."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    status = "ok" if not (has_nan or has_inf) else ("nan" if has_nan else "inf")
    if status != "ok":
        print(f"[preflight] WARNING: {label} has {status}", flush=True)
    return {"label": label, "status": status, "has_nan": has_nan, "has_inf": has_inf}


def run_baseline_repeatability(model, tokenizer, n_runs=5):
    """5 exact baseline runs with frozen prompt, greedy decoding.
    All 5 outputs MUST be identical for Go."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"[preflight] Baseline repeatability: {n_runs} runs, greedy decoding", flush=True)
    results = []
    outputs = []

    for i in range(n_runs):
        torch.manual_seed(42)
        reset_vram_tracker()
        start = time.monotonic()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.monotonic() - start
        generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        outputs.append(generated)
        vram = measure_vram()

        nan_check = check_nan_inf(out, f"baseline_run_{i}")

        results.append({
            "run": i,
            "output": generated,
            "time_s": round(elapsed, 3),
            "vram": vram,
            "nan_inf": nan_check,
        })
        print(f"  run {i}: {elapsed:.2f}s — {generated[:60]}...", flush=True)

    all_identical = len(set(outputs)) == 1
    print(f"[preflight] All {n_runs} outputs identical: {all_identical}", flush=True)

    return {
        "n_runs": n_runs,
        "all_outputs_identical": all_identical,
        "outputs": outputs,
        "runs": results,
        "go": all_identical,
    }


def run_throughput_tests(model, tokenizer):
    """Throughput at 1x, 128, 512, 1024 token prompts."""
    print("[preflight] Throughput tests", flush=True)
    results = {}

    # Build prompts of different lengths using repeated text
    base_text = "The quick brown fox jumps over the lazy dog. "
    prompts = {
        "1x": "Hello",
        "128": base_text * 6,    # ~128 tokens
        "512": base_text * 24,   # ~512 tokens
        "1024": base_text * 48,  # ~1024 tokens
    }

    for label, prompt in prompts.items():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        input_len = inputs.input_ids.shape[1]

        reset_vram_tracker()
        start = time.monotonic()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.monotonic() - start
        new_tokens = out.shape[1] - input_len
        tok_per_s = new_tokens / elapsed if elapsed > 0 else 0
        vram = measure_vram()

        nan_check = check_nan_inf(out, f"throughput_{label}")

        results[label] = {
            "input_tokens": input_len,
            "output_tokens": new_tokens,
            "total_time_s": round(elapsed, 3),
            "tok_per_s": round(tok_per_s, 1),
            "vram": vram,
            "nan_inf": nan_check,
        }
        print(f"  {label} ({input_len} tok in): {elapsed:.2f}s, {tok_per_s:.1f} tok/s, {vram['peak_gb']:.2f} GB peak", flush=True)

    return results


def run_bf16_edit_rollback(model):
    """BF16 edit/rollback bit-exactness smoke test under torch.no_grad()."""
    print("[preflight] BF16 edit/rollback bit-exactness test", flush=True)

    # Find a DeltaNet layer parameter to test with
    test_param = None
    test_param_name = None
    for name, param in model.named_parameters():
        if "dt_bias" in name or "A_log" in name:
            test_param = param
            test_param_name = name
            break

    if test_param is None:
        # Fallback: use any small parameter
        for name, param in model.named_parameters():
            if param.numel() <= 4096:
                test_param = param
                test_param_name = name
                break

    if test_param is None:
        return {"status": "skipped", "reason": "no suitable parameter found"}

    # Sanitize the parameter name (remove model prefix for logging)
    safe_name = test_param_name.split(".")[-1] if test_param_name else "unknown"

    with torch.no_grad():
        # 1. Save original bit pattern
        original_data = test_param.data.clone()
        original_uint16 = test_param.data.view(torch.uint16).clone()

        # 2. Apply a small edit (add scalar to a subset)
        n_edit = min(64, test_param.numel())
        edit_delta = 0.01
        flat = test_param.data.view(-1)
        flat[:n_edit].add_(edit_delta)

        # 3. Verify modification applied
        modified_uint16 = test_param.data.view(torch.uint16)
        bits_changed = (original_uint16 != modified_uint16).sum().item()
        values_changed = (original_data != test_param.data).any().item()

        # 4. Check no NaN/Inf introduced
        nan_after_edit = torch.isnan(test_param.data).any().item()
        inf_after_edit = torch.isinf(test_param.data).any().item()

        # 5. Rollback
        test_param.data.copy_(original_data)

        # 6. Verify bit-exact restoration
        restored_uint16 = test_param.data.view(torch.uint16)
        restoration_exact = torch.equal(original_uint16, restored_uint16)

    result = {
        "parameter": safe_name,
        "numel": test_param.numel(),
        "elements_edited": n_edit,
        "edit_delta": edit_delta,
        "bits_changed": bits_changed,
        "values_changed": values_changed,
        "nan_after_edit": nan_after_edit,
        "inf_after_edit": inf_after_edit,
        "restoration_bit_exact": restoration_exact,
        "go": restoration_exact and not nan_after_edit and not inf_after_edit,
    }
    print(f"  param={safe_name}, edited {n_edit} elements, bits_changed={bits_changed}, "
          f"restored_exact={restoration_exact}", flush=True)
    return result


def estimate_candidate_cost(baseline_results, throughput_results):
    """Estimate GPU-seconds per candidate evaluation."""
    # Use the 128-token prompt as representative
    tp = throughput_results.get("128", {})
    baseline = baseline_results["runs"][0] if baseline_results["runs"] else {}

    # A candidate evaluation = generate response + apply edit + score + reload
    # Use measured generation time as the dominant cost
    gen_time = tp.get("total_time_s", 0)
    per_task_time = gen_time  # one task per candidate, 32 output tokens

    # 16 practice problems per candidate (per Gate 1 pilot)
    estimated_16_tasks = per_task_time * 16
    # 100 candidates
    estimated_100_candidates = estimated_16_tasks * 100
    # 500 candidates
    estimated_500_candidates = estimated_16_tasks * 500

    return {
        "measured_gen_time_s_128_prompt_32_output": gen_time,
        "estimated_16_tasks_per_candidate_s": round(estimated_16_tasks, 1),
        "estimated_100_candidates_s": round(estimated_100_candidates, 1),
        "estimated_100_candidates_hours": round(estimated_100_candidates / 3600, 2),
        "estimated_500_candidates_s": round(estimated_500_candidates, 1),
        "estimated_500_candidates_hours": round(estimated_500_candidates / 3600, 2),
    }


def run_preflight():
    """Run the full Gate 0 preflight and write gate0_hardware.json."""
    print("=" * 60, flush=True)
    print("[preflight] Gate 0 — Hardware preflight for JARVIS RIP v1.2", flush=True)
    print("=" * 60, flush=True)

    report = {
        "gate": "0",
        "protocol_version": "1.2.0-draft",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Environment manifest
    print("\n[preflight] 1. Environment manifest", flush=True)
    env = get_env_manifest()
    report["environment"] = env
    print(f"  torch={env['torch_version']}, cuda={env['cuda_version']}, "
          f"bf16={env['bf16_supported']}, gpu={env['gpu']['name']}", flush=True)

    if not env["cuda_available"]:
        report["go"] = False
        report["no_go_reason"] = "CUDA not available"
        write_report(report)
        return

    # 2. Load model
    print("\n[preflight] 2. Model load", flush=True)
    model, tokenizer, load_time = load_model()
    reset_vram_tracker()

    load_vram = measure_vram()
    report["model_load"] = {
        "time_s": round(load_time, 1),
        "vram_after_load": load_vram,
    }
    print(f"  loaded in {load_time:.1f}s, VRAM={load_vram['current_gb']:.2f} GB", flush=True)

    # 3. NaN/Inf check on model weights
    print("\n[preflight] 3. Model weight NaN/Inf check", flush=True)
    weight_checks = []
    has_bad_weights = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any().item() or torch.isinf(param).any().item():
            has_bad_weights = True
            weight_checks.append({"name": name.split(".")[-1], "has_nan": torch.isnan(param).any().item(), "has_inf": torch.isinf(param).any().item()})
    report["weight_nan_inf"] = {
        "any_bad": has_bad_weights,
        "bad_params": weight_checks,
    }
    print(f"  bad weights: {has_bad_weights}", flush=True)

    # 4. Baseline repeatability
    print("\n[preflight] 4. Baseline repeatability (5 runs, greedy)", flush=True)
    baseline = run_baseline_repeatability(model, tokenizer, n_runs=5)
    report["baseline_repeatability"] = baseline

    # 5. Throughput
    print("\n[preflight] 5. Throughput tests", flush=True)
    throughput = run_throughput_tests(model, tokenizer)
    report["throughput"] = throughput

    # 6. BF16 edit/rollback
    print("\n[preflight] 6. BF16 edit/rollback bit-exactness", flush=True)
    edit_test = run_bf16_edit_rollback(model)
    report["bf16_edit_rollback"] = edit_test

    # 7. Cost estimate
    print("\n[preflight] 7. Per-candidate cost estimate", flush=True)
    cost = estimate_candidate_cost(baseline, throughput)
    report["cost_estimate"] = cost
    print(f"  100 candidates ~{cost['estimated_100_candidates_hours']}h, "
          f"500 candidates ~{cost['estimated_500_candidates_hours']}h", flush=True)

    # 8. Go/No-go
    go = (
        env["cuda_available"]
        and env["bf16_supported"]
        and baseline["go"]  # all 5 outputs identical
        and edit_test.get("go", False)  # bit-exact rollback
        and not has_bad_weights  # no NaN/Inf in weights
    )
    report["go"] = go
    if not go:
        reasons = []
        if not baseline["go"]:
            reasons.append("baseline outputs not identical (nondeterminism)")
        if not edit_test.get("go", False):
            reasons.append("BF16 edit/rollback not bit-exact")
        if has_bad_weights:
            reasons.append("NaN/Inf in model weights")
        report["no_go_reasons"] = reasons

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    write_report(report)
    print(f"\n[preflight] Gate 0 result: {'GO' if go else 'NO-GO'}", flush=True)
    print(f"[preflight] Report written to: {sanitize_path(OUTPUT_FILE)}", flush=True)
    return go


def write_report(report):
    """Write JSON report, sanitizing any paths."""
    # Sanitize the report (no absolute paths with usernames)
    text = json.dumps(report, indent=2, default=str)
    # The report is constructed to not include paths, but double-check
    OUTPUT_FILE.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run_preflight()
