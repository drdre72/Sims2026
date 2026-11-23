# ai-research — JARVIS Recursive Intelligence Research

Isolated ML research environment for the JARVIS RIP v1.2 protocol
(`Horizon/ROADMAP_v1.2.md` in the JarvisProject repo). Separate from
Sims2026's hardware-simulation `.venv` — do not install GPU libraries
into the existing environment.

## Structure

```
ai-research/
  .venv/            # ignored; PyTorch 2.8.0+cu128, transformers 5.24.0
  locks/            # frozen package manifests
  research/         # ROADMAP, preregistration, literature, reviewer log
  schemas/          # proposal/observation/checkpoint JSON schemas
  src/jarvis_research/  # research code (preflight, model_adapter, etc.)
  tests/            # unit tests (fake tensors, mock models)
  configs/          # experiment configs
  data/             # ignored; practice only, local
  results/          # ignored; gate0_hardware.json etc.
```

## Gate 0 results (2026-11-23)

**Status: GO** — all critical checks passed.

| Check | Result |
|---|---|
| CUDA available | True (CUDA 12.8, RTX A4500, 21.46 GB VRAM) |
| BF16 supported | True |
| Model load | 10.4s, 8.41 GB VRAM (12 GB free) |
| Weight NaN/Inf | None |
| Baseline repeatability (5 runs, greedy) | All identical: True |
| Throughput (fallback kernels) | ~12 tok/s at all prompt lengths |
| BF16 edit/rollback bit-exact | True (dt_bias, 32 elements, delta 0.01) |
| Estimated 100 candidates | ~1.17h |
| Estimated 500 candidates | ~5.87h |

**Fallback kernels:** flash-linear-attention and causal_conv1d are not
installed (native Windows / Triton limitation). The model runs correctly
with reference PyTorch implementations but slower. Gate 0 measured this
rather than assuming optimized kernels.

**Full report:** `results/gate0_hardware.json` (gitignored, local-only).

## Running the preflight

```powershell
PYTHONPATH=ai-research/src ai-research\.venv\Scripts\python.exe ai-research\src\jarvis_research\preflight.py
```
