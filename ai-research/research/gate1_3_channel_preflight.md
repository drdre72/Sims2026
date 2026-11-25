# HZN-OC-01 Preflight — Measured Channel Costs

**Component status (reviewer directive 2):** symbolic encoder ✅, tokenizer profiling ✅, observation audit layer ✅, comprehension harness (C1–C4) ✅ — implemented as `gate1_3_channel.py` + `gate1_3_comprehension.py`, 13/13 new tests, 41/41 suite. This document reports the first preflight measurement.

## 1. The tokenizer was right to be checked

Measured on the real campaign tokenizer over the real annotated catalogs (192 rows arm A, 248 rows arm B — the exact rendered surfaces):

| Level | Arm A tokens | /row | ratio | Arm B tokens | /row | ratio |
|---|---|---|---|---|---|---|
| linguistic | 8,774 | 45.7 | 1.00 | 8,863 | 35.7 | 1.00 |
| s1 (spaces, familiar keys) | 7,784 | 40.5 | **0.887** | 7,623 | 30.7 | **0.860** |
| s2 (`O:/F:/AR:/LC:` symbolic) | 8,461 | 44.1 | 0.964 | 8,443 | 34.0 | 0.953 |

**S2 as drafted loses to S1** — the symbolic vocabulary is shorter in *characters* but BPE fragments it. Character compression did not translate into token compression, as the reviewer predicted.

## 2. Where the tokens actually are

Per-row measurement, arm B row `1|L0|f1|n=81920|ar=0.04|lc=+0.06` (35.7 tok/row avg):

| Field group | Variant | Tokens |
|---|---|---|
| obs columns (ar+lc) | `\|ar=0.20\|lc=+0.01` | **15** |
| obs columns, s2 integer-coded | `\|AR:20\|LC:+1` | **9** |
| obs columns, s2 space-separated | `AR:20 LC:+1` | **7** |
| numel, full digits | `\|n=23592960` | **11** |
| numel, K-suffix | `\|N:23M` | **6** |
| per-row mag cap (arm A) | `\|m=0.004588` | **10** |
| mag cap, 2 significant figures | `\|M:0.0046` | **8** |
| full row, current | | **44** |
| full row, K-compact hybrid | | **37** |

Structural conclusion the reviewer anticipated and the data now confirms:

- **The observation columns are ~34–42% of row cost, not the majority.** The structural fields (param_id, layer, family, numel, per-row bounds) are the hog — `numel` alone ≈ 11 tok/row (~25%).
- **The current schema is already near the floor at full field precision.** Best measured hybrid ≈ −16%/row; with separators optimized ≈ −20–25% catalog-wide. The "10–30× compression" figure was wrong at this level — it priced the *information*, not the rendered surface.
- The deeper lever is not obs-column coding — it is **field precision and the non-observation transaction** (instructions, legend, history, output schema), consistent with the reviewer's Amdahl analysis.

## 3. Design questions the measurement forces (for the reviewer)

1. **numel precision** — `23592960` (11 tok) vs `23M` (6). numel is proposer-visible scale metadata; exact-value parity would be violated by K-suffix. Is a preregistered precision reduction (exact → order-of-magnitude) acceptable for this field?
2. **mag-cap precision** — `0.004588` (10 tok, arm A per-row) vs `0.0045` floor-rounded 2sf (8). Floor-rounding stays inside the legal bound but *reduces* the advertised surface; acceptable?
3. **numel at all?** — its role is edit-count feasibility (`count ≤ numel`). If the validator already enforces it, the proposer may only need coarse scale. Removal is a bigger cut than suffixing — also a field-parity violation requiring spec amendment.
4. **The 78–79% proposal-stage share** decomposes: catalog ≈ 8.8–8.9k tok of the ~9.4–9.5k serialized prompt ≈ **93% of the surface**. Instructions + arm context + history are only ~600 tok. So "compress the channel" ≈ compress the catalog; the catalog is ~85–90% structural fields.

## 4. Honest implications for HZN-OC-01

- The first S-II draft is dominated (S1 beats it on tokens). The encoding registry should evolve by measured token count — next candidates: space-separated symbolic (`O 17 L5 F7 N23M AR82 LC-3`), precision-relaxed structural variants pending §3 answers.
- The interesting scientific question shifts slightly: not "can we compress the evidence 10×" but "does a ~20% channel-cost reduction + identical evidence change anything?" — and, deeper, "is in-context catalog completeness needed at all" (the addressable-store arm now looks *more* attractive relative to compression than it did).
- Preflight gate for campaign design holds: an S-level must (a) preserve field parity or have each precision loss preregistered, (b) beat S1 on measured tokens, (c) pass C1–C4 comprehension, before any trajectory runs.

## 5. Artifacts

- `results/gate1_3_channel_profile.json` — full per-level profile + parity results
- `results/_profile_channels.py` — deterministic reproduction (model + obs pipeline, same as campaign)
- `src/jarvis_research/gate1_3_channel.py`, `gate1_3_comprehension.py`, `tests/test_gate1_3_channel.py`
