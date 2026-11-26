"""HZN-IR-10 sentinel — paired direct-ID semantic selection (inference only).

The decisive test the R3 experiment could not perform: the model selects
a row and returns its id DIRECTLY — the operation the edit loop actually
needs. No descriptor resolution stage, so descriptor collisions cannot
confound the measurement.

Design:
- Same answerable cases at 32 and 248 rows. "Answerable" = a UNIQUE
  winner at rendered precision over the full 248 catalog (rendered ties
  skipped, not scored). The correct id is IDENTICAL at both scales: the
  32-row condition presents the winner + 31 seeded distractors, so only
  distractor count differs (31 vs 247).
- Question types (HZN-IR-10 D3): largest lc, smallest lc, largest |lc|,
  largest abs_lc (explicit alc= column in every row — tests whether the
  failure is the absolute-value operation vs scanning), largest ar,
  smallest ar. Distinct prompts throughout — no repeated greedy probes.
- Multi-row transcription control at 32 rows: ordinal copy probes
  (R2 critique — the original copy test showed one row at a time).
- Single consistent output instruction; prompt/catalog/model binding in
  a run manifest (D6). Resumable per cell.
"""
import hashlib, json, random, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_3_direct_select.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_3_comprehension import synthetic_catalog
from jarvis_research.gate1_3_channel import canonical_record
from jarvis_research.gate1_3_store import ObservationStore, catalog_sha256
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance

SEEDS = [104, 106, 108, 111]  # chosen by CPU scan: maximal answerable-type coverage at 248 rows (rendered ties are structural); original 101/202 contribute too few answerable cases
N_DISTRACTORS = 31
SYSTEM = ("You are reading an observation catalog from a self-editing "
          "experiment. Answer each probe precisely. Reply with ONLY the "
          "requested format — no reasoning, no extra text.")
HARNESS_VERSION = "direct-select-1"

QTYPES = [
    ("largest_lc", "largest lc", lambda v: max(v), "lc"),
    ("smallest_lc", "smallest lc", lambda v: min(v), "lc"),
    ("largest_abs_lc", "largest |lc|", lambda v: max(abs(x) for x in v), "lc"),
    ("largest_alc_col", "largest alc", lambda v: max(v), "alc"),
    ("largest_ar", "largest ar", lambda v: max(v), "ar"),
    ("smallest_ar", "smallest ar", lambda v: min(v), "ar"),
]


def render_rows(rows, fam_code, alc=False):
    """Linguistic rows, optionally with an explicit alc= column. Values
    at rendered precision; alc is mathematically exact at that
    precision (D3 requirement)."""
    lines = []
    for r in rows:
        ar = round(r["obs_act_rank"], 2) if r["obs_available"] else None
        lc = round(r["obs_log_contrast"], 2) if r["obs_available"] else None
        base = (f"{r['param_id']}|L{r['layer']}|{fam_code[r['family']]}"
                f"|n={r['numel']}")
        if r["obs_available"]:
            base += f"|ar={ar:.2f}|lc={lc:.2f}"
            if alc:
                base += f"|alc={abs(lc):.2f}"
        else:
            base += "|no-obs"
        lines.append(base)
    fams = "; ".join(f"{c}={f}" for f, c in sorted(fam_code.items()))
    return f"families: {fams}\n" + "\n".join(lines)


def gen_text(model, tok, user_text, max_new=12):
    enc = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user_text}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_in = int(enc["input_ids"].shape[1])
    sdpa = torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
         torch.nn.attention.SDPBackend.MATH])
    t0 = time.perf_counter()
    with sdpa, torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    lat = time.perf_counter() - t0
    text = tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text, n_in, lat


def winner_for(cat, qtype):
    """Unique winner id at rendered precision over the full catalog;
    None when tied (unanswerable — skipped, never scored)."""
    obs = [r for r in cat if r["obs_available"]]
    if qtype in ("largest_abs_lc", "largest_alc_col"):
        vals = [abs(round(r["obs_log_contrast"], 2)) for r in obs]
    elif qtype in ("largest_lc", "smallest_lc"):
        vals = [round(r["obs_log_contrast"], 2) for r in obs]
    else:  # ar types
        vals = [round(r["obs_act_rank"], 2) for r in obs]
    target = min(vals) if qtype.startswith("smallest") else max(vals)
    winners = [r for r, v in zip(obs, vals) if v == target]
    if len(winners) != 1:
        return None, vals
    return winners[0]["param_id"], vals


def build_probes(cat, seed):
    """All probes for one seed. Returns list of dicts with paired
    probe_key across sizes."""
    fam_code = {f: f"f{i}" for i, f in
                enumerate(sorted({r["family"] for r in cat}))}
    probes = []
    for qtype, phrasing, _, key in QTYPES:
        win, vals = winner_for(cat, qtype)
        if win is None:
            probes.append({"probe_key": f"{seed}:{qtype}:skip",
                           "qtype": qtype, "skipped": "rendered_tie"})
            continue
        alc = key == "alc"
        win_row = next(r for r in cat if r["param_id"] == win)
        rng = random.Random(int.from_bytes(
            hashlib.sha256(f"sub:{seed}:{qtype}".encode()).digest()[:8],
            "little"))
        pool = [r for r in cat if r["param_id"] != win]
        distractors = rng.sample(pool, N_DISTRACTORS)
        sub = sorted(distractors + [win_row], key=lambda r: r["param_id"])
        # sanity: winner must still win within the subset at rendered
        # precision (guaranteed by construction — distractors are a
        # subset of the full catalog's losers)
        for size, rows in ((32, sub), (248, cat)):
            txt = render_rows(rows, fam_code, alc=alc)
            q = (f"Which row has the {phrasing}? Reply with ONLY the "
                 f"id number.")
            probes.append({
                "probe_key": f"{seed}:{qtype}:{size}",
                "qtype": qtype, "phrasing": phrasing, "size": size,
                "expected": win, "n_rows": len(rows),
                "target_pos": rows.index(win_row),
                "prompt_text": txt + "\n\n" + q})
    # Multi-row transcription controls (R2 critique): ordinal copy
    # within ANY answerable 32-row subset (seed 101's largest-|lc| case
    # is a rendered tie and gets skipped, so anchor on the first
    # available one instead).
    anchor = next((p for p in probes
                   if p.get("size") == 32 and not p.get("skipped")),
                  None)
    if anchor is not None:
        rows_txt = anchor["prompt_text"].rsplit("\n\n", 1)[0]
        row_lines = rows_txt.splitlines()[1:]  # skip families: header
        for pos in (1, 9, 17, 31):
            if pos <= len(row_lines):
                true_id = int(row_lines[pos - 1].split("|")[0])
                q = (f"What is the leading id number of row {pos} "
                     f"(counting the first data row as row 1)? Reply "
                     f"with ONLY the number.")
                probes.append({
                    "probe_key": f"{seed}:copy_pos{pos}:32",
                    "qtype": "copy_transcription", "size": 32,
                    "expected": true_id, "n_rows": len(row_lines),
                    "target_pos": pos - 1,
                    "prompt_text": rows_txt + "\n\n" + q})
    return probes


def grade_id(expected, resp):
    import re
    t = resp.strip()
    if not re.fullmatch(r"[+-]?\d+", t):
        return ("refusal" if t.lower() in ("ambiguous", "not present")
                else "malformed")
    return "correct" if int(t) == expected else "wrong_id"


def main():
    report = {"model": MODEL_ID, "harness": HARNESS_VERSION,
              "manifest": {}, "cells": {}}
    if OUT.exists():
        report = json.loads(OUT.read_text())
        print(f"[resume] {len(report['cells'])} cells", flush=True)

    provenance = local_provenance(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=provenance["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=provenance["revision"], dtype=DTYPE,
        device_map={"": "cuda"}, local_files_only=True)
    model.eval()

    cats = {s: synthetic_catalog(248, seed=s) for s in SEEDS}
    canon = {s: [canonical_record(r, True, (-0.005, 0.005)) for r in c]
             for s, c in cats.items()}
    report["manifest"] = {
        "model_revision": provenance["revision"],
        "harness_version": HARNESS_VERSION,
        "catalog_sha256": {str(s): catalog_sha256(canon[s]) for s in SEEDS},
        "seeds": SEEDS, "n_distractors": N_DISTRACTORS,
    }

    for s in SEEDS:
        for probe in build_probes(cats[s], s):
            key = probe["probe_key"]
            if probe.get("skipped"):
                report["cells"][key] = probe
                continue
            if key in report["cells"] and "verdict" in report["cells"][key]:
                print(f"[skip] {key}", flush=True)
                continue
            torch.cuda.reset_peak_memory_stats()
            prompt_hash = hashlib.sha256(
                probe["prompt_text"].encode()).hexdigest()[:16]
            resp, n_in, lat = gen_text(model, tok, probe["prompt_text"])
            verdict = grade_id(probe["expected"], resp)
            if verdict == "wrong_id":
                import re as _re
                valid = any(r["param_id"] == int(resp.strip())
                            for r in cats[s]) if _re.fullmatch(
                                r"[+-]?\d+", resp.strip()) else False
                verdict = ("wrong_id_valid" if valid
                           else "wrong_id_nonexistent")
            cell = {k: v for k, v in probe.items() if k != "prompt_text"}
            cell.update({"response": resp, "verdict": verdict,
                         "prompt_sha256": prompt_hash,
                         "input_tokens": n_in,
                         "latency_s": round(lat, 2),
                         "peak_vram_mb": int(
                             torch.cuda.max_memory_allocated() / 1e6)})
            report["cells"][key] = cell
            OUT.write_text(json.dumps(report, indent=1))
            print(f"[{key}] {verdict} exp={probe['expected']} "
                  f"resp={resp.strip()[:30]!r} {lat:.1f}s", flush=True)
    print("done -> results/gate1_3_direct_select.json")


if __name__ == "__main__":
    sys.exit(main())
