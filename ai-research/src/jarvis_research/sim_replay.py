"""HZN-SIM-00.5 counterfactual replay preflight (CPU-only, deterministic).

Evaluates whether the HZN-SIM-00 substrate's structural retrieval would have
surfaced the documented gold landmarks at six historical decision points,
using only evidence available at each tick. Preregistration:
research/hzn_sim_00_5_replay_preflight.md; frozen decision-point table:
research/evidence/sim_00_5_decision_points_v1.json (committed before this
module existed). No LLM anywhere; seeded RNG only.
"""
import json
import math
import random
from pathlib import Path

from jarvis_research.sim_corpus import build_gold_corpus
from jarvis_research.sim_memory import (
    SimMemory, DEP_MODEL, DEP_HARNESS, DEP_EVAL, DEP_ONTOLOGY, DEP_MEASUREMENT,
    DEP_INTERPRETATION, DEP_TYPES,
)
from jarvis_research.cognitive_bus import canonical_json, sha256_text

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "research" / "evidence"
DECISION_POINTS_PATH = EVIDENCE_DIR / "sim_00_5_decision_points_v1.json"
SEED = 20261128
N_DRAWS = 1000

FRAME_KEY_TYPES = ("ontology_category", "dep_type")

MISS_NOT_AVAILABLE = "not_available_at_tick"
MISS_CATEGORY_ABSENT = "category_absent"
MISS_VOCABULARY_MISMATCH = "vocabulary_mismatch"
MISS_DEPENDENCY_NOT_DECLARED = "dependency_not_declared"
MISS_CAUSES = (MISS_NOT_AVAILABLE, MISS_CATEGORY_ABSENT, MISS_VOCABULARY_MISMATCH,
               MISS_DEPENDENCY_NOT_DECLARED)

VOCAB_CAUSES = (MISS_CATEGORY_ABSENT, MISS_VOCABULARY_MISMATCH)
MECH_CAUSES = (MISS_DEPENDENCY_NOT_DECLARED, MISS_NOT_AVAILABLE)


# ---------------------------------------------------------------------------
# Decision-point table: loading and validation
# ---------------------------------------------------------------------------
def load_decision_points(path=DECISION_POINTS_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dps = data["decision_points"]
    for dp in dps:
        if isinstance(dp.get("frame_minimal"), dict):
            dp["frame_minimal"] = [dp["frame_minimal"]]  # single-key form
        for field in ("dp_id", "tick", "decision", "frame_minimal", "frame_full",
                      "required_landmarks", "control"):
            if field not in dp:
                raise ValueError(f"{dp.get('dp_id', '?')}: missing field {field}")
        for frame, name in ((dp["frame_minimal"], "frame_minimal"),
                             (dp["frame_full"], "frame_full")):
            if not isinstance(frame, list) or not frame:
                raise ValueError(f"{dp['dp_id']}: {name} must be a non-empty list")
            for key in frame:
                if set(key) - set(FRAME_KEY_TYPES) or len(key) != 1:
                    raise ValueError(f"{dp['dp_id']}: bad frame key {key}")
        if set(k for k in _flatten(dp["frame_minimal"])) - set(_flatten(dp["frame_full"])):
            raise ValueError(f"{dp['dp_id']}: frame_minimal keys must be a subset of frame_full")
        if dp["control"] and dp["required_landmarks"]:
            raise ValueError(f"{dp['dp_id']}: control DP must carry no required landmarks")
    return data


def validate_decision_points(data, mem):
    """Guards the not_available_at_tick class and key existence before any run."""
    errors = []
    for dp in data["decision_points"]:
        for lm in dp["required_landmarks"] + dp.get("related_unscored", []):
            hyp = mem.hypothesis(lm)
            if hyp is None:
                errors.append(f"{dp['dp_id']}: unknown hypothesis {lm}")
            elif hyp.availability_tick > dp["tick"]:
                errors.append(f"{dp['dp_id']}: {lm} available at {hyp.availability_tick} > tick {dp['tick']}")
        for key in _flatten(dp["frame_full"]):
            if "ontology_category" in key and mem.ontology.category(key["ontology_category"]) is None:
                errors.append(f"{dp['dp_id']}: unknown category {key['ontology_category']}")
            if "dep_type" in key and key["dep_type"] not in DEP_TYPES:
                errors.append(f"{dp['dp_id']}: unknown dep_type {key['dep_type']}")
    return errors


def _flatten(frame):
    return [tuple(sorted(k.items())) for k in frame]


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def visible_hypotheses(mem, tick):
    return [h for h in mem.hypotheses() if h.availability_tick <= tick]


def frame_retrieve(mem, frame, at_tick, clearance="evolve"):
    """OR-union over per-key retrieve() calls — retrieve() takes a single
    ontology_category, so multi-category frames are unions of single-key queries.
    Multiple keys never conjoin (frozen frame semantics). Carries each
    candidate's retrieval-reported status for the status-fidelity check."""
    out = {}
    for key in frame:
        if "ontology_category" in key:
            results = mem.retrieve(ontology_category=key["ontology_category"],
                                   clearance=clearance, at_tick=at_tick)
        else:
            results = mem.retrieve(dep_type=key["dep_type"],
                                   clearance=clearance, at_tick=at_tick)
        for row in results:
            hid = row["hypothesis_id"]
            entry = out.setdefault(hid, {"match_keys": set(), "status": row["status"]})
            entry["match_keys"].update(row["match_keys"])
    return {hid: {"match_keys": sorted(v["match_keys"]), "status": v["status"]}
            for hid, v in sorted(out.items())}


def classify_miss(mem, dp, frame, landmark_id, visible_ids):
    """Exactly one cause per unsurfaced landmark (preregistered priority):
    not_available_at_tick > category_absent > vocabulary_mismatch >
    dependency_not_declared. vocabulary_mismatch requires the frame to have
    attempted category naming; dependency_not_declared fires when the frame
    keys on dependencies the landmark does not declare (including dep-only
    frames)."""
    hyp = mem.hypothesis(landmark_id)
    if hyp.availability_tick > dp["tick"]:
        return MISS_NOT_AVAILABLE
    categories = [k["ontology_category"] for k in frame if "ontology_category" in k]
    dep_types = [k["dep_type"] for k in frame if "dep_type" in k]
    if categories and not any(
            h.ontology_category == c for h in visible_ids for c in categories):
        return MISS_CATEGORY_ABSENT
    if categories and hyp.ontology_category not in categories:
        return MISS_VOCABULARY_MISMATCH
    if dep_types:
        interp = mem.interpretation(hyp.current_interpretation_id())
        if not any(d.dep_type in dep_types for d in interp.depends_on):
            return MISS_DEPENDENCY_NOT_DECLARED
    return MISS_VOCABULARY_MISMATCH


def analytic_hit_probability(n_visible, k, n_landmarks):
    """Exact probability that a uniform draw of k from n contains all landmarks."""
    if n_landmarks == 0:
        return 1.0
    if k < n_landmarks or n_visible < n_landmarks:
        return 0.0
    return math.comb(n_visible - n_landmarks, k - n_landmarks) / math.comb(n_visible, k)


def run_arm_R(visible_ids, required, k, seed, n_draws=N_DRAWS):
    """Seeded uniform sampling without replacement; returns sampled mean hit
    rate and per-landmark appearance rates."""
    rng = random.Random(seed)
    ids = sorted(visible_ids)
    hits = 0
    per_landmark = {lm: 0 for lm in required}
    if k >= len(ids):
        for _ in range(n_draws):
            hits += 1
            for lm in per_landmark:
                per_landmark[lm] += 1
    else:
        for _ in range(n_draws):
            draw = rng.sample(ids, k)
            if all(lm in draw for lm in required):
                hits += 1
            for lm in per_landmark:
                if lm in draw:
                    per_landmark[lm] += 1
    return {"sampled_hit_rate": hits / n_draws,
            "per_landmark_rate": {lm: per_landmark[lm] / n_draws for lm in required}}


def run_arm_Y(visible_hyps, required, k):
    """Top-k by availability_tick descending, ties by hypothesis_id ascending."""
    ordered = sorted(visible_hyps, key=lambda h: (-h.availability_tick, h.hypothesis_id))
    top = {h.hypothesis_id for h in ordered[:k]}
    return {"hit": all(lm in top for lm in required),
            "per_landmark": {lm: (lm in top) for lm in required}}


# ---------------------------------------------------------------------------
# Status fidelity (bug detector)
# ---------------------------------------------------------------------------
def status_fidelity(mem, dp, frame_result):
    """Check the reported *effective* status at the tick. The separate
    hand-derived status oracle tests the dependency evaluator itself; intrinsic
    status is not a valid oracle after a version bump or confound."""
    mismatches = []
    for hid, entry in frame_result.items():
        at_tick = mem.status_at_tick(hid, dp["tick"])
        if at_tick is not None and entry["status"] != at_tick:
            mismatches.append({"hypothesis_id": hid, "dp_tick": dp["tick"],
                               "at_tick_status": at_tick,
                               "reported_status": entry["status"]})
    return mismatches


# ---------------------------------------------------------------------------
# Cascade breadth probes
# ---------------------------------------------------------------------------
def _affected_by_status_change(mem):
    """Hypotheses whose effective status diverges from intrinsic after a
    registry/conference event — the mechanically propagated cascade."""
    affected = []
    for hyp in mem.hypotheses():
        iid = hyp.current_interpretation_id()
        intrinsic = mem.interpretation(iid).status
        if mem.hypothesis_status(hyp.hypothesis_id) != intrinsic:
            affected.append(hyp.hypothesis_id)
    return sorted(affected)


def cascade_probes():
    historical = {}
    mem = build_gold_corpus()
    bus_chain = set(mem.hypothesis("H-BUS-STALE-PRECEDENCE").interpretations)
    dep_on_bus = [h.hypothesis_id for h in mem.hypotheses()
                  if any(d.dep_type == DEP_INTERPRETATION and d.dep_id in bus_chain
                         for d in mem.interpretation(h.current_interpretation_id()).depends_on)]
    historical["H1_bus_chain_revision"] = {
        "affected": dep_on_bus, "count": len(dep_on_bus),
        "note": "no interpretation-to-interpretation edges exist in the corpus — dependency-declaration gap, not a substrate failure"}
    onto_deps = [h.hypothesis_id for h in mem.hypotheses()
                 if any(d.dep_type == DEP_ONTOLOGY
                        for d in mem.interpretation(h.current_interpretation_id()).depends_on)]
    historical["H2_ontology_revision"] = {
        "affected": onto_deps, "count": len(onto_deps),
        "note": "no interpretation declares an ontology dependency — gap finding"}

    def fresh():
        return build_gold_corpus()

    m = fresh()
    m.registry.replace(DEP_MODEL, "qwen3.5-4b-bf16", "qwen-next-4b", tick=110)
    p1 = _affected_by_status_change(m)
    m = fresh()
    m.registry.bump(DEP_HARNESS, "gate1_harness", tick=110)
    p2 = _affected_by_status_change(m)
    m = fresh()
    m.registry.bump(DEP_EVAL, "gate1_eval_v4", tick=110)
    p3 = _affected_by_status_change(m)
    m = fresh()
    m.register_confound("epi06-h1-nonnormal", "synthetic probe confound",
                        {"probe": "P4"})
    p4 = [h.hypothesis_id for h in m.hypotheses()
          if any(d.dep_type == DEP_MEASUREMENT and d.dep_id == "epi06-h1-nonnormal"
                 for d in m.interpretation(h.current_interpretation_id()).depends_on)]
    m = fresh()
    m.revise_ontology(effective_tick=110, note="synthetic probe v3")
    p5 = _affected_by_status_change(m)
    synthetic = {
        "P1_model_replacement": {"affected": p1, "count": len(p1)},
        "P2_harness_bump": {"affected": p2, "count": len(p2)},
        "P3_eval_bump": {"affected": p3, "count": len(p3)},
        "P4_confound_registration": {"affected": p4, "count": len(p4),
                                      "note": "no direct measurement dependencies declared — gap finding"},
        "P5_ontology_bump": {"affected": p5, "count": len(p5)},
    }
    return {"historical": historical, "synthetic": synthetic}


# ---------------------------------------------------------------------------
# Full preflight run
# ---------------------------------------------------------------------------
def run_preflight(decision_points_path=DECISION_POINTS_PATH):
    data = load_decision_points(decision_points_path)
    mem = build_gold_corpus()
    errors = validate_decision_points(data, mem)
    if errors:
        raise ValueError("decision-point table invalid: " + "; ".join(errors))

    dp_results = []
    scored = [dp for dp in data["decision_points"] if not dp["control"]]
    total_required = sum(len(dp["required_landmarks"]) for dp in scored)
    surfaced_required = 0
    full_hits = 0
    s_ge_r_dps = 0
    s_ge_y_dps = 0
    at_tick_violations = 0
    arm_l_future = {}
    candidate_fractions = []
    miss_causes = []

    for dp in data["decision_points"]:
        tick = dp["tick"]
        visible = visible_hypotheses(mem, tick)
        visible_ids = [h.hypothesis_id for h in visible]
        row = {"dp_id": dp["dp_id"], "tick": tick, "control": dp["control"],
               "n_visible": len(visible_ids)}
        frame_results = {}
        for level in ("frame_minimal", "frame_full"):
            candidates = frame_retrieve(mem, dp[level], at_tick=tick)
            frame_results[level] = candidates
            at_tick_violations += sum(
                1 for hid in candidates if mem.hypothesis(hid).availability_tick > tick)
            required = dp["required_landmarks"]
            surfaced = [lm for lm in required if lm in candidates]
            related = dp.get("related_unscored", [])
            landmarks = {}
            for lm in required:
                entry = {"surfaced": lm in candidates}
                if not entry["surfaced"]:
                    entry["miss_cause"] = classify_miss(mem, dp, dp[level], lm, visible)
                    if level == "frame_full" and not dp["control"]:
                        miss_causes.append(entry["miss_cause"])
                landmarks[lm] = entry
            for lm in related:
                landmarks[lm] = {"surfaced": lm in candidates, "unscored": True}
            row[level] = {
                "candidates": sorted(candidates),
                "size": len(candidates),
                "fraction_of_visible": (len(candidates) / len(visible_ids)) if visible_ids else 0.0,
                "hit": bool(required) and len(surfaced) == len(required),
                "partial": 0 < len(surfaced) < len(required),
                "landmarks": landmarks,
                "distractor_load": len(candidates) - len(surfaced) - sum(
                    1 for lm in related if lm in candidates),
            }
        full = row["frame_full"]
        row["status_mismatches"] = status_fidelity(mem, dp, frame_results["frame_full"])
        if not dp["control"]:
            surfaced_required += len([lm for lm in dp["required_landmarks"]
                                      if lm in full["candidates"]])
            if full["hit"]:
                full_hits += 1
                s_hit = 1
            else:
                s_hit = 0
            k = full["size"]
            n = len(visible_ids)
            r_analytic = analytic_hit_probability(n, k, len(dp["required_landmarks"]))
            r_sampled = run_arm_R(visible_ids, dp["required_landmarks"], k,
                                  SEED + int(dp["dp_id"].split("-")[1]))
            y = run_arm_Y(visible, dp["required_landmarks"], k)
            row["arms"] = {
                "k": k, "n_visible": n,
                "S_hit": s_hit,
                "R_analytic": r_analytic, "R_sampled": r_sampled,
                "Y_hit": y["hit"], "Y_per_landmark": y["per_landmark"],
                "S_ge_R": s_hit >= r_analytic, "S_ge_Y": s_hit >= (1 if y["hit"] else 0),
            }
            s_ge_r_dps += row["arms"]["S_ge_R"]
            s_ge_y_dps += row["arms"]["S_ge_Y"]
            candidate_fractions.append(full["fraction_of_visible"])
        # leakage probe: same frame-full keys, no availability filter
        unfiltered = frame_retrieve(mem, dp["frame_full"], at_tick=None)
        future = [hid for hid in unfiltered if mem.hypothesis(hid).availability_tick > tick]
        arm_l_future[dp["dp_id"]] = {"future_entries": sorted(future), "count": len(future)}
        dp_results.append(row)

    n_scored = len(scored)
    metrics = {
        "landmark_recall": surfaced_required / total_required,
        "surfaced_required": surfaced_required, "total_required": total_required,
        "full_hit_dps": full_hits, "scored_dps": n_scored,
        "s_ge_r_dps": s_ge_r_dps, "s_ge_y_dps": s_ge_y_dps,
        "mean_candidate_fraction": (sum(candidate_fractions) / len(candidate_fractions)
                                    if candidate_fractions else 0.0),
        "miss_causes": miss_causes,
    }
    leakage = {
        "at_tick_violations": at_tick_violations,
        "arm_L_future_counts": {k: v["count"] for k, v in arm_l_future.items()},
        "arm_L_validity": any(v["count"] > 0 for v in arm_l_future.values()),
    }
    verdict = evaluate_decision_logic(metrics, leakage, n_scored)
    result = {
        "preflight": "HZN-SIM-00.5", "version": 1, "seed": SEED, "n_draws": N_DRAWS,
        "spec": "research/hzn_sim_00_5_replay_preflight.md",
        "decision_points_path": str(decision_points_path),
        "decision_points": dp_results, "metrics": metrics, "leakage": leakage,
        "cascade": cascade_probes(), "verdict": verdict,
    }
    result["digest"] = sha256_text(canonical_json(result))
    return result


def evaluate_decision_logic(metrics, leakage, n_scored):
    """Mechanical application of preregistration section 6."""
    if leakage["at_tick_violations"] > 0 or not leakage["arm_L_validity"]:
        return {"verdict": "PREFLIGHT_INVALID",
                "reason": "leakage invariant violated or availability filter vacuous"}
    utility = (metrics["landmark_recall"] >= 5 / 7
               and metrics["full_hit_dps"] >= 3 / 5
               and metrics["s_ge_r_dps"] > n_scored / 2
               and metrics["s_ge_y_dps"] > n_scored / 2)
    if utility:
        if metrics["mean_candidate_fraction"] > 0.5:
            return {"verdict": "SUPPORTED_WITH_PRECISION_WARNING",
                    "reason": "recall achieved through breadth; SIM-01A must add deterministic narrowing/ranking before the loop"}
        return {"verdict": "RETRIEVAL_UTILITY_SUPPORTED",
                "reason": "all binding thresholds met"}
    causes = metrics["miss_causes"]
    vocab = sum(1 for c in causes if c in VOCAB_CAUSES)
    mech = sum(1 for c in causes if c in MECH_CAUSES)
    if vocab > mech:
        return {"verdict": "VOCABULARY_GAP_FAILURE",
                "reason": "misses dominated by category_absent/vocabulary_mismatch; ontology refinement before SIM-01A"}
    if mech > vocab:
        return {"verdict": "MECHANISM_FAILURE",
                "reason": "misses dominated by dependency_not_declared/not_available_at_tick"}
    return {"verdict": "INCONCLUSIVE", "reason": "mixed outcome outside preregistered bounds"}


def export_evidence(result, out_path):
    Path(out_path).write_text(json.dumps(result, indent=2, sort_keys=True),
                              encoding="utf-8")
