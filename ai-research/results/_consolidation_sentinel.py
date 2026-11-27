"""HZN-OOI-04S — limited GPU sentinel for autonomous consolidation.

Two independent feedback-condition runs over the same fixed nine-task campaign.
The environment offers neutral capabilities without instructing the model that
procedure construction is the objective. Inference-only; frozen backbone.
"""
import hashlib, json, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_5_consolidation_sentinel_v4.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from jarvis_research.gate1_5_consolidation import (
    MAX_TURNS_PER_TASK, SENTINEL_FEEDBACK, TemporalRuntime,
    build_sentinel_campaign, oracle_policy, run_runtime_episode,
    summarize_runtime)

MAX_GENERATIONS = 110
MAX_WALL_SECONDS = 20 * 60
MAX_NEW = 128
HARNESS_VERSION = "ooi04-sentinel-1"
SYSTEM = ("You are Jarvis operating in a bounded local inference environment. "
          "Follow the environment output contract exactly. Choose the most useful "
          "available action; no capability is mandatory.")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gen_text(model, tok, user_text, max_new=MAX_NEW):
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
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    lat = time.perf_counter() - t0
    ids = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(ids, skip_special_tokens=True), n_in, int(ids.shape[0]), lat


def snapshot(rt):
    return json.loads(json.dumps({
        "past": rt.past, "future": rt.future, "present": rt.present,
        "completed": rt.completed, "action_log": rt.action_log,
        "next_pid": rt._next_pid,
    }))


def restore(feedback, snap):
    rt = TemporalRuntime(feedback)
    rt.past = snap["past"]; rt.future = snap["future"]; rt.present = snap["present"]
    rt.completed = snap["completed"]; rt.action_log = snap["action_log"]
    rt._next_pid = snap["next_pid"]
    return rt


def summarize_model_run(rt, generations):
    s = summarize_runtime(rt)
    if generations:
        s["mean_input_tokens"] = sum(g["input_tokens"] for g in generations) / len(generations)
        s["mean_output_tokens"] = sum(g["output_tokens"] for g in generations) / len(generations)
        s["mean_latency_s"] = sum(g["latency_s"] for g in generations) / len(generations)
        s["total_input_tokens"] = sum(g["input_tokens"] for g in generations)
        s["total_output_tokens"] = sum(g["output_tokens"] for g in generations)
    s["n_generations"] = len(generations)
    return s


def main():
    tasks = build_sentinel_campaign()
    report = {"model": MODEL_ID, "harness": HARNESS_VERSION,
              "manifest": {}, "runs": {}, "oracle_reference": {}, "summary": {}}
    if OUT.exists():
        report = json.loads(OUT.read_text(encoding="utf-8"))
        print("[resume] loaded existing sentinel report", flush=True)

    provenance = local_provenance(MODEL_ID)
    report["manifest"] = {
        "model_revision": provenance["revision"], "harness_version": HARNESS_VERSION,
        "consolidation_version": 4, "feedback_conditions": list(SENTINEL_FEEDBACK),
        "n_tasks": len(tasks), "max_turns_per_task": MAX_TURNS_PER_TASK,
        "max_new": MAX_NEW,
        "campaign_sha256": hashlib.sha256(json.dumps(
            tasks, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    oracle = run_runtime_episode(TemporalRuntime("F1"), oracle_policy, tasks)
    report["oracle_reference"] = summarize_runtime(oracle)

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=provenance["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=provenance["revision"], dtype=DTYPE,
        device_map={"": "cuda"}, local_files_only=True)
    model.eval()

    t0_wall = time.perf_counter(); n_gen = 0
    def budget_ok():
        return n_gen < MAX_GENERATIONS and time.perf_counter() - t0_wall < MAX_WALL_SECONDS

    for feedback in SENTINEL_FEEDBACK:
        run = report["runs"].setdefault(feedback, {
            "status": "in_progress", "task_index": 0, "snapshot": None,
            "generations": [], "summary": {}})
        if run.get("status") == "complete":
            continue
        rt = restore(feedback, run["snapshot"]) if run.get("snapshot") else TemporalRuntime(feedback)
        start = int(run.get("task_index", 0))
        for ti in range(start, len(tasks)):
            if not budget_ok():
                run["status"] = "stopped_budget"; break
            # A snapshot already contains the active partially completed task.
            if not run.get("active_task") or run.get("task_index") != ti:
                rt.start_task(tasks[ti])
            run["active_task"] = True; run["task_index"] = ti
            while budget_ok():
                prompt = rt.render_prompt()
                resp, n_in, n_out, lat = gen_text(model, tok, prompt)
                n_gen += 1
                rec = rt.apply_response(resp)
                rec["input_tokens"] = n_in; rec["output_tokens"] = n_out
                rec["latency_s"] = round(lat, 2); rec["prompt_sha256"] = sha256_text(prompt)
                # Drop the large AST duplicate from the action log copy only after state use.
                run["generations"].append(json.loads(json.dumps(rec)))
                run["snapshot"] = snapshot(rt)
                OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
                print(f"[{feedback}:{tasks[ti]['episode_id']}:t{rec['turn']}] "
                      f"{rec['parsed'].get('action')} valid={rec['valid']} done={rec['completed']} "
                      f"correct={rec.get('correct')} {lat:.1f}s", flush=True)
                if rec["completed"]:
                    break
            if not budget_ok():
                run["status"] = "stopped_budget"; break
            # Ensure termination even if the inner loop hit budget exactly.
            if not rt.completed or rt.completed[-1]["task_id"] != tasks[ti]["episode_id"]:
                run["status"] = "stopped_incomplete_task"; break
            run["task_index"] = ti + 1; run["active_task"] = False
            run["snapshot"] = snapshot(rt)
            OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
        else:
            run["status"] = "complete"
        run["summary"] = summarize_model_run(rt, run["generations"])
        run["snapshot"] = snapshot(rt)
        OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    report["summary"] = {fb: report["runs"].get(fb, {}).get("summary", {})
                         for fb in SENTINEL_FEEDBACK}
    report["summary"]["oracle_reference"] = report["oracle_reference"]
    report["wall_seconds"] = round(time.perf_counter() - t0_wall, 1)
    report["n_model_generations"] = n_gen
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"done: {n_gen} generations, {time.perf_counter() - t0_wall:.0f}s wall", flush=True)
    print(json.dumps(report["summary"], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
