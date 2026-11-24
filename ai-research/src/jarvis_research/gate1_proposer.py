"""Exploratory Gate 1 self-proposer. Only practice feedback enters its prompt.

The caller owns the model: generate() operates on the same live object that
the patch manager edits. Selection/regression content is never passed here.
This is prompt isolation, NOT filesystem isolation from a tool-enabled agent.
"""

import hashlib
import json
import time

import torch


PROMPT = """You are proposing a modification to your own currently loaded weights.
Return exactly one JSON object, no explanation outside JSON:
{"edits":[{"param":"exact name","operation":"OP","magnitude":NUMBER,"count":INTEGER}],
 "rationale":"brief reason","predicted_effect":"brief prediction",
 "predicted_practice_delta_ce":NUMBER}
Use ONE edit per proposal. Smaller practice cross-entropy is better.
The only measurement you receive is PRACTICE feedback. You have no access to
selection, regression or final test results.
Allowed operation: OP. Parameter names must be chosen exactly from the list.
For add_sparse: count 1..512; |magnitude| <= min(0.5, 0.5*tensor_std).
For scale_dense: magnitude in [0.70, 0.99]; omit count.
Respect earlier rejection feedback. An invalid response consumes a step.
"""
PROMPT_VERSION = hashlib.sha256((PROMPT + "enable_thinking=False").encode()).hexdigest()
MAX_NEW = 512
TEMPERATURE = 0.7
TOP_P = 0.9


def parse_proposal(raw):
    """Try all complete JSON objects; never repair a malformed edit."""
    decoder = json.JSONDecoder()
    for pos, char in enumerate(raw):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "edits" in obj:
            return obj
    return None


class SelfProposer:
    def __init__(self, model, tokenizer, patch_manager, max_new=MAX_NEW):
        if model is not patch_manager.model:
            raise ValueError("proposer must use the edited target model")
        self.model = model
        self.tokenizer = tokenizer
        self.patch_manager = patch_manager
        self.max_new = max_new
        arm = patch_manager.arm
        self.names = sorted(
            n for n in patch_manager.params
            if patch_manager._classify_family(n) is not None
            and (patch_manager._layer_idx(n) is not None)
            and arm["min_layer"] <= patch_manager._layer_idx(n) <= arm["max_layer"])
        if not self.names:
            raise ValueError("no legal parameter names")

    def propose(self, seed, step, practice_ce, history):
        arm = self.patch_manager.arm
        # Limit prompt size without hiding early/mid/late options.
        layers = [0, 1, 12, 16, 22, 28, 31] if arm["min_layer"] == 0 else [12, 16, 22, 28, 31]
        names = [n for n in self.names if self.patch_manager._layer_idx(n) in layers]
        options = []
        for name in names:
            std = float(self.patch_manager.params[name].detach().float().std())
            options.append(f"{name} std={std:.5f}")
        user = (f"Arm: {arm['name']}; OP={arm['operation']}\n"
                f"Practice CE: {practice_ce:.6f}\n"
                f"Previous practice-only feedback: {json.dumps(history[-4:])}\n"
                f"Legal options:\n" + "\n".join(options))
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": PROMPT.replace("OP", arm["operation"])},
             {"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=False)
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        torch.manual_seed(seed * 100003 + step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            generated = self.model.generate(
                **encoded, do_sample=True, temperature=TEMPERATURE,
                top_p=TOP_P, max_new_tokens=self.max_new)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        raw = self.tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        return raw, parse_proposal(raw), elapsed
