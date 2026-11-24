"""Exploratory Gate 1 self-proposer. Only practice feedback enters its prompt.

The caller owns the model: generate() operates on the same live object that
the patch manager edits. Selection/regression content is never passed here.
This is prompt isolation, NOT filesystem isolation from a tool-enabled agent.

Gate 1.1: the proposer sees the COMPLETE legal catalog — the identical list
of indexed parameter descriptors that every control draws from — so the
model and the controls search the same parameter space. Proposals reference
tensors by catalog param_id (an integer), which removes name-transcription
errors and lets the prompt carry shape/numel/std/legal-range metadata that
makes schema-valid proposals possible without trial and error.
"""

import hashlib
import json
import time

import torch


PROMPT = """You are proposing a modification to your own currently loaded weights.
Return exactly one JSON object, no explanation outside JSON:
{"edits":[{"param_id":INTEGER,"operation":"OP","magnitude":NUMBER,"count":INTEGER}],
 "rationale":"brief reason","predicted_effect":"brief prediction",
 "predicted_practice_delta_ce":NUMBER}
Use ONE edit per proposal. Smaller practice cross-entropy is better.
The only measurement you receive is PRACTICE feedback. You have no access to
selection, regression or final test results.
param_id MUST be an integer copied exactly from the LEGAL CATALOG below.
Catalog row format: id|layer|famcode|numel[|m=cap][|c<=N]
A 'families:' legend line maps each famcode to its full family name, and a
'bounds:' line states the arm-wide magnitude/count limits that apply to
every row. A per-row m=cap means magnitude must satisfy -cap <= m <= +cap;
a per-row c<=N means count must be <= N (otherwise the arm-wide count range
and numel apply).
magnitude MUST be inside its legal range; count MUST be inside the arm-wide
count range and <= numel (omit count for scale_dense).
Respect earlier rejection feedback. An invalid response consumes a step.
To deliberately make no change, return {"edits":[],"abstain":true} —
abstention is a valid outcome.
"""
# HZN-IR-04 §5: the abstention instruction lives in the BASE prompt so
# every A-condition (A0/A1/A2) shares it — an A1-vs-A0 difference then
# cannot be attributed to the instruction alone.
PROMPT_OBS = PROMPT + (
    "Observation columns ar/lc: ar is the within-family rank in [0,1] "
    "of mean output activation scale on practice text; lc is the "
    "signed, eps-stabilized log of the ratio of activation on high- vs "
    "low-loss practice text at 2-decimal precision. These are "
    "correlational measurements of the unmodified checkpoint, not "
    "causal evidence.\n")

PROMPT_VERSION = hashlib.sha256(
    (PROMPT + "enable_thinking=False|catalog-v5-compact").encode()).hexdigest()
PROMPT_VERSION_OBS = hashlib.sha256(
    (PROMPT_OBS + "enable_thinking=False|catalog-v7-obs-parity").encode()).hexdigest()
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


def _catalog_line(entry: dict, fam_code: dict, uniform_mag: bool,
                  obs: bool = False) -> str:
    """One compact catalog row: id|layer|famcode|numel[|mag_min|mag_max].

    The full parameter name is deliberately omitted — param_id is the
    authoritative reference and the patch manager resolves it. Family names
    are replaced by short codes defined in the legend line. Arm-wide
    magnitude/count bounds are stated once in the 'bounds:' legend, so they
    are only repeated per-row when they actually differ across tensors
    (sparse_additive std-relative caps). Keeping rows short matters: this
    hybrid linear-attention model's prefill memory grows steeply with
    prompt length (~12.7k tokens previously peaked near 32 GiB).
    """
    line = (f"{entry['param_id']}|L{entry['layer']}|{fam_code[entry['family']]}"
            f"|n={entry['numel']}")
    if not uniform_mag:
        # Advertise a cap 0.1% INSIDE the true bound: 4-sig-fig display
        # rounding can otherwise round up past the legal limit and every
        # boundary proposal fails validation on fp noise.
        if abs(entry["mag_min"] + entry["mag_max"]) < 1e-12:
            line += f"|m={entry['mag_max'] * 0.999:.4g}"
        else:
            line += (f"|mag[{entry['mag_min'] * 0.999:.4g},"
                     f"{entry['mag_max'] * 0.999:.4g}]")
    if "count_max" in entry and entry["count_max"] < 512:
        line += f"|c<={entry['count_max']}"
    if obs:
        # Proposer-visible evidence columns (HZN-IR-04 R1): ar = within-
        # family activation rank; lc = signed log-contrast at the same
        # 2-decimal precision interp-v2 consumes, so the model and the
        # conventional algorithm see a bit-identical contrast channel.
        # Raw stats stay in the machine-readable catalog only — a wider
        # prompt pushed the catalog to ~12k tokens on the first pass,
        # back inside the prefill envelope that OOM'd the first pilot.
        if entry.get("obs_available"):
            line += (f"|ar={entry['obs_act_rank']:.2f}"
                     f"|lc={entry['obs_log_contrast']:+.2f}")
        else:
            line += "|ar=NA|lc=NA"
    return line


class SelfProposer:
    def __init__(self, model, tokenizer, patch_manager, max_new=MAX_NEW,
                 catalog_override=None, obs: bool = False):
        if model is not patch_manager.model:
            raise ValueError("proposer must use the edited target model")
        self.model = model
        self.tokenizer = tokenizer
        self.patch_manager = patch_manager
        self.max_new = max_new
        self.obs = obs
        self.prompt_version = (PROMPT_VERSION_OBS if obs
                               else PROMPT_VERSION)
        # The COMPLETE legal catalog — identical to what controls draw
        # from. Observation-annotated (or shuffled) rows arrive via
        # catalog_override so structural fields never diverge.
        self.catalog = catalog_override or patch_manager.catalog
        if not self.catalog:
            raise ValueError("no legal parameter names")
        # Short family codes keep the prompt small; the legend maps them back.
        families = sorted({e["family"] for e in self.catalog})
        self._fam_code = {f: f"f{i}" for i, f in enumerate(families)}
        legend = ", ".join(f"{c}={f}" for f, c in self._fam_code.items())
        arm = patch_manager.arm
        mags = {(round(e["mag_min"], 8), round(e["mag_max"], 8))
                for e in self.catalog}
        uniform_mag = len(mags) == 1
        if uniform_mag:
            (lo, hi), = mags
            bounds = f"mag[{lo:.4g},{hi:.4g}]"
        else:
            bounds = "per-row m=cap means -cap<=m<=+cap"
        cnt = ""
        if "count_min" in arm:
            cnt = (f" count[{arm['min_count']},{arm['max_count']}]"
                   f" and <= numel")
        self._catalog_text = (f"families: {legend}\n"
                              f"bounds: {bounds}{cnt}\n"
                              + "\n".join(_catalog_line(
                                            e, self._fam_code,
                                            uniform_mag, self.obs)
                                          for e in self.catalog))

    def _user_text(self, step, practice_ce, history):
        arm = self.patch_manager.arm
        return (f"Arm: {arm['name']}; OP={arm['operation']}\n"
                f"Practice CE: {practice_ce:.6f}\n"
                f"Previous practice-only feedback: {json.dumps(history[-4:])}\n"
                f"LEGAL CATALOG ({len(self.catalog)} entries):\n"
                f"{self._catalog_text}")

    def _encode(self, step, practice_ce, history):
        """Build the fully serialized chat prompt (system + user)."""
        arm = self.patch_manager.arm
        prompt = PROMPT_OBS if self.obs else PROMPT
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": prompt.replace("OP", arm["operation"])},
             {"role": "user", "content": self._user_text(
                 step, practice_ce, history)}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=False)

    def serialized_prompt_tokens(self, step=0, practice_ce=0.0,
                                 history=None) -> int:
        """Token count of the COMPLETE serialized prompt — instructions,
        template framing and history included (HZN-IR-04 §5): the
        catalog-text-only figure underreports the real prefill cost.
        """
        encoded = self._encode(step, practice_ce, history or [])
        return int(encoded["input_ids"].shape[1])

    def propose(self, seed, step, practice_ce, history):
        encoded = self._encode(step, practice_ce, history)
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        torch.manual_seed(seed * 100003 + step)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        start = time.perf_counter()
        # Pin the memory-efficient SDPA backend: under allocator pressure
        # PyTorch may otherwise select the math backend, which materializes
        # an O(seq^2) attention matrix and makes long-prompt generation
        # pathologically slow on this card.
        sdpa_ctx = torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
             torch.nn.attention.SDPBackend.MATH])
        with sdpa_ctx, torch.no_grad():
            generated = self.model.generate(
                **encoded, do_sample=True, temperature=TEMPERATURE,
                top_p=TOP_P, max_new_tokens=self.max_new)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        raw = self.tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        return raw, parse_proposal(raw), elapsed
