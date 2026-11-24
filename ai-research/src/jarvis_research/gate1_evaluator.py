#!/usr/bin/env python3
"""gate1_evaluator.py — protected evaluation partitions for Gate 1.

The evaluator is the scientific contract for Gate 1. It defines four
text partitions with strictly controlled access:

  PRACTICE   — visible to the proposer during proposal formulation.
                Used for the proposer's feedback signal ONLY.
  SELECTION  — the primary promotion metric. NEVER shown to the proposer.
                A candidate must improve here to advance.
  REGRESSION — per-domain capability checks. A candidate must not regress
                beyond the preregistered tolerance on ANY domain.
  FINAL      — not present in this process. Requires an independently
                administered evaluator and OS-enforced access isolation.

Access contract:
  - This process keeps selection/regression out of the model prompt. It does
    NOT provide filesystem isolation; confirmation requires a separate
    evaluator identity and access policy.

Metrics (reviewer v2 requirements):
  - Token-weighted aggregate perplexity: exp(sum_NLL / sum_tokens), NOT
    arithmetic mean of per-passage perplexities.
  - Per-domain cross-entropy loss (for regression tolerance thresholds).
  - Per-passage perplexity + token count (for diagnostics).
  - GPU-seconds instrumentation on every evaluation call.

Usage:
    from gate1_evaluator import Evaluator
    ev = Evaluator.load(practice_path, selection_path, regression_path)
    score = ev.evaluate_selection(model, tok)  # returns SelectionScore
    reg = ev.evaluate_regression(model, tok)   # returns RegressionReport
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_THIS = Path(__file__).resolve()
AI_RESEARCH = _THIS.parents[2]
SIMS_ROOT = AI_RESEARCH.parent

# No final-set path is known to this development process.

# Domains used for regression checks. Each domain has multiple passages.
REGRESSION_DOMAINS = [
    "science", "code", "narrative", "math", "history",
    "technical", "conversation", "factual",
]


@dataclass
class PassageResult:
    name: str
    domain: str
    ppl: float
    nll_sum: float
    n_tokens: int


@dataclass
class DomainResult:
    domain: str
    aggregate_ppl: float  # token-weighted: exp(sum_nll / sum_tokens)
    mean_ppl: float        # arithmetic mean (for reference only)
    n_passages: int
    n_tokens: int
    per_passage: List[PassageResult] = field(default_factory=list)
    cross_entropy: float = 0.0


@dataclass
class SelectionScore:
    """Primary promotion metric. Computed on the SELECTION partition."""
    aggregate_ppl: float      # token-weighted across all selection passages
    n_passages: int
    n_tokens: int
    per_passage: List[PassageResult] = field(default_factory=list)
    gpu_seconds: float = 0.0
    sha256: str = ""           # hash of the selection text set (for audit)
    cross_entropy: float = 0.0


@dataclass
class RegressionReport:
    """Per-domain capability check. Computed on the REGRESSION partition."""
    per_domain: Dict[str, DomainResult] = field(default_factory=dict)
    gpu_seconds: float = 0.0
    n_domains: int = 0
    n_passages: int = 0


@dataclass
class PracticeReport:
    """Feedback signal for the proposer. Computed on the PRACTICE partition.
    This is the ONLY partition the proposer sees."""
    aggregate_ppl: float
    n_passages: int
    n_tokens: int
    per_passage: List[PassageResult] = field(default_factory=list)
    gpu_seconds: float = 0.0


class Evaluator:
    """Protected evaluation partitions for Gate 1.

    The Evaluator holds references to practice, selection, and regression
    text partitions. The FINAL (sealed) partition is NOT loaded here —
    it is loaded only by the confirmatory evaluator.
    """

    def __init__(
        self,
        practice_texts: Dict[str, str],
        selection_texts: Dict[str, str],
        regression_texts: Dict[str, Dict[str, str]],
        practice_meta: Optional[dict] = None,
        selection_meta: Optional[dict] = None,
        regression_meta: Optional[dict] = None,
    ):
        # practice_texts: {name: text}
        # selection_texts: {name: text}
        # regression_texts: {domain: {name: text}}
        self.practice = practice_texts
        self.selection = selection_texts
        self.regression = regression_texts
        self.practice_meta = practice_meta or {}
        self.selection_meta = selection_meta or {}
        self.regression_meta = regression_meta or {}
        assert_disjoint_partitions(practice_texts, selection_texts, regression_texts)

    @classmethod
    def load(
        cls,
        practice_path: Path,
        selection_path: Path,
        regression_path: Path,
    ) -> "Evaluator":
        """Load partitions from JSON files. Each file has a defined schema."""
        practice_data = json.loads(practice_path.read_text(encoding="utf-8"))
        selection_data = json.loads(selection_path.read_text(encoding="utf-8"))
        regression_data = json.loads(regression_path.read_text(encoding="utf-8"))

        if (practice_path.resolve() == selection_path.resolve()
                or practice_path.resolve() == regression_path.resolve()
                or selection_path.resolve() == regression_path.resolve()):
            raise ValueError("evaluation partition paths must differ")
        practice_texts = {k: v for k, v in practice_data["passages"].items()}
        selection_texts = {k: v for k, v in selection_data["passages"].items()}
        # Regression passages are written flat (e.g. "science_regression_1")
        # and grouped into domains by the prefix before the first underscore.
        regression_flat = {k: v for k, v in regression_data["passages"].items()}
        regression_texts: Dict[str, Dict[str, str]] = {}
        for name, text in regression_flat.items():
            domain = name.split("_")[0]
            regression_texts.setdefault(domain, {})[name] = text

        return cls(
            practice_texts=practice_texts,
            selection_texts=selection_texts,
            regression_texts=regression_texts,
            practice_meta=practice_data.get("meta", {}),
            selection_meta=selection_data.get("meta", {}),
            regression_meta=regression_data.get("meta", {}),
        )

    @staticmethod
    def _hash_texts(texts: Dict[str, str]) -> str:
        h = hashlib.sha256()
        for k in sorted(texts.keys()):
            h.update(k.encode())
            h.update(b"\x00")
            h.update(texts[k].encode())
            h.update(b"\x00")
        return h.hexdigest()[:16]

    @torch.no_grad()
    def _eval_passages(self, model, tok, texts: Dict[str, str], domain: str = ""
                       ) -> Tuple[List[PassageResult], float, int, float]:
        """Evaluate a set of passages. Returns (results, total_nll, total_tokens, gpu_s)."""
        t0 = time.perf_counter()
        results: List[PassageResult] = []
        total_nll = 0.0
        total_tokens = 0
        for name, text in texts.items():
            ids = tok(text, return_tensors="pt").input_ids.to("cuda")
            if ids.shape[1] < 2:
                continue
            out = model(ids)
            logits = out.logits.float()
            sl, lab = logits[:, :-1, :], ids[:, 1:]
            loss_sum = torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.size(-1)), lab.reshape(-1), reduction="sum")
            n = lab.numel()
            nll = float(loss_sum.item())
            ppl = float(torch.exp(loss_sum / n).item()) if torch.isfinite(loss_sum) else float("inf")
            total_nll += nll
            total_tokens += n
            results.append(PassageResult(
                name=name, domain=domain, ppl=round(ppl, 4),
                nll_sum=round(nll, 4), n_tokens=n,
            ))
        gpu_s = time.perf_counter() - t0
        return results, total_nll, total_tokens, gpu_s

    @torch.no_grad()
    def evaluate_practice(self, model, tok) -> PracticeReport:
        """Feedback signal for the proposer. PRACTICE partition only."""
        results, total_nll, total_tokens, gpu_s = self._eval_passages(
            model, tok, self.practice)
        agg_ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)).item()) if total_tokens > 0 else float("inf")
        return PracticeReport(
            aggregate_ppl=round(agg_ppl, 4),
            n_passages=len(results),
            n_tokens=total_tokens,
            per_passage=results,
            gpu_seconds=round(gpu_s, 4),
        )

    @torch.no_grad()
    def evaluate_selection(self, model, tok) -> SelectionScore:
        """Primary promotion metric. SELECTION partition. NEVER shown to proposer."""
        results, total_nll, total_tokens, gpu_s = self._eval_passages(
            model, tok, self.selection)
        agg_ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)).item()) if total_tokens > 0 else float("inf")
        return SelectionScore(
            aggregate_ppl=round(agg_ppl, 4),
            n_passages=len(results),
            n_tokens=total_tokens,
            per_passage=results,
            gpu_seconds=round(gpu_s, 4),
            sha256=self._hash_texts(self.selection),
            cross_entropy=total_nll / total_tokens if total_tokens else float("inf"),
        )

    @torch.no_grad()
    def evaluate_regression(self, model, tok) -> RegressionReport:
        """Per-domain capability check. REGRESSION partition."""
        t0 = time.perf_counter()
        per_domain: Dict[str, DomainResult] = {}
        total_passages = 0
        for domain_name, passages in self.regression.items():
            results, total_nll, total_tokens, _ = self._eval_passages(
                model, tok, passages, domain=domain_name)
            agg_ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)).item()) if total_tokens > 0 else float("inf")
            mean_ppl = sum(r.ppl for r in results) / len(results) if results else float("inf")
            per_domain[domain_name] = DomainResult(
                domain=domain_name,
                aggregate_ppl=round(agg_ppl, 4),
                mean_ppl=round(mean_ppl, 4),
                n_passages=len(results),
                n_tokens=total_tokens,
                per_passage=results,
                cross_entropy=total_nll / total_tokens if total_tokens else float("inf"),
            )
            total_passages += len(results)
        gpu_s = time.perf_counter() - t0
        return RegressionReport(
            per_domain=per_domain,
            gpu_seconds=round(gpu_s, 4),
            n_domains=len(per_domain),
            n_passages=total_passages,
        )

    def to_dict(self) -> dict:
        return {
            "practice": {"n_passages": len(self.practice), "meta": self.practice_meta},
            "selection": {"n_passages": len(self.selection), "meta": self.selection_meta},
            "regression": {
                "n_domains": len(self.regression),
                "domains": {d: len(p) for d, p in self.regression.items()},
                "meta": self.regression_meta,
            },
        }


def check_regression(
    baseline: RegressionReport,
    candidate: RegressionReport,
    tolerance: Dict[str, float],
) -> Tuple[bool, Dict[str, dict]]:
    """Check whether candidate passes regression tolerances.

    Tolerance is per-domain, expressed as max acceptable INCREASE in
    aggregate cross-entropy (NLL per token), NOT perplexity ratio.
    Cross-entropy is preferred (reviewer: "thresholds on changes in
    cross-entropy loss, supplemented by perplexity ratios").

    A candidate PASSES if no domain's cross-entropy increase exceeds
    its tolerance. Returns (passed, per_domain_details).
    """
    details = {}
    passed = True
    for domain, tol in tolerance.items():
        base_dom = baseline.per_domain.get(domain)
        cand_dom = candidate.per_domain.get(domain)
        if base_dom is None or cand_dom is None:
            details[domain] = {"status": "missing", "passed": False}
            passed = False
            continue
        # Cross-entropy per token = ln(ppl). Delta in CE per token.
        base_ce = base_dom.cross_entropy
        cand_ce = cand_dom.cross_entropy
        delta_ce = cand_ce - base_ce
        ppl_ratio = cand_dom.aggregate_ppl / base_dom.aggregate_ppl if base_dom.aggregate_ppl > 0 else float("inf")
        domain_passed = delta_ce <= tol
        details[domain] = {
            "baseline_ce": round(base_ce, 6),
            "candidate_ce": round(cand_ce, 6),
            "delta_ce": round(delta_ce, 6),
            "tolerance_ce": tol,
            "baseline_ppl": base_dom.aggregate_ppl,
            "candidate_ppl": cand_dom.aggregate_ppl,
            "ppl_ratio": round(ppl_ratio, 4),
            "passed": domain_passed,
        }
        if not domain_passed:
            passed = False
    return passed, details


# ---------------------------------------------------------------------------
# Partition generation. Creates the three non-sealed partitions.
# The FINAL (sealed) partition is generated separately and stored in
# gate1_sealed/ outside the ai-research tree.
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode("utf-8")).hexdigest()


def assert_disjoint_partitions(practice, selection, regression):
    """Fail closed on identical normalized content across any partitions."""
    seen = {}
    for label, texts in (
        ("practice", practice), ("selection", selection),
        ("regression", {k: v for domain in regression.values()
                        for k, v in domain.items()}),
    ):
        for name, text in texts.items():
            digest = _text_hash(text)
            if digest in seen:
                raise ValueError(f"partition overlap: {seen[digest]} and {label}/{name}")
            seen[digest] = f"{label}/{name}"


def generate_partitions(
    practice_path: Path,
    selection_path: Path,
    regression_path: Path,
    n_per_domain_practice: int = 2,
    n_per_domain_selection: int = 3,
    n_per_domain_regression: int = 3,
    seed: int = 42,
    pools: Optional[Dict[str, List[str]]] = None,
) -> dict:
    """Generate practice/selection/regression partitions.

    All text is original, written for Gate 1. Each domain gets passages
    allocated to the three partitions. The passages are distinct across
    partitions — no overlap.

    `pools` may supply an alternate per-domain passage list (e.g. the v3
    pool, which is disjoint from the examined v2 passages). When omitted,
    the default development pool is used.

    The FINAL sealed partition is NOT generated here. It must be generated
    by a separate process and stored outside this account's reach.
    """
    # Development pools cover the three exploratory partitions ONLY. They
    # cannot produce an independently administered confirmation set. NOT
    # the seed303_validate.py passages (those are now "examined").
    pools = pools if pools is not None else _build_passage_pools()
    paths = [practice_path, selection_path, regression_path]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite an existing evaluation partition")

    import random
    rng = random.Random(seed)

    allocations = {"practice": {}, "selection": {}, "regression": {}}
    sizes = {"practice": n_per_domain_practice,
             "selection": n_per_domain_selection,
             "regression": n_per_domain_regression}
    if any(n < 1 for n in sizes.values()):
        raise ValueError("partition sizes must be positive")
    for domain in REGRESSION_DOMAINS:
        pool = pools[domain][:]
        if sum(sizes.values()) > len(pool):
            raise ValueError(f"not enough passages for {domain}")
        rng.shuffle(pool)
        offset = 0
        for label, count in sizes.items():
            for i, text in enumerate(pool[offset:offset + count], 1):
                allocations[label][f"{domain}_{label}_{i}"] = text
            offset += count
    regression_groups = {domain: {
        k: v for k, v in allocations["regression"].items()
        if k.startswith(f"{domain}_")
    } for domain in REGRESSION_DOMAINS}
    assert_disjoint_partitions(allocations["practice"], allocations["selection"],
                               regression_groups)
    summary = {"seed": seed, "partitions": {}}
    for label, path, n_per in [
        ("practice", practice_path, n_per_domain_practice),
        ("selection", selection_path, n_per_domain_selection),
        ("regression", regression_path, n_per_domain_regression),
    ]:
        passages = allocations[label]
        data = {
            "meta": {
                "partition": label,
                "n_passages": len(passages),
                "n_domains": len(REGRESSION_DOMAINS),
                "seed": seed,
                "generator": "gate1_evaluator.generate_partitions",
                "note": ("Original text for Gate 1. NOT the seed303_validate.py "
                         "passages (those are examined diagnostic data)."),
            },
            "passages": passages,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        summary["partitions"][label] = {
            "path": str(path),
            "n_passages": len(passages),
        }

    return summary


def _build_passage_pools() -> Dict[str, List[str]]:
    """Build pools of original passages per domain for Gate 1.

    Each domain has 12 development passages. All text is original, written
    for this evaluation, not derived from any prior dataset.
    """
    pools: Dict[str, List[str]] = {}
    # Each domain defined inline. The passages are distinct from all
    # prior experiments (seed303_validate.py DIAGNOSTIC and SEALED sets).
    pools["science"] = [
        ("The greenhouse effect works because certain gases in the atmosphere "
         "absorb infrared radiation emitted by the Earth's surface. Sunlight "
         "passes through, warms the ground, and the warm ground radiates heat "
         "back upward. Greenhouse gases like carbon dioxide and methane trap "
         "some of this outgoing radiation, warming the lower atmosphere."),
        ("Stars are classified by their spectral type, which indicates their "
         "surface temperature. The sequence from hottest to coolest runs "
         "O, B, A, F, G, K, M. Our Sun is a G-type star with a surface "
         "temperature of about 5800 Kelvin. Red dwarfs, the most common stars "
         "in the galaxy, are M-type with temperatures below 4000 Kelvin."),
        ("DNA replication is semiconservative: each new double helix contains "
         "one original strand and one newly synthesized strand. The enzyme "
         "helicase unwinds the double helix, and DNA polymerase adds "
         "nucleotides complementary to the template strand. Replication "
         "proceeds in the 5 prime to 3 prime direction on both strands."),
        ("The periodic table organizes elements by atomic number, which is the "
         "count of protons in the nucleus. Elements in the same column share "
         "chemical properties because they have the same number of valence "
         "electrons. The noble gases in the rightmost column are inert because "
         "their outer electron shell is completely filled."),
        ("Tectonic plates move at rates of a few centimeters per year, about "
         "the speed fingernails grow. Convection currents in the mantle "
         "drive this motion. Where plates pull apart, magma rises to form "
         "new crust at mid-ocean ridges. Where they collide, mountains form "
         "or one plate sinks beneath the other in subduction zones."),
        ("Enzymes are biological catalysts that speed up chemical reactions "
         "without being consumed. They work by lowering the activation energy "
         "needed for a reaction. Each enzyme has an active site that binds "
         "specific substrates, like a lock and key. Temperature and pH affect "
         "enzyme activity, with each enzyme having an optimal range."),
        ("The speed of light in a vacuum is a fundamental constant, "
         "approximately 299792458 meters per second. According to special "
         "relativity, nothing with mass can reach this speed. As an object "
         "approaches light speed, its relativistic mass increases without "
         "bound, requiring infinite energy to accelerate further."),
        ("Neurons communicate through electrochemical signals. When a neuron "
         "receives sufficient stimulation, it fires an action potential, a "
         "brief electrical impulse that travels down the axon. At the synapse, "
         "the signal crosses to the next neuron via chemical neurotransmitters. "
         "The brain has roughly 86 billion neurons forming trillions of "
         "synaptic connections."),
        ("Genetic drift is the random fluctuation of allele frequencies in a "
         "population. Unlike natural selection, which is non-random, drift is "
         "especially strong in small populations. The founder effect occurs "
         "when a new population is established by a small number of "
         "individuals, carrying only a fraction of the original genetic "
         "diversity."),
        ("Acid rain forms when sulfur dioxide and nitrogen oxides react with "
         "water vapor in the atmosphere to produce sulfuric and nitric acids. "
         "These acids fall as precipitation, lowering the pH of lakes and "
         "soils. Acid rain damages forests, acidifies waterways harming fish, "
         "and corrodes buildings and statues made of limestone and marble."),
        ("The first law of thermodynamics states that energy cannot be created "
         "or destroyed, only transformed from one form to another. In any "
         "process, the total energy of the system plus its surroundings "
         "remains constant. This principle underlies all energy accounting in "
         "physics, chemistry, and biology."),
        ("Corals are animals that build reefs by secreting calcium carbonate "
         "skeletons. They host symbiotic algae called zooxanthellae, which "
         "provide food through photosynthesis and give corals their color. "
         "When ocean temperatures rise, corals expel these algae in a process "
         "called bleaching, which can be fatal if stress persists."),
    ]
    pools["code"] = [
        ("def merge_sort(arr):\n"
         "    if len(arr) <= 1:\n"
         "        return arr\n"
         "    mid = len(arr) // 2\n"
         "    left = merge_sort(arr[:mid])\n"
         "    right = merge_sort(arr[mid:])\n"
         "    return merge(left, right)"),
        ("try:\n"
         "    result = 10 / 0\n"
         "except ZeroDivisionError as e:\n"
         "    print(f\"Cannot divide by zero: {e}\")\n"
         "finally:\n"
         "    print(\"Cleanup complete\")"),
        ("with open('data.txt', 'r') as f:\n"
         "    lines = f.readlines()\n"
         "    for i, line in enumerate(lines):\n"
         "        print(f\"{i}: {line.strip()}\")"),
        ("import numpy as np\n"
         "matrix = np.random.randn(4, 4)\n"
         "eigenvalues = np.linalg.eigvals(matrix)\n"
         "print(f\"Eigenvalues: {eigenvalues}\")\n"
         "# Real matrices can have complex eigenvalues\n"
         "# when the matrix is not symmetric."),
        ("def factorial(n):\n"
         "    \"\"\"Return n! for non-negative n.\"\"\"\n"
         "    if n < 0:\n"
         "        raise ValueError(\"n must be non-negative\")\n"
         "    result = 1\n"
         "    for i in range(2, n + 1):\n"
         "        result *= i\n"
         "    return result"),
        ("from collections import defaultdict\n"
         "word_count = defaultdict(int)\n"
         "for word in text.split():\n"
         "    word_count[word.lower()] += 1\n"
         "# defaultdict(int) automatically initializes\n"
         "# missing keys to 0, avoiding KeyErrors."),
        ("class Stack:\n"
         "    def __init__(self):\n"
         "        self.items = []\n"
         "    def push(self, item):\n"
         "        self.items.append(item)\n"
         "    def pop(self):\n"
         "        if not self.items:\n"
         "            raise IndexError(\"pop from empty stack\")\n"
         "        return self.items.pop()"),
        ("def is_palindrome(s):\n"
         "    s = s.lower().replace(' ', '')\n"
         "    return s == s[::-1]\n"
         "# Uses Python slice notation [::-1]\n"
         "# which reverses the string.\n"
         "print(is_palindrome('race car'))  # True"),
        ("async def fetch_all(urls):\n"
         "    import aiohttp\n"
         "    async with aiohttp.ClientSession() as session:\n"
         "        tasks = [session.get(url) for url in urls]\n"
         "        responses = await asyncio.gather(*tasks)\n"
         "        return [await r.text() for r in responses]"),
        ("def quicksort(arr):\n"
         "    if len(arr) <= 1:\n"
         "        return arr\n"
         "    pivot = arr[len(arr) // 2]\n"
         "    left = [x for x in arr if x < pivot]\n"
         "    mid = [x for x in arr if x == pivot]\n"
         "    right = [x for x in arr if x > pivot]\n"
         "    return quicksort(left) + mid + quicksort(right)"),
        ("import json\n"
         "config = json.loads('{\"debug\": true, \"port\": 8080}')\n"
         "if config['debug']:\n"
         "    print(f\"Running on port {config['port']}\")\n"
         "# json.loads parses a JSON string into\n"
         "# a Python dictionary."),
        ("def fibonacci_memo(n, memo={}):\n"
         "    if n in memo:\n"
         "        return memo[n]\n"
         "    if n <= 1:\n"
         "        return n\n"
         "    memo[n] = fibonacci_memo(n-1) + fibonacci_memo(n-2)\n"
         "    return memo[n]"),
    ]
    pools["narrative"] = [
        ("The detective studied the crime scene photographs for the third time. "
         "Something was wrong with the shadows. If the murder happened at noon, "
         "as the coroner claimed, the shadow of the building should fall "
         "westward. But in the photo it stretched east, as if the sun were "
         "setting. She picked up the phone and called the coroner's office."),
        ("The old fisherman mended his nets by candlelight, calloused fingers "
         "working the twine with practiced ease. Forty years he had launched "
         "his boat before dawn, and forty years the sea had given him just "
         "enough to survive. Tomorrow his grandson would join him for the "
         "first time, and he wondered if the boy would understand the patience "
         "that fishing demanded."),
        ("Maria pressed her forehead against the cold window of the train and "
         "watched the countryside blur past. Three hours to her new city, her "
         "new life. In her lap she clutched a worn photograph of her parents "
         "waving goodbye at the platform, their faces already becoming "
         "memories. She wondered if she had made the right choice."),
        ("The soldier crept through the ruined building, rifle raised, "
         "listening for any sound over the distant rumble of artillery. Dust "
         "filtered down from the shattered ceiling with each step. In the next "
         "room he found a child's teddy bear, half buried in rubble, its glass "
         "eyes staring at nothing. He tucked it into his pack and moved on."),
        ("The clockmaker's shop smelled of brass and old wood. Shelves lined "
         "every wall, holding timepieces of every shape and size, all ticking "
         "in a synchronized chorus. The old man bent over his workbench, "
         "tweezers in hand, positioning a gear smaller than a grain of rice. "
         "He had repaired this particular clock once before, sixty years ago."),
        ("Rain hammered the tin roof of the cabin as Sarah fed wood to the "
         "stove. Outside, the trail had turned to mud, and the river was rising. "
         "She checked the radio one more time, but the storm had knocked out "
         "the signal hours ago. She would have to wait until morning to try "
         "the road, if the road was still there."),
        ("The professor adjusted her glasses and looked out at the lecture "
         "hall. Two hundred students, and she could already tell which ones "
         "would thrive and which would struggle. The first problem set would "
         "separate them soon enough. She began writing equations on the board, "
         "her chalk tapping a steady rhythm in the sudden quiet."),
        ("From the cliff top, the valley spread out like a patchwork quilt of "
         "green and gold fields. A river wound through its center, catching "
         "the late afternoon light. Tomas had walked three days to reach this "
         "spot, and now he understood why his father had always spoken of it "
         "with such reverence. This was where their family had begun."),
        ("The cat watched the bird feeder with the patience of a predator. "
         "It had sat motionless for twenty minutes, its eyes tracking every "
         "movement. The sparrows were cautious, landing and taking off in "
         "quick bursts, but eventually one lingered too long on the feeder. "
         "The cat coiled, then launched itself in a single fluid motion."),
        ("The astronaut floated in the airlock, watching the pressure gauge. "
         "In ten minutes she would step outside for her first spacewalk. "
         "Through the small window, the Earth hung like a blue marble against "
         "the black. She checked her tether one last time and took a slow "
         "breath, savoring the recycled air that was about to become vacuum."),
        ("The letter lay unopened on the kitchen table for three days. "
         "Eleanor knew what it contained because she recognized the handwriting "
         "on the envelope. It was from David, and David only wrote when he had "
         "something difficult to say. She finally picked it up, slit it open, "
         "and began to read the words that would change everything."),
        ("The stage lights came up and the theater went silent. Anna took "
         "one breath, then another, and stepped forward into the spotlight. "
         "She had rehearsed this monologue five hundred times, but tonight "
         "the words felt different, as if she were hearing them for the first "
         "time. She opened her mouth and Hamlet's grief became her own."),
    ]
    pools["math"] = [
        ("To solve the quadratic equation x squared minus five x plus six "
         "equals zero, we can factor it as x minus two times x minus three "
         "equals zero. Setting each factor to zero gives x equals two or x "
         "equals three. We can verify by substituting: two squared minus "
         "five times two plus six equals four minus ten plus six equals zero."),
        ("The Pythagorean theorem states that in a right triangle, the square "
         "of the hypotenuse equals the sum of the squares of the other two "
         "sides. If the legs measure three and four units, the hypotenuse "
         "squared equals nine plus sixteen, which is twenty five, so the "
         "hypotenuse measures five units. This relationship holds for every "
         "right triangle."),
        ("To find the derivative of x cubed, we apply the power rule: "
         "multiply by the exponent and reduce it by one. The derivative is "
         "three x squared. Geometrically, this gives the slope of the tangent "
         "line at any point on the curve. At x equals two, the slope is "
         "three times four, which is twelve."),
        ("The integral of two x with respect to x is x squared plus a "
         "constant. We can verify by differentiating x squared, which gives "
         "two x. The constant of integration represents the family of "
         "functions that all have the same derivative, differing only by a "
         "vertical shift."),
        ("In a standard deck of fifty two cards, the probability of drawing "
         "an ace on the first pick is four divided by fifty two, which "
         "simplifies to one thirteenth. If the first card is not replaced, "
         "the probability of drawing a second ace becomes three divided by "
         "fifty one. This is sampling without replacement."),
        ("The natural logarithm of e is exactly one, because the logarithm "
         "asks the question: e raised to what power equals e? The answer is "
         "one. Similarly, the natural log of one is zero, because any number "
         "raised to the zero power equals one. These are fundamental "
         "properties of logarithms."),
        ("A geometric series has the form one plus r plus r squared plus r "
         "cubed and so on. If the absolute value of r is less than one, the "
         "series converges to one divided by one minus r. For example, with "
         "r equal to one half, the sum is one divided by one minus one half, "
         "which equals two."),
        ("The mean of a set of numbers is found by summing all values and "
         "dividing by the count. The median is the middle value when sorted. "
         "For the numbers one, three, five, seven, and nine, the mean is "
         "twenty five divided by five, which is five, and the median is also "
         "five. Mean and median are equal only in symmetric distributions."),
        ("Matrix multiplication is not commutative: A times B generally does "
         "not equal B times A. To multiply two matrices, the number of columns "
         "in the first must equal the number of rows in the second. Each "
         "element of the product is the dot product of a row of the first "
         "matrix with a column of the second."),
        ("The chain rule is used to differentiate composite functions. If "
         "y equals f of g of x, then the derivative of y with respect to x "
         "equals the derivative of f with respect to g times the derivative "
         "of g with respect to x. For example, if y equals sin of x squared, "
         "then y prime equals two x times cosine of x squared."),
        ("An arithmetic sequence increases by a constant difference. The "
         "sequence two, five, eight, eleven has a common difference of three. "
         "The n-th term is found by the formula a plus n minus one times d, "
         "where a is the first term and d is the common difference. The "
         "tenth term is two plus nine times three, which is twenty nine."),
        ("The fundamental theorem of calculus connects differentiation and "
         "integration. It states that the definite integral of a function "
         "from a to b equals the antiderivative evaluated at b minus the "
         "antiderivative at a. This means integration and differentiation are "
         "inverse operations, a profound connection in calculus."),
    ]
    pools["history"] = [
        ("The Magna Carta, signed in 1215 by King John of England, limited "
         "royal power and established that even the king was subject to law. "
         "It arose from disputes between the king and rebellious barons over "
         "taxation and arbitrary rule. Though initially a peace treaty, it "
         "became a foundation for constitutional governance."),
        ("The fall of the Berlin Wall in November 1989 marked the symbolic "
         "end of the Cold War. For nearly thirty years the wall had divided "
         "East and West Berlin, splitting families and trapping millions "
         "behind the Iron Curtain. Its fall was triggered by mass protests "
         "and a bureaucratic error about travel restrictions."),
        ("The ancient Silk Road was not a single road but a network of "
         "trade routes spanning over six thousand kilometers. Caravans "
         "carried silk, spices, and precious metals between China, India, "
         "Persia, and the Mediterranean. The routes also spread ideas, "
         "religions, and diseases across continents."),
        ("The Harlem Renaissance was a cultural movement in the 1920s that "
         "celebrated African American art, literature, and music. Figures "
         "like Langston Hughes, Zora Neale Hurston, and Duke Ellington "
         "created works that explored Black identity and challenged racial "
         "stereotypes. The movement centered in Harlem, New York."),
        ("The French Revolution began in 1789 with the storming of the "
         "Bastille, a prison that symbolized royal tyranny. The revolution "
         "dismantled the feudal system, declared the Rights of Man, and "
         "eventually executed King Louis XVI. The Reign of Terror followed, "
         "during which thousands were executed by guillotine."),
        ("The Treaty of Westphalia in 1648 ended the Thirty Years War, one "
         "of the longest and most destructive conflicts in European history. "
         "It established the principle of state sovereignty, that each ruler "
         "could determine the religion of their own territory. This shaped "
         "the modern international system of nation states."),
        ("Gutenberg's printing press, invented around 1440, revolutionized "
         "Europe by making books affordable and widely available. Before "
         "the press, books were copied by hand, making them rare and "
         "expensive. The press enabled the rapid spread of new ideas, "
         "fueling the Renaissance and the Protestant Reformation."),
        ("The transatlantic slave trade transported over twelve million "
         "enslaved Africans to the Americas between the sixteenth and "
         "nineteenth centuries. The trade fueled the economies of European "
         "colonial powers while devastating African societies. Abolition "
         "movements gained momentum in the late eighteenth century."),
        ("The Meiji Restoration of 1868 transformed Japan from a feudal "
         "society into a modern industrial nation. Emperor Meiji oversaw "
         "reforms that abolished the samurai class, established compulsory "
         "education, and built railroads and factories. Within decades, "
         "Japan emerged as a world power."),
        ("The Apollo eleven mission landed humans on the Moon for the first "
         "time on July twentieth, 1969. Neil Armstrong and Buzz Aldrin "
         "walked on the lunar surface while Michael Collins orbited above. "
         "The mission fulfilled President Kennedy's goal of reaching the "
         "Moon before the end of the decade."),
        ("The Black Death, a plague pandemic in the fourteenth century, "
         "killed an estimated one third of Europe's population. The disease, "
         "spread by fleas on rats, caused fever and swollen lymph nodes. "
         "The massive death toll disrupted feudal labor systems, giving "
         "survivors more bargaining power and accelerating social change."),
        ("The Declaration of Independence, adopted on July fourth, 1776, "
         "announced the American colonies' separation from Britain. Drafted "
         "primarily by Thomas Jefferson, it asserted that all men are created "
         "equal and have unalienable rights to life, liberty, and the "
         "pursuit of happiness. These ideals would inspire revolutions "
         "worldwide."),
    ]
    pools["technical"] = [
        ("Batch normalization normalizes the inputs of each layer to have "
         "zero mean and unit variance, then applies a learned scale and "
         "shift. This stabilizes training by reducing internal covariate "
         "shift. It also allows higher learning rates and acts as a mild "
         "regularizer, often reducing the need for dropout."),
        ("Recurrent neural networks process sequences by maintaining a hidden "
         "state that is updated at each time step. The hidden state encodes "
         "information from previous inputs. Long Short Term Memory networks "
         "add gates to control what information is stored or forgotten, "
         "addressing the vanishing gradient problem in long sequences."),
        ("Dropout is a regularization technique that randomly sets a fraction "
         "of activations to zero during training. This prevents neurons "
         "from co-adapting too much, forcing the network to learn more robust "
         "features. At test time, all neurons are active and outputs are "
         "scaled to match the training expectation."),
        ("The Adam optimizer combines momentum and adaptive learning rates. "
         "It maintains running averages of both the gradient and the squared "
         "gradient, then uses these to scale the update for each parameter. "
         "Adam is widely used because it often works well with minimal "
         "hyperparameter tuning."),
        ("In a decision tree, each internal node tests a feature against a "
         "threshold, and branches represent outcomes of the test. Leaf nodes "
         "give predictions. Random forests combine many trees trained on "
         "bootstrap samples with random feature subsets, reducing overfitting "
         "compared to a single tree."),
        ("Cross entropy loss measures the difference between two probability "
         "distributions. For classification, it compares the predicted "
         "probability distribution against the true label. Minimizing cross "
         "entropy is equivalent to maximum likelihood estimation. The loss "
         "is smooth and provides useful gradients even when predictions are "
         "confident and wrong."),
        ("The ReLU activation function outputs the input directly if positive, "
         "and zero if negative. It is cheap to compute and does not saturate "
         "for positive values, unlike sigmoid or tanh. However, neurons can "
         "die if their inputs are consistently negative, producing zero "
         "gradient and stopping learning."),
        ("Transfer learning takes a model trained on one task and adapts it "
         "to a related task. The early layers, which learn general features, "
         "are usually frozen or fine tuned with a small learning rate. Only "
         "the final layers are retrained from scratch. This works when the "
         "source and target tasks share underlying structure."),
        ("Gradient clipping prevents exploding gradients in deep networks "
         "or recurrent models. If the gradient norm exceeds a threshold, "
         "the gradient is rescaled so its norm equals the threshold. This "
         "keeps updates stable without changing their direction. It is "
         "especially useful in training recurrent networks on long sequences."),
        ("The softmax function converts a vector of real numbers into a "
         "probability distribution. Each element is exponentiated and then "
         "normalized by the sum of all exponentials. The output values are "
         "positive and sum to one. Softmax is differentiable, making it "
         "suitable as the final layer in classification networks."),
        ("Attention mechanisms allow a model to focus on relevant parts of "
         "the input. In self attention, each token computes a weighted sum "
         "of all other tokens. The weights come from query key dot products. "
         "Multi head attention runs several attention computations in "
         "parallel, letting the model attend to different relationships "
         "simultaneously."),
        ("Data augmentation artificially expands a training set by applying "
         "transformations to existing examples. For images, common "
         "augmentations include flipping, cropping, rotating, and color "
         "adjustment. For text, techniques include synonym replacement and "
         "back translation. Augmentation reduces overfitting and improves "
         "generalization to unseen variations."),
    ]
    pools["conversation"] = [
        ("Alex: Did you finish the quarterly report?\n"
         "Sam: Not yet. I need the sales figures from the regional offices.\n"
         "Alex: The deadline is tomorrow morning.\n"
         "Sam: I will email the offices today and work late if needed.\n"
         "Alex: Let me know if you need help with the charts."),
        ("Mom: What do you want for your birthday?\n"
         "Kid: A bicycle! A red one with basket on the front.\n"
         "Mom: We will see. Have you been good this year?\n"
         "Kid: The best! I cleaned my room every week.\n"
         "Mom: Every single week? I find that hard to believe."),
        ("Boss: The client wants the prototype by Friday.\n"
         "Dev: That is tight. Which features are essential?\n"
         "Boss: Login, dashboard, and basic reporting.\n"
         "Dev: I can do login and dashboard. Reporting needs the backend.\n"
         "Boss: Then prioritize login and dashboard first."),
        ("Teacher: Your essay had some excellent arguments.\n"
         "Student: Thank you. I spent a lot of time researching.\n"
         "Teacher: But the conclusion felt rushed. Can you expand it?\n"
         "Student: I was not sure how to tie everything together.\n"
         "Teacher: Try restating your thesis and summarizing your main points."),
        ("Friend: Want to grab dinner this weekend?\n"
         "Me: Saturday works. That new Italian place on Main Street?\n"
         "Friend: I heard their pasta is amazing. Seven o'clock?\n"
         "Me: Perfect. Should I make a reservation?\n"
         "Friend: Good idea. It gets busy on weekends."),
        ("Doctor: How are you feeling after the medication change?\n"
         "Patient: Much better. The headaches are almost gone.\n"
         "Doctor: Any side effects? Nausea or dizziness?\n"
         "Patient: A little dizzy in the mornings, but it fades.\n"
         "Doctor: That should improve over the next week."),
        ("Sarah: Have you decided on a college yet?\n"
         "Tom: I am torn between two. One has a great program, but the other "
         "is closer to home.\n"
         "Sarah: What matters more to you?\n"
         "Tom: The program, I think. But being far from home scares me.\n"
         "Sarah: You can always visit on breaks."),
        ("Customer: Excuse me, where is the produce section?\n"
         "Worker: Aisle four, on the left near the back wall.\n"
         "Customer: And do you have organic apples?\n"
         "Worker: Yes, they are in the refrigerated case at the end.\n"
         "Customer: Thank you. One more thing, where can I find almond milk?"),
        ("Coach: You need to work on your backhand.\n"
         "Player: I know. It keeps going into the net.\n"
         "Coach: You are not following through. Swing up and across.\n"
         "Player: Like this?\n"
         "Coach: Better. Keep your wrist firm and contact the ball in front."),
        ("Dad: How was school today?\n"
         "Daughter: Good. We started a project on volcanoes.\n"
         "Dad: Volcanoes are fascinating. Which one are you studying?\n"
         "Daughter: Mount Vesuvius. It destroyed Pompeii.\n"
         "Dad: I remember learning about that. Did people know it was coming?"),
        ("Riley: Are you coming to the concert tonight?\n"
         "Jordan: I want to, but tickets sold out in minutes.\n"
         "Riley: My brother has two extras. He cannot go anymore.\n"
         "Jordan: Really? How much does he want for them?\n"
         "Riley: Face value. He is not trying to profit from it."),
        ("Manager: The team seems stressed lately.\n"
         "Lead: Two big deadlines landed in the same week.\n"
         "Manager: Can we shift one to next week?\n"
         "Lead: The client presentation, maybe. The internal review is fixed.\n"
         "Manager: Let me talk to the client about the timeline."),
    ]
    pools["factual"] = [
        ("The Great Wall of China stretches over twenty one thousand "
         "kilometers, though much of it is no longer intact. Construction "
         "began in the seventh century BC and continued for two millennia. "
         "The wall was built to defend against nomadic invasions from the "
         "north. It is one of the largest construction projects in history."),
        ("Honey is the only food that does not spoil. Jars found in ancient "
         "Egyptian tombs over three thousand years old were still edible. "
         "Honey's low water content and acidic pH create an environment where "
         "bacteria cannot grow. Bees produce honey by regurgitating nectar "
         "and evaporating most of its water content."),
        ("Venus is the hottest planet in the solar system despite not being "
         "closest to the Sun. Its thick atmosphere of carbon dioxide traps "
         "heat through a runaway greenhouse effect, raising surface "
         "temperatures to over four hundred sixty degrees Celsius. Mercury, "
         "closer to the Sun, lacks an atmosphere to retain heat."),
        ("The human body has about five liters of blood, which circulates "
         "through roughly one hundred thousand kilometers of blood vessels. "
         "Red blood cells live for about one hundred twenty days and carry "
         "oxygen using hemoglobin. The heart pumps blood through a vast "
         "network of arteries, capillaries, and veins."),
        ("Antarctica is the coldest, driest, and windiest continent. It "
         "holds about sixty percent of the world's fresh water locked in "
         "its ice sheet. Despite its size, it has no permanent residents, "
         "only researchers at scientific stations. In winter, temperatures "
         "can drop below minus eighty degrees Celsius."),
        ("Octopuses have three hearts: two pump blood through the gills and "
         "one circulates it through the rest of the body. Their blood is "
         "blue because it uses copper based hemocyanin instead of iron based "
         "hemoglobin to transport oxygen. Octopuses are also highly "
         "intelligent, capable of solving puzzles and using tools."),
        ("The Amazon rainforest produces about twenty percent of the world's "
         "oxygen and is home to roughly ten percent of all known species. "
         "It spans nine countries, with the majority in Brazil. Deforestation "
         "for cattle ranching and agriculture threatens this biodiversity. "
         "The Amazon River is the largest by volume in the world."),
        ("Lightning is hotter than the surface of the Sun, reaching "
         "temperatures of about thirty thousand degrees Celsius. A bolt "
         "can carry up to one billion volts of electricity. Lightning strikes "
         "the Earth about one hundred times per second globally. Thunder is "
         "the sound of air rapidly expanding as it is heated by the "
         "lightning channel."),
        ("A group of flamingos is called a flamboyance. Flamingos get their "
         "pink color from carotenoids in the algae and crustaceans they eat. "
         "Baby flamingos are born gray and only develop pink plumage after "
         "about three years. They stand on one leg to reduce heat loss "
         "through their legs."),
        ("The Mariana Trench in the western Pacific is the deepest point in "
         "the ocean, reaching nearly eleven thousand meters below sea level. "
         "The pressure at the bottom is over one thousand times atmospheric "
         "pressure. Despite these extreme conditions, life exists there, "
         "including amphipods and single celled organisms adapted to the "
         "crushing depth."),
        ("Jupiter is the largest planet in our solar system, with a mass "
         "more than twice that of all other planets combined. Its Great Red "
         "Spot is a storm that has raged for at least four hundred years. "
         "Jupiter has ninety five known moons, including Ganymede, the "
         "largest moon in the solar system, which is bigger than Mercury."),
        ("Bananas are botanically classified as berries, while strawberries "
         "are not. A berry develops from a single flower with one ovary and "
         "contains multiple seeds. Bananas fit this definition, but "
         "strawberries develop from a flower with multiple ovaries. This "
         "shows that common names and scientific classifications often "
         "disagree."),
    ]
    # Verify each domain has 12 passages
    for domain, passages in pools.items():
        assert len(passages) == 12, f"{domain} has {len(passages)}, expected 12"
    return pools


def _build_passage_pools_v3() -> Dict[str, List[str]]:
    """Gate 1.1 pool: 8 fresh passages per domain, DISJOINT from the v2 pool.

    The v2 exploratory pilot examined its practice/selection/regression
    passages, so v3 cannot reuse them. Each domain gets exactly 8 passages —
    enough for 2 practice + 3 selection + 3 regression with no remainder.
    All text is original.
    """
    pools: Dict[str, List[str]] = {}
    pools["science"] = [
        ("The ozone layer sits in the stratosphere roughly twenty to thirty "
         "kilometers above the surface and absorbs most of the sun's harmful "
         "ultraviolet radiation. Its discovery followed observations that "
         "skin cancer rates were rising in regions with intense sunlight, "
         "prompting scientists to investigate what was filtering the rays."),
        ("Mitochondria are organelles often called the powerhouse of the "
         "cell because they generate most of the cell's supply of ATP "
         "through cellular respiration. They have their own DNA, which "
         "supports the endosymbiotic theory that they were once free "
         "living bacteria that entered a larger cell."),
        ("A black hole forms when a massive star collapses under its own "
         "gravity at the end of its life. The collapse is so complete that "
         "not even light can escape from within the event horizon. Black "
         "holes are detected indirectly through their effects on nearby "
         "matter, which heats up and emits X rays as it spirals inward."),
        ("The water cycle describes how water moves between the atmosphere, "
         "oceans, and land. Solar energy evaporates water from surfaces, "
         "the vapor condenses into clouds, and precipitation returns water "
         "to the ground. Some water runs off into rivers and streams, some "
         "soaks into the ground to recharge aquifers, and the cycle repeats."),
        ("Sound travels as a longitudinal wave, meaning the particles of the "
         "medium vibrate parallel to the direction the wave travels. In air "
         "at room temperature, sound moves at about 343 meters per second. "
         "It travels faster in denser media like water and steel because "
         "the particles are closer together and transmit vibrations more "
         "efficiently."),
        ("Photosynthesis converts light energy into chemical energy stored "
         "in glucose. Plants absorb carbon dioxide and water, and using "
         "sunlight captured by chlorophyll, they produce glucose and "
         "release oxygen as a byproduct. This process is the foundation of "
         "most food chains and produces nearly all the oxygen in the "
         "atmosphere."),
        ("The human immune system distinguishes between the body's own cells "
         "and foreign invaders like bacteria and viruses. White blood cells "
         "patrol the bloodstream and tissues, and when they encounter a "
         "pathogen they trigger an inflammatory response. Vaccines work by "
         "presenting a harmless version of a pathogen so the immune system "
         "learns to recognize and fight the real one."),
        ("Plate tectonics explains that the Earth's outer shell is divided "
         "into rigid plates that float on the semi molten mantle below. These "
         "plates move over geological time, driven by convection currents. "
         "Their interactions produce earthquakes, volcanic eruptions, and "
         "mountain ranges at the boundaries where they meet."),
    ]
    pools["code"] = [
        ("def quicksort(arr):\n"
         "    if len(arr) <= 1:\n"
         "        return arr\n"
         "    pivot = arr[len(arr) // 2]\n"
         "    left = [x for x in arr if x < pivot]\n"
         "    middle = [x for x in arr if x == pivot]\n"
         "    right = [x for x in arr if x > pivot]\n"
         "    return quicksort(left) + middle + quicksort(right)"),
        ("class Stack:\n"
         "    def __init__(self):\n"
         "        self.items = []\n"
         "    def push(self, item):\n"
         "        self.items.append(item)\n"
         "    def pop(self):\n"
         "        if self.is_empty():\n"
         "            return None\n"
         "        return self.items.pop()\n"
         "    def is_empty(self):\n"
         "        return len(self.items) == 0"),
        ("def fibonacci(n):\n"
         "    if n <= 1:\n"
         "        return n\n"
         "    a, b = 0, 1\n"
         "    for _ in range(2, n + 1):\n"
         "        a, b = b, a + b\n"
         "    return b"),
        ("def is_anagram(s1, s2):\n"
         "    if len(s1) != len(s2):\n"
         "        return False\n"
         "    return sorted(s1.replace(' ', '')) == sorted(s2.replace(' ', ''))"),
        ("def two_sum(nums, target):\n"
         "    seen = {}\n"
         "    for i, num in enumerate(nums):\n"
         "        complement = target - num\n"
         "        if complement in seen:\n"
         "            return [seen[complement], i]\n"
         "        seen[num] = i\n"
         "    return []"),
        ("def flatten(lst):\n"
         "    result = []\n"
         "    for item in lst:\n"
         "        if isinstance(item, list):\n"
         "            result.extend(flatten(item))\n"
         "        else:\n"
         "            result.append(item)\n"
         "    return result"),
        ("def count_vowels(s):\n"
         "    vowels = 'aeiouAEIOU'\n"
         "    return sum(1 for c in s if c in vowels)"),
        ("def rotate_list(lst, k):\n"
         "    if not lst:\n"
         "        return lst\n"
         "    k = k % len(lst)\n"
         "    return lst[-k:] + lst[:-k]"),
    ]
    pools["narrative"] = [
        ("The lighthouse keeper had tended the light on Greyrock Point for "
         "twenty three years, through storms that shook the tower and "
         "winters that sealed the cove in ice. When the automated beacon "
         "was scheduled to replace him the following spring, he began "
         "keeping a journal of every ship he had ever guided home."),
        ("Mara found the key beneath the third floorboard of her grandmother's "
         "attic, wrapped in a cloth that smelled of cedar and old paper. The "
         "key was smaller than any door she knew, and the note attached to it "
         "read only: for the door that was never built."),
        ("On the last night of the expedition, the team gathered around a "
         "campfire at the edge of the glacier and reviewed the season's "
         "findings. Their instruments had recorded something unexpected in "
         "the ice cores, a signal that did not match any known natural "
         "pattern, and they agreed to return the following year to confirm "
         "it."),
        ("The old theater on Meridian Street had been dark for decades when "
         "a group of volunteers decided to restore it. They discovered "
         "original blueprints hidden behind a panel in the projection room, "
         "complete with seating charts from the theater's opening night in "
         "1931."),
        ("Every autumn, the village of Alder Creek held a lantern festival "
         "to honor the travelers who had passed through over the centuries. "
         "Families wrote names on rice paper lanterns and set them adrift on "
         "the river, and the current carried the lights downstream toward "
         "the sea."),
        ("The cartographer spent three years mapping the uncharted valleys "
         "of the northern range, traveling by mule and foot through terrain "
         "that appeared on no existing survey. When she finally published "
         "her maps, they revealed a pass through the mountains that traders "
         "had used for generations without ever drawing it."),
        ("In the quiet hours before dawn, the baker would arrive at the "
         "empty shop and begin the day's first batch of bread. The ritual "
         "had not changed in forty years: same flour, same starter, same "
         "pre dawn quiet, and by the time the first customers came, the "
         "smell of fresh loaves filled the street."),
        ("The two brothers had not spoken since the funeral, but every "
         "harvest season they both returned to the family farm without "
         "comment. They worked the fields in silence, ate dinner at "
         "opposite ends of the long table, and parted at dusk, each "
         "carrying the same unspoken understanding that some things "
         "outlasted the argument that had divided them."),
    ]
    pools["math"] = [
        ("The sum of the interior angles of any triangle is always 180 "
         "degrees. This holds regardless of whether the triangle is "
         "acute, right, or obtuse. For a quadrilateral, the sum is 360 "
         "degrees, and for any polygon with n sides, the formula is "
         "(n minus 2) times 180 degrees."),
        ("A prime number is a natural number greater than one that has no "
         "positive divisors other than one and itself. The first few primes "
         "are 2, 3, 5, 7, 11, 13. There are infinitely many primes, a fact "
         "proven by Euclid over two thousand years ago."),
        ("The Pythagorean theorem states that in a right triangle, the square "
         "of the hypotenuse equals the sum of the squares of the other two "
         "sides: a squared plus b squared equals c squared. It applies only "
         "to right triangles, and its converse is also true: if a squared "
         "plus b squared equals c squared, the triangle is right."),
        ("Compound interest is calculated on both the initial principal and "
         "the accumulated interest from previous periods. The formula is A "
         "equals P times (1 plus r) to the power of t, where P is principal, "
         "r is the rate, and t is time. Even small rate differences produce "
         "large outcomes over long periods."),
        ("The quadratic formula x equals negative b plus or minus the square "
         "root of b squared minus 4ac, all over 2a, solves any equation of "
         "the form ax squared plus bx plus c equals zero. The expression "
         "under the root, the discriminant, determines whether there are "
         "two real solutions, one repeated solution, or two complex ones."),
        ("A matrix is a rectangular array of numbers arranged in rows and "
         "columns. Matrix multiplication combines two matrices by taking the "
         "dot product of each row of the first with each column of the "
         "second. The result has dimensions equal to the rows of the first "
         "matrix and the columns of the second."),
        ("The derivative of a function measures its instantaneous rate of "
         "change at any point. Geometrically, it is the slope of the tangent "
         "line to the curve at that point. The power rule states that the "
         "derivative of x to the power n is n times x to the power n minus "
         "one, for any real number n."),
        ("Probability measures the likelihood of an event occurring, expressed "
         "as a number between 0 and 1. Independent events are those where one "
         "does not affect the probability of the other, and their joint "
         "probability is the product of their individual probabilities. "
         "Conditional probability deals with events given that another event "
         "has already occurred."),
    ]
    pools["history"] = [
        ("The printing press, invented by Johannes Gutenberg around 1440, "
         "revolutionized the spread of information in Europe. Before its "
         "invention, books were copied by hand and were rare and expensive. "
         "The press made books affordable, accelerated literacy, and enabled "
         "the rapid dissemination of new ideas during the Renaissance and "
         "Reformation."),
        ("The Silk Road was a network of trade routes connecting China to the "
         "Mediterranean world for over a thousand years. Despite its name, it "
         "carried far more than silk: spices, precious metals, glassware, "
         "religious texts, and ideas traveled along its length, making it one "
         "of history's most important channels of cultural exchange."),
        ("The Industrial Revolution began in Britain in the late eighteenth "
         "century and transformed manufacturing from hand production to "
         "machine based factories. Steam power, mechanized textiles, and new "
         "transport systems drove unprecedented economic growth, but also "
         "created crowded cities, harsh working conditions, and new social "
         "classes."),
        ("The French Revolution of 1789 overthrew the monarchy and established "
         "the First French Republic. Inspired by Enlightenment ideals of "
         "liberty and equality, revolutionaries stormed the Bastille, abolished "
         "feudal privileges, and declared the rights of man. The revolution "
         "ultimately gave way to Napoleon's rise to power."),
        ("The Roman Empire reached its greatest territorial extent under "
         "Emperor Trajan in the early second century CE, stretching from "
         "Britain to Mesopotamia. Roman engineering produced roads, aqueducts, "
         "and concrete structures that lasted for millennia. The empire's "
         "administrative systems influenced governance for centuries after "
         "its fall."),
        ("The Manhattan Project was the secret American effort during World "
         "War II to develop nuclear weapons. Led by physicist J. Robert "
         "Oppenheimer, it brought together thousands of scientists and "
         "engineers at sites across the country. The first atomic bomb was "
         "detonated in the New Mexico desert in July 1945."),
        ("The Great Depression began with the stock market crash of October "
         "1929 and lasted through most of the 1930s. Unemployment reached "
         "twenty five percent in the United States, banks failed by the "
         "thousands, and drought turned farmland into the Dust Bowl. "
         "Government programs like the New Deal attempted to stabilize the "
         "economy and provide relief."),
        ("The invention of the telephone by Alexander Graham Bell in 1876 "
         "transformed communication by allowing voice to be transmitted "
         "electrically over distances. Within a generation, telephone "
         "networks spanned continents and oceans, shrinking the time between "
         "sending a message and receiving a response from days to seconds."),
    ]
    pools["technical"] = [
        ("TCP guarantees reliable, ordered delivery of data between two "
         "endpoints. It establishes a connection through a three way "
         "handshake, numbers each byte, and retransmits any segments that "
         "are lost or corrupted. Flow control prevents a fast sender from "
         "overwhelming a slow receiver, and congestion control adjusts "
         "transmission rates when the network is busy."),
        ("A hash function maps input data of arbitrary size to a fixed "
         "size output called a digest. Cryptographic hash functions are "
         "designed so that even a tiny change in the input produces a "
         "completely different output, and it is computationally infeasible "
         "to find two inputs that produce the same digest. They are used "
         "for password storage, data integrity checks, and digital "
         "signatures."),
        ("The HTTP protocol is stateless: each request is independent and "
         "the server does not remember previous requests. Cookies and "
         "sessions were invented to add state on top of HTTP, allowing "
         "websites to remember user logins and shopping carts. HTTPS adds "
         "TLS encryption so that data in transit cannot be read or "
         "modified by third parties."),
        ("A database index is a data structure that speeds up retrieval by "
         "maintaining a sorted or hashed copy of selected columns. Without "
         "an index, a query must scan every row in a table, which is slow "
         "for large tables. Indexes speed up reads but slow down writes "
         "because every insert, update, or delete must also update the "
         "index."),
        ("The Model View Controller pattern separates an application into "
         "three components: the Model holds the data and business logic, "
         "the View presents data to the user, and the Controller handles "
         "user input and updates both Model and View. This separation makes "
         "code easier to maintain, test, and reuse across different "
         "interfaces."),
        ("Cache memory sits between the CPU and main memory and stores copies "
         "of frequently used data for fast access. When the CPU needs data, "
         "it checks the cache first. If the data is present, it is called a "
         "cache hit and retrieved quickly; otherwise it is a cache miss and "
         "must be fetched from slower main memory and copied into the cache "
         "for next time."),
        ("A compiler translates source code into machine code through several "
         "stages: lexical analysis breaks text into tokens, parsing builds an "
         "abstract syntax tree, semantic analysis checks types and rules, "
         "optimization improves efficiency, and code generation emits the "
         "final instructions. Each stage catches different classes of "
         "errors."),
        ("The CAP theorem states that a distributed data store can provide "
         "at most two of three guarantees: consistency, availability, and "
         "partition tolerance. In a network partition, the system must choose "
         "between remaining consistent but unavailable, or available but "
         "potentially inconsistent. Most real systems prioritize partition "
         "tolerance and choose between the other two per operation."),
    ]
    pools["conversation"] = [
        ("Could you explain how photosynthesis works in simple terms? Plants "
         "take in sunlight, water, and carbon dioxide, and use the energy "
         "from sunlight to make their own food in the form of sugar, while "
         "releasing oxygen as a waste product."),
        ("I was wondering if you could help me figure out why my sourdough "
         "starter keeps collapsing. It might be that the kitchen is too "
         "warm, or the flour doesn't have enough protein, or it needs more "
         "time to develop before shaping."),
        ("What do you think about the new transit line that's been proposed "
         "for the east side of the city? Some people say it will reduce "
         "commute times significantly, while others worry about the "
         "construction disruption and the cost to taxpayers."),
        ("Have you ever tried making your own pasta from scratch? It's not "
         "as hard as people think, just flour, eggs, and a bit of patience. "
         "The key is letting the dough rest long enough for the gluten to "
         "relax."),
        ("I heard there's going to be a severe storm this weekend. Do you "
         "think we should cancel our hiking trip, or is it better to wait "
         "and see what the forecast says tomorrow?"),
        ("What's the best way to learn a new language as an adult? I've "
         "been trying to study Spanish for a few months but I keep losing "
         "motivation. Some people say immersion is the only way, others "
         "swear by flashcards."),
        ("My neighbor just installed solar panels and says his electric bill "
         "dropped by sixty percent. I'm curious whether the initial "
         "investment is worth it in our climate, or if it takes too long "
         "to pay off."),
        ("Do you have any recommendations for books about space exploration "
         "that aren't too technical? I'd like to read about the Apollo "
         "missions and what it was like for the astronauts, but I'm not "
         "looking for equations."),
    ]
    pools["factual"] = [
        ("The Great Wall of China is not a single continuous wall but a "
         "series of walls and fortifications built over many centuries by "
         "different dynasties. The most famous sections were built during "
         "the Ming Dynasty between the fourteenth and seventeenth centuries. "
         "Contrary to popular belief, the wall is not visible from the Moon "
         "with the naked eye."),
        ("Honey never spoils. Archaeologists have found pots of honey in "
         "ancient Egyptian tombs that are over three thousand years old and "
         "still perfectly edible. Honey's low moisture content, acidic pH, "
         "and natural hydrogen peroxide production make it inhospitable to "
         "bacteria and microorganisms."),
        ("The human body contains approximately 37 trillion cells, of which "
         "about 30 trillion are red blood cells. Red blood cells live for "
         "about 120 days before being recycled by the spleen and bone "
         "marrow. The body produces roughly two million new red blood cells "
         "every second."),
        ("Water covers approximately 71 percent of the Earth's surface, but "
         "only about 2.5 percent of it is fresh water. Of that fresh water, "
         "the vast majority is locked in glaciers and ice caps. Less than "
         "one percent of all water on Earth is accessible in rivers, lakes, "
         "and groundwater."),
        ("The Amazon rainforest produces roughly 20 percent of the world's "
         "oxygen, though this figure is often overstated. More importantly, "
         "the Amazon stores an enormous amount of carbon in its trees and "
         "soil, making it a critical buffer against climate change. It is "
         "home to an estimated ten percent of all known species on Earth."),
        ("Shakespeare wrote approximately 39 plays and 154 sonnets during his "
         "career. His works have been translated into more languages than "
         "any other writer, and phrases he coined, such as 'break the ice' "
         "and 'wild goose chase,' are still used in everyday English today."),
        ("The Eiffel Tower was originally built as the entrance arch for the "
         "1889 World's Fair in Paris. It was intended to be temporary and "
         "demolished after twenty years, but it proved so popular that it "
         "was allowed to stay. It stands 330 meters tall and was the "
         "world's tallest structure until the Chrysler Building was "
         "completed in 1930."),
        ("Octopuses have three hearts: two pump blood to the gills and one "
         "pumps blood to the rest of the body. Their blood is blue because "
         "it uses copper based hemocyanin instead of iron based hemoglobin. "
         "They also have nine brains: a central brain plus a mini brain in "
         "each of their eight arms."),
    ]
    for domain, passages in pools.items():
        assert len(passages) == 8, f"{domain} v3 pool has {len(passages)}, expected 8"
    return pools


if __name__ == "__main__":
    # Generate the three non-sealed partitions.
    import argparse
    ap = argparse.ArgumentParser(description="Generate Gate 1 evaluation partitions")
    ap.add_argument("--practice", type=int, default=2,
                    help="passages per domain for practice partition")
    ap.add_argument("--selection", type=int, default=3,
                    help="passages per domain for selection partition")
    ap.add_argument("--regression", type=int, default=3,
                    help="passages per domain for regression partition")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = AI_RESEARCH / "data" / "gate1_eval_v2"
    data_dir.mkdir(parents=True, exist_ok=True)

    summary = generate_partitions(
        practice_path=data_dir / "practice.json",
        selection_path=data_dir / "selection.json",
        regression_path=data_dir / "regression.json",
        n_per_domain_practice=args.practice,
        n_per_domain_selection=args.selection,
        n_per_domain_regression=args.regression,
        seed=args.seed,
    )
    summary["data_dir"] = str(data_dir)
    summary["note"] = (
        "Three non-sealed partitions generated. The FINAL sealed partition "
        "must be generated separately and stored in gate1_sealed/ outside "
        "the ai-research tree. It is NOT generated by this script."
    )
    print(json.dumps(summary, indent=2))
