"""
navyra_profile.synth — synthetic traffic generation.  [TO BUILD]

PROJECT TASK 4. Generate realistic synthetic LLM traffic with a KNOWN
ground-truth templatedness, so the profiler's estimates can be validated:

  make_traffic(n=10_000, template_share=0.4, n_templates=25,
               paraphrase_strength=0.3, seed=0) -> list[str]

Approaches to explore (pick and justify):
- template banks with slot-filling (names, dates, amounts from faker)
- paraphrase augmentation (rule-based swaps; optionally a small local
  paraphrase model)
- mixing in genuinely diverse text (public domain corpora) as the
  non-templated share

Generators must produce KNOWN quantities of each tier: exact repeats
(tier 1), same-bucket paraphrases (tier 2), cross-bucket paraphrases
(tier 3), and near-duplicates differing only in a digit/date/name
(tier 4). The validation harness (task 5) sweeps these shares and checks
each reported tier tracks its ground truth. That closes the loop: we can
then state the profiler's per-tier accuracy quantitatively.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from faker import Faker

from .analyser import _bucket
from .fingerprint import Fingerprinter, hamming


_VALIDATION_RADIUS = 6


@dataclass
class SynthResult:
    prompts: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)
    base_of: list[int | None] = field(default_factory=list)
    t3_validation: dict = field(default_factory=dict)  # {"passed": N, "fallback": N}


_FAKER_SLOTS = {
    "order_id": lambda fake: str(fake.random_int(min=1000, max=99999)),
    "city": lambda fake: fake.city(),
    "name": lambda fake: fake.first_name(),
    "date": lambda fake: fake.date(),
    "service": lambda fake: fake.company(),
    "topic": lambda fake: fake.bs(),
}

_BASE_TEMPLATES = [
    ("What's the status of order {order_id}?", 5),
    ("Can you summarize {topic} for me?", 3),
    ("What's the weather like in {city} today?", 2),
    ("Hi {name}, can you confirm the meeting on {date}?", 4),
    ("How do I contact support at {service}?", 1),
]


def _build_template_bank(n_templates: int) -> list[tuple[str, int]]:
    bank = list(_BASE_TEMPLATES)
    i = 0
    while len(bank) < n_templates:
        base_template, base_weight = _BASE_TEMPLATES[i % len(_BASE_TEMPLATES)]
        bank.append((f"Quick one — {base_template[0].lower()}{base_template[1:]}",
                      max(1, base_weight // 2)))
        i += 1
    return bank[:n_templates]


def _fill_template(template: str, fake: Faker) -> str:
    placeholders = re.findall(r"\{(\w+)\}", template)
    values = {p: _FAKER_SLOTS[p](fake) for p in placeholders}
    return template.format(**values)


_SYNONYM_SWAPS = {
    "status": ["state", "progress", "current state", "standing", "condition"],
    "summarize": ["summarise", "give an overview of", "recap", "outline", "break down"],
    "weather": ["forecast", "conditions", "climate", "outlook"],
    "confirm": ["verify", "double-check", "validate", "check", "make sure of"],
    "contact": ["reach", "get in touch with", "connect with", "speak to", "reach out to"],
}

_LENGTHENERS = ["Just to clarify,", "If you don't mind,", "Quickly —",
                "Sorry to ask, but", "I was wondering,", "Before I forget,"]


def _swap_words(text: str, strength: float, rng: random.Random) -> tuple[str, bool]:
    words = text.split()
    swappable_indices = [i for i, w in enumerate(words)
                          if w.lower().strip("?.,'") in _SYNONYM_SWAPS]
    if not swappable_indices:
        return text, False
    forced = rng.choice(swappable_indices)
    for i in swappable_indices:
        if i == forced or rng.random() < strength:
            key = words[i].lower().strip("?.,'")
            words[i] = rng.choice(_SYNONYM_SWAPS[key])
    return " ".join(words), True


def _paraphrase(text: str, strength: float, rng: random.Random) -> str:
    result, changed = _swap_words(text, strength, rng)
    if changed:
        return result
    if text.rstrip().endswith("?"):
        return text.rstrip()[:-1] + ", please?"
    return text + " please."


def _paraphrase_cross_bucket(text: str, strength: float, rng: random.Random,
                              bucket_size: int = 8) -> str:
    reworded, _ = _swap_words(text, strength, rng)
    current_len = len(reworded.split())
    current_bucket = ((current_len + bucket_size - 1) // bucket_size) * bucket_size
    words_needed = (current_bucket - current_len) + 1
    filler = " ".join(rng.choices(_LENGTHENERS, k=max(1, words_needed // 2)))
    return filler + " " + reworded


_REMOVABLE_SCAFFOLDING = [
    "Quick one —", "Quick one -", "Just to clarify,", "If you don't mind,",
    "Sorry to ask, but", "I was wondering,", "Before I forget,",
    "can you", "could you", "for me", "please", "today",
]


def _paraphrase_cross_bucket_shorter(text: str, strength: float,
                                      rng: random.Random,
                                      bucket_size: int = 8) -> str | None:
    """Cross-bucket paraphrase by REMOVING low-content scaffolding rather
    than adding filler.
    """
    reworded, _ = _swap_words(text, strength, rng)
    original_bucket = _bucket(len(reworded.split()), bucket_size)

    candidates = _REMOVABLE_SCAFFOLDING.copy()
    rng.shuffle(candidates)

    stripped = reworded
    for phrase in candidates:
        if phrase.lower() in stripped.lower():
            # remove the first occurrence, case-insensitively
            idx = stripped.lower().index(phrase.lower())
            stripped = (stripped[:idx] + stripped[idx + len(phrase):])
            stripped = " ".join(stripped.split())  # tidy whitespace
            if _bucket(len(stripped.split()), bucket_size) != original_bucket:
                # capitalise properly if we removed a leading phrase
                if stripped and stripped[0].islower():
                    stripped = stripped[0].upper() + stripped[1:]
                return stripped

    return None  # couldn't cross a boundary by shortening alone


def _make_validated_t3(base_prompt: str, base_fp, strength: float,
                        rng: random.Random, fp: Fingerprinter,
                        used: set, strategy_stats: dict,
                        bucket_size: int = 8, max_attempts: int = 8):
    """Generate a T3 candidate and VALIDATE it directly against the real
    Fingerprinter — the same mechanism analyse() itself uses — rather
    than assuming any particular construction method preserves meaning.
    Retries with fresh randomness until a candidate both crosses into a
    different length bucket AND stays within the analyzer's own match
    radius of its base. Falls back to the closest attempt if none pass
    within max_attempts (rare, but text-generation constraints mean a
    perfect candidate isn't always guaranteed on the first few tries).
 
    Returns (text, passed_validation: bool).
    """
    base_bucket = _bucket(len(base_prompt.split()), bucket_size)
    best_candidate = None
    best_distance = None
    strategy_used = None
 
    def _try(candidate, strategy_name):
        nonlocal best_candidate, best_distance
        if candidate is None or candidate in used:
            return None
        if _bucket(len(candidate.split()), bucket_size) == base_bucket:
            return None  # didn't actually cross buckets
        cand_fp = fp.fingerprints([candidate])[0]
        d = int(hamming(np.array([cand_fp]), np.uint64(base_fp))[0])
        strategy_stats[strategy_name]["attempted"] += 1
        if best_distance is None or d < best_distance:
            best_candidate, best_distance = candidate, d
        if d <= _VALIDATION_RADIUS:
            strategy_stats[strategy_name]["valid"] += 1
            return candidate
        return None
 
    for _ in range(max_attempts):
        short_candidate = _paraphrase_cross_bucket_shorter(
            base_prompt, strength, rng, bucket_size)
        long_candidate = _paraphrase_cross_bucket(
            base_prompt, strength, rng, bucket_size)
 
        long_pass = _try(long_candidate, "longer")
        if long_pass:
            return long_pass, True, "longer"
 
        short_pass = _try(short_candidate, "shorter")
        if short_pass:
            return short_pass, True, "shorter"
 
    return (best_candidate or base_prompt), False, strategy_used


def _near_duplicate(template: str, base_prompt: str, fake: Faker) -> str:
    placeholders = re.findall(r"\{(\w+)\}", template)
    if not placeholders:
        return base_prompt
    values = {p: _FAKER_SLOTS[p](fake) for p in placeholders}
    return template.format(**values)


def _unique_prompt(fake: Faker) -> str:
    return fake.sentence(nb_words=8)


_TIER_MIX = {
    "new": 0.40,
    "T1": 0.15,
    "T2": 0.20,
    "T3": 0.15,
    "T4": 0.10,
}


def make_traffic(n: int = 10_000,
                  template_share: float = 0.4,
                  n_templates: int = 25,
                  paraphrase_strength: float = 0.3,
                  seed: int = 0) -> SynthResult:
    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)
 
    bank = _build_template_bank(n_templates)
    bank_templates = [t for t, w in bank]
    bank_weights = [w for t, w in bank]
 
    n_templated = round(n * template_share)
    n_unique = n - n_templated
 
    items = []
    bases = []  # (template, filled_text, item_index, fingerprint)
    used_variants = defaultdict(set)
    t3_validation_stats = {"passed": 0, "fallback": 0}
    strategy_stats = {
        "shorter": {"attempted": 0, "valid": 0},
        "longer": {"attempted": 0, "valid": 0},
    }
 
    fp = Fingerprinter()  # created once — reused for every base and T3 candidate
 
    roles = list(_TIER_MIX)
    role_weights = list(_TIER_MIX.values())
 
    for _ in range(n_templated):
        role = rng.choices(roles, weights=role_weights, k=1)[0]
 
        if role == "new" or not bases:
            template = rng.choices(bank_templates, weights=bank_weights, k=1)[0]
            filled = _fill_template(template, fake)
            idx = len(items)
            items.append({"text": filled, "tier": "unique", "depends_on": None})
            base_fp = fp.fingerprints([filled])[0]
            bases.append((template, filled, idx, base_fp))
            continue
 
        template, base_prompt, base_idx, base_fp = rng.choice(bases)
        if role == "T1":
            text = base_prompt
        elif role == "T2":
            text = _paraphrase(base_prompt, paraphrase_strength, rng)
            attempts = 0
            while text in used_variants[base_idx] and attempts < 5:
                text = _paraphrase(base_prompt, paraphrase_strength, rng)
                attempts += 1
            used_variants[base_idx].add(text)
        elif role == "T3":
            text, passed, strategy = _make_validated_t3(
                base_prompt, base_fp, paraphrase_strength, rng, fp,
                used_variants[base_idx], strategy_stats)
            used_variants[base_idx].add(text)
            if passed:
                t3_validation_stats["passed"] += 1
                t3_validation_stats[f"passed_via_{strategy}"] = (
                    t3_validation_stats.get(f"passed_via_{strategy}", 0) + 1)
            else:
                # didn't pass real semantic validation — don't claim T3;
                t3_validation_stats["fallback"] += 1
                role = "unique"  
        elif role == "T4":
            text = _near_duplicate(template, base_prompt, fake)
        items.append({"text": text, "tier": role, "depends_on": base_idx})
 
    for _ in range(n_unique):
        items.append({"text": _unique_prompt(fake), "tier": "unique", "depends_on": None})
 
    children = defaultdict(list)
    indegree = [0] * len(items)
    for i, it in enumerate(items):
        if it["depends_on"] is not None:
            children[it["depends_on"]].append(i)
            indegree[i] = 1
 
    ready = [i for i in range(len(items)) if indegree[i] == 0]
    rng.shuffle(ready)
    order = []
    while ready:
        i = ready.pop(rng.randrange(len(ready)))
        order.append(i)
        for c in children[i]:
            indegree[c] -= 1
            if indegree[c] == 0:
                ready.insert(rng.randrange(len(ready) + 1), c)
 
    prompts = [items[i]["text"] for i in order]
    tiers = [items[i]["tier"] for i in order]
 
    position_of_original_index = {orig_i: pos for pos, orig_i in enumerate(order)}
    base_of = [
        position_of_original_index[items[i]["depends_on"]]
        if items[i]["depends_on"] is not None else None
        for i in order
    ]
 
    t3_validation_stats["by_strategy"] = strategy_stats
    return SynthResult(prompts=prompts, tiers=tiers, base_of=base_of,
                        t3_validation=t3_validation_stats)