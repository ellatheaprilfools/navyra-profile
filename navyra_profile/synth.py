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
from dataclasses import dataclass, field
 
from faker import Faker
 
 
@dataclass
class SynthResult:
    prompts: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)  # "T1"/"T2"/"T3"/"T4"/"unique", parallel to prompts
 

 
_FAKER_SLOTS = {
    "order_id": lambda fake: str(fake.random_int(min=1000, max=99999)),
    "city": lambda fake: fake.city(),
    "name": lambda fake: fake.first_name(),
    "date": lambda fake: fake.date(),
    "service": lambda fake: fake.company(),
    "topic": lambda fake: fake.bs(),
}
 
# template, weight to simulate real data 
_BASE_TEMPLATES = [
    ("What's the status of order {order_id}?", 5),
    ("Can you summarize {topic} for me?", 3),
    ("What's the weather like in {city} today?", 2),
    ("Hi {name}, can you confirm the meeting on {date}?", 4),
    ("How do I contact support at {service}?", 1),
]
 
 
def _build_template_bank(n_templates: int) -> list[tuple[str, int]]:
    """Expand _BASE_TEMPLATES up to n_templates entries by cycling through
    them with decaying weight, if more templates are requested than the
    hand-written bank has.
    """
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
    "status": ["state", "progress"],
    "summarize": ["summarise", "give an overview of"],
    "weather": ["forecast", "conditions"],
    "confirm": ["verify", "double-check"],
    "contact": ["reach", "get in touch with"],
}
 
_LENGTHENERS = ["Just to clarify,", "If you don't mind,", "Sorry to ask, but", "I was wondering,", "Before I forget,"]
 
 
def _swap_words(text: str, strength: float, rng: random.Random) -> tuple[str, bool]:
    """Attempt synonym swaps on eligible words. Returns (result, changed) —
    changed is False if nothing in the text was swappable at all.
    """
    words = text.split()
    swappable_indices = [i for i, w in enumerate(words)
                          if w.lower().strip("?.,'") in _SYNONYM_SWAPS]
    if not swappable_indices:
        return text, False
 
    forced = rng.choice(swappable_indices)  # guarantee at least one real swap
    for i in swappable_indices:
        if i == forced or rng.random() < strength:
            key = words[i].lower().strip("?.,'")
            words[i] = rng.choice(_SYNONYM_SWAPS[key])
    return " ".join(words), True
 
 
def _paraphrase(text: str, strength: float, rng: random.Random) -> str:
    """Same-bucket paraphrase (T2): reword via synonym swap. If nothing is
    swappable, fall back to a minimal guaranteed change that shouldn't
    push the token count into a different bucket.
    """
    result, changed = _swap_words(text, strength, rng)
    if changed:
        return result
    if text.rstrip().endswith("?"):
        return text.rstrip()[:-1] + ", please?"
    return text + " please."
 
 
def _paraphrase_cross_bucket(text: str, strength: float, rng: random.Random,
                              bucket_size: int = 8) -> str:
    """Cross-bucket paraphrase (T3): reword AND guarantee the token count
    crosses into a different length bucket than the original.
    """
    reworded, _ = _swap_words(text, strength, rng)
    current_len = len(reworded.split())
    current_bucket = ((current_len + bucket_size - 1) // bucket_size) * bucket_size
    words_needed = (current_bucket - current_len) + 1
 
    filler = " ".join(rng.choices(_LENGTHENERS, k=max(1, words_needed // 2)))
    return filler + " " + reworded
 
 
def _near_duplicate(template: str, base_prompt: str, fake: Faker) -> str:
    """Re-fill the template's slot(s) with fresh values. Used for T4 — this
    is what makes it dangerous for whole-answer caching: nearly identical
    text, one or more critical details changed.
 
    LIMITATION: for templates with more than one placeholder, this refills
    ALL of them, not just one — so "differs by exactly one detail" only
    strictly holds for the current single/low-slot templates. Worth fixing
    to change only one placeholder if multi-slot templates get added later.
    """
    placeholders = re.findall(r"\{(\w+)\}", template)
    if not placeholders:
        return base_prompt
    values = {p: _FAKER_SLOTS[p](fake) for p in placeholders}
    return template.format(**values)
 
 

 
def _unique_prompt(fake: Faker) -> str:
    return fake.sentence(nb_words=8)
 

 
_TIER_MIX = {
    "new": 0.40,    # a fresh, never-seen-before filled template
    "T1": 0.15,     # exact repeat of an earlier templated prompt
    "T2": 0.20,     # same-bucket paraphrase of an earlier templated prompt
    "T3": 0.15,     # cross-bucket paraphrase of an earlier templated prompt
    "T4": 0.10,     # near-duplicate (one slot changed) of an earlier templated prompt
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
 
    prompts: list[str] = []
    tiers: list[str] = []
 
    # bases: list of (template, filled_prompt) already emitted, available
    # for T1/T2/T3/T4 to be built from
    bases: list[tuple[str, str]] = []
 
    n_templated = round(n * template_share)
    n_unique = n - n_templated
 
    roles = list(_TIER_MIX)
    role_weights = list(_TIER_MIX.values())
 
    for _ in range(n_templated):
        role = rng.choices(roles, weights=role_weights, k=1)[0]
 
        if role == "new" or not bases:
            template = rng.choices(bank_templates, weights=bank_weights, k=1)[0]
            filled = _fill_template(template, fake)
            bases.append((template, filled))
            prompts.append(filled)
            tiers.append("unique")  # first occurrence: nothing to repeat yet
            continue
 
        template, base_prompt = rng.choice(bases)
 
        if role == "T1":
            prompts.append(base_prompt)
        elif role == "T2":
            prompts.append(_paraphrase(base_prompt, paraphrase_strength, rng))
        elif role == "T3":
            prompts.append(_paraphrase_cross_bucket(base_prompt, paraphrase_strength, rng))
        elif role == "T4":
            prompts.append(_near_duplicate(template, base_prompt, fake))
        tiers.append(role)
 
    for _ in range(n_unique):
        prompts.append(_unique_prompt(fake))
        tiers.append("unique")
 
    # shuffle together so templated/unique traffic is interleaved, like a
    # real stream, rather than all templated prompts arriving first
    combined = list(zip(prompts, tiers))
    rng.shuffle(combined)
    prompts, tiers = zip(*combined) if combined else ([], [])
 
    return SynthResult(prompts=list(prompts), tiers=list(tiers))