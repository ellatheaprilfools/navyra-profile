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

def make_traffic(n: int, template_share: float, **kw) -> list[str]:
    raise NotImplementedError("Project task 4 — see docstring")
