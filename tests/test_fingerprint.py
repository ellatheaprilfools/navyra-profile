"""Starter tests — extend substantially (project task 5 includes tests)."""
import numpy as np
from navyra_profile.fingerprint import Fingerprinter, HashingBackend, hamming
from navyra_profile.analyser import analyse


def test_identical_texts_identical_fingerprints():
    fp = Fingerprinter(backend=HashingBackend())
    a = fp.fingerprints(["cancel my subscription please"] * 2)
    assert a[0] == a[1]


def test_similar_closer_than_different():
    fp = Fingerprinter(backend=HashingBackend())
    fps = fp.fingerprints([
        "please cancel my subscription and refund this month",
        "cancel my subscription and refund me for this month",
        "the mitochondrion is the powerhouse of the cell",
    ])
    d_sim = hamming(np.uint64(fps[0]), np.uint64(fps[1]))
    d_diff = hamming(np.uint64(fps[0]), np.uint64(fps[2]))
    assert d_sim < d_diff


DIVERSE = [
    "The mitochondrion is the powerhouse of the cell.",
    "Napoleon was defeated at Waterloo in 1815.",
    "Quarterly revenue grew fourteen percent on renewals.",
    "def quicksort(arr): return sorted(arr)",
    "Het weer in Amsterdam is vandaag bewolkt.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Treaty of Westphalia ended the Thirty Years War.",
    "Interest rates were held steady by the central bank.",
    "A haiku has seventeen syllables in three lines.",
    "Tectonic plates drift a few centimetres per year.",
]


def test_analyse_tiers_repetition():
    # exact repeats land in tier 1, not tier 2
    prompts = ["reset my password please"] * 50 + DIVERSE * 5
    rep = analyse(prompts, fingerprinter=Fingerprinter(backend=HashingBackend()))
    assert rep.exact_repeat_rate > 0.4          # tier 1 catches repeats
    assert rep.n_prompts == 100

    # non-exact, high-overlap paraphrases land in tier 2.
    # NOTE: radius 14 here because the HashingBackend is deliberately weak;
    # the real embedding backend separates these at the default radius 6.
    para = (["please reset my password",
             "please reset my password today",
             "can you please reset my password"] * 10) + DIVERSE * 2
    rep2 = analyse(para, radius=14,
                   fingerprinter=Fingerprinter(backend=HashingBackend()))
    assert rep2.would_hit_rate >= 0.10          # tier 2 semantic hits
    assert rep2.exact_repeat_rate > 0.3         # the x10 repeats are exact

    # genuinely diverse traffic stays at zero even at the wide radius
    rep3 = analyse(DIVERSE, radius=14,
                   fingerprinter=Fingerprinter(backend=HashingBackend()))
    assert rep3.would_hit_rate < 0.05
