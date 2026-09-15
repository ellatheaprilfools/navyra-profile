"""Tests for navyra_profile.synth — Task 5's "test suite grown"."""

import pytest

from navyra_profile.synth import make_traffic


def test_reproducible_with_same_seed():
    r1 = make_traffic(n=300, template_share=0.4, seed=0)
    r2 = make_traffic(n=300, template_share=0.4, seed=0)
    assert r1.prompts == r2.prompts
    assert r1.tiers == r2.tiers


def test_different_seeds_produce_different_output():
    r1 = make_traffic(n=300, template_share=0.4, seed=0)
    r2 = make_traffic(n=300, template_share=0.4, seed=1)
    assert r1.prompts != r2.prompts


def test_returns_correct_total_count():
    r = make_traffic(n=500, template_share=0.4, seed=0)
    assert len(r.prompts) == 500
    assert len(r.tiers) == 500
    assert len(r.base_of) == 500


def test_all_tiers_present_at_reasonable_scale():
    # at n=2000 with default paraphrase_strength, every tier should
    # appear at least once — regression guard for the ordering bug found
    # during task 4, which silently made some tiers vanish/inflate
    r = make_traffic(n=2000, template_share=0.4, seed=0)
    seen_tiers = set(r.tiers)
    assert "T1" in seen_tiers
    assert "T2" in seen_tiers
    assert "T4" in seen_tiers
    assert "unique" in seen_tiers
    # T3 may occasionally be empty if every candidate happened to fail
    # validation at this seed — not asserted as strictly present


def test_no_order_corruption():
    # Regression test for the ordering bug found during task 4: derived
    # copies (T1/T2/T3/T4) must never appear before their own base in
    # the output stream, since the analyzer's exact-match detection is
    # order-dependent and would otherwise mislabel which one is "the
    # original".
    r = make_traffic(n=1000, template_share=0.4, seed=0)
    for i, base_idx in enumerate(r.base_of):
        if base_idx is not None:
            assert base_idx < i, (
                f"prompt at position {i} depends on base at position "
                f"{base_idx}, which comes AFTER it — ordering is broken"
            )


def test_base_of_indices_are_valid():
    r = make_traffic(n=500, template_share=0.4, seed=0)
    for base_idx in r.base_of:
        if base_idx is not None:
            assert 0 <= base_idx < len(r.prompts)


def test_t1_is_exact_text_match_of_its_base():
    r = make_traffic(n=1000, template_share=0.4, seed=0)
    for i, (t, base_idx) in enumerate(zip(r.tiers, r.base_of)):
        if t == "T1":
            assert r.prompts[i] == r.prompts[base_idx]


def test_t2_never_exactly_matches_its_own_base():
    # Regression test: an earlier bug let paraphrasing silently produce
    # zero change, which the analyzer would then correctly (but
    # misleadingly, relative to our own T2 label) flag as an exact
    # repeat instead of a semantic match.
    r = make_traffic(n=1000, template_share=0.4, seed=0)
    for i, (t, base_idx) in enumerate(zip(r.tiers, r.base_of)):
        if t == "T2":
            assert r.prompts[i] != r.prompts[base_idx]


def test_t4_differs_from_its_base():
    r = make_traffic(n=1000, template_share=0.4, seed=0)
    for i, (t, base_idx) in enumerate(zip(r.tiers, r.base_of)):
        if t == "T4":
            assert r.prompts[i] != r.prompts[base_idx]


def test_template_share_roughly_controls_templated_fraction():
    # loose sanity check, not a tight statistical assertion — a low
    # template_share should produce a much higher "unique" fraction than
    # a high one
    r_low = make_traffic(n=1000, template_share=0.1, seed=0)
    r_high = make_traffic(n=1000, template_share=0.8, seed=0)

    low_unique_frac = sum(1 for t in r_low.tiers if t == "unique") / len(r_low.tiers)
    high_unique_frac = sum(1 for t in r_high.tiers if t == "unique") / len(r_high.tiers)

    assert low_unique_frac > high_unique_frac


def test_zero_prompts_requested():
    r = make_traffic(n=0, template_share=0.4, seed=0)
    assert r.prompts == []
    assert r.tiers == []


@pytest.mark.parametrize("share", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_handles_full_range_of_template_share(share):
    # edge cases: 0.0 (no templated traffic at all) and 1.0 (entirely
    # templated) shouldn't crash
    r = make_traffic(n=200, template_share=share, seed=0)
    assert len(r.prompts) == 200