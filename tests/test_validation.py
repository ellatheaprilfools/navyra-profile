"""Tests for navyra_profile.validation — Task 5's sweep harness."""

from navyra_profile.validation import run_sweep, SweepResult, SweepPoint


def test_sweep_returns_one_point_per_share():
    result = run_sweep(template_shares=[0.2, 0.5], n=300, seed=0)
    assert len(result.points) == 2
    assert result.points[0].template_share == 0.2
    assert result.points[1].template_share == 0.5


def test_sweep_errors_are_signed_correctly():
    result = run_sweep(template_shares=[0.4], n=300, seed=0)
    p = result.points[0]
    assert p.t1_error == p.t1_reported - p.t1_true
    assert p.t2_error == p.t2_reported - p.t2_true
    assert p.t3_error == p.t3_reported - p.t3_true


def test_sweep_rates_are_valid_fractions():
    result = run_sweep(template_shares=[0.3], n=300, seed=0)
    p = result.points[0]
    for rate in (p.t1_true, p.t2_true, p.t3_true,
                 p.t1_reported, p.t2_reported, p.t3_reported):
        assert 0.0 <= rate <= 1.0


def test_sweep_summary_produces_readable_output():
    result = run_sweep(template_shares=[0.2, 0.4], n=300, seed=0)
    text = result.summary()
    assert "T1" in text
    assert "T2" in text
    assert "T3" in text
    assert "Mean absolute error" in text


def test_empty_sweep_does_not_crash():
    result = run_sweep(template_shares=[], n=300, seed=0)
    assert result.points == []
    # summary() on an empty sweep shouldn't raise a ZeroDivisionError
    text = result.summary()
    assert "0 sweep points" in text


def test_higher_template_share_increases_true_repetition():
    # a loose sanity check: more templated traffic should mean more
    # deliberate T1+T2+T3 repetition in the ground truth, not less
    result = run_sweep(template_shares=[0.1, 0.8], n=1000, seed=0)
    low, high = result.points
    low_repetition = low.t1_true + low.t2_true + low.t3_true
    high_repetition = high.t1_true + high.t2_true + high.t3_true
    assert high_repetition > low_repetition