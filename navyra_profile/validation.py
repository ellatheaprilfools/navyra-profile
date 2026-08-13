"""
navyra_profile.validation — accuracy validation harness.

PROJECT TASK 5. Runs synthetic traffic with known ground truth through
analyse(), and compares reported tier rates against the known truth,
across a range of template_share values — quantifying the profiler's
per-tier accuracy rather than just asserting it.
"""


from __future__ import annotations

from dataclasses import dataclass, field

from .analyser import analyse
from .synth import make_traffic


@dataclass
class SweepPoint:
    template_share: float
    n: int

    # ground truth, read directly from synth's own tier labels
    t1_true: float
    t2_true: float
    t3_true: float
    t4_true_count: int

    # reported by analyse(), converted to share-of-total for fair comparison
    t1_reported: float
    t2_reported: float
    t3_reported: float
    t4_reported_count: int

    # signed error: reported - true. Positive = analyzer overestimates.
    t1_error: float = field(init=False)
    t2_error: float = field(init=False)
    t3_error: float = field(init=False)

    # how much of T3's ground truth was itself validated against the real
    # embedding model during generation vs. fell back (see synth.py) —
    # included so a low T3 accuracy reading can be traced back to
    # generator behavior, not just presumed to be analyzer error
    t3_validation_pass_rate: float = 0.0

    def __post_init__(self):
        self.t1_error = self.t1_reported - self.t1_true
        self.t2_error = self.t2_reported - self.t2_true
        self.t3_error = self.t3_reported - self.t3_true


@dataclass
class SweepResult:
    points: list[SweepPoint] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'share':>6} {'T1 true':>8} {'T1 rep':>8} {'T1 err':>8}  "
            f"{'T2 true':>8} {'T2 rep':>8} {'T2 err':>8}  "
            f"{'T3 true':>8} {'T3 rep':>8} {'T3 err':>8}",
        ]
        for p in self.points:
            lines.append(
                f"{p.template_share:>6.2f} "
                f"{p.t1_true:>8.1%} {p.t1_reported:>8.1%} {p.t1_error:>+8.1%}  "
                f"{p.t2_true:>8.1%} {p.t2_reported:>8.1%} {p.t2_error:>+8.1%}  "
                f"{p.t3_true:>8.1%} {p.t3_reported:>8.1%} {p.t3_error:>+8.1%}"
            )
        lines.append("")
        lines.append(self._accuracy_summary())
        return "\n".join(lines)

    def _accuracy_summary(self) -> str:
        def mean_abs_error(errors: list[float]) -> float:
            return sum(abs(e) for e in errors) / len(errors) if errors else 0.0

        t1_mae = mean_abs_error([p.t1_error for p in self.points])
        t2_mae = mean_abs_error([p.t2_error for p in self.points])
        t3_mae = mean_abs_error([p.t3_error for p in self.points])
        return (
            f"Mean absolute error across {len(self.points)} sweep points:\n"
            f"  T1: {t1_mae:.1%}\n"
            f"  T2: {t2_mae:.1%}\n"
            f"  T3: {t3_mae:.1%}"
        )


def _tier_share_of_total(tiers: list[str], label: str) -> float:
    if not tiers:
        return 0.0
    return sum(1 for t in tiers if t == label) / len(tiers)


def run_sweep(template_shares: list[float] = (0.1, 0.2, 0.4, 0.6, 0.8),
              n: int = 2000,
              n_templates: int = 25,
              paraphrase_strength: float = 0.3,
              radius: int = 6,
              seed: int = 0) -> SweepResult:
    """Run make_traffic() + analyse() at each template_share in
    template_shares, comparing reported tier rates against known ground
    truth. Returns a SweepResult with per-point detail and overall
    accuracy summary (mean absolute error per tier).
    """
    points = []

    for share in template_shares:
        synth_result = make_traffic(
            n=n, template_share=share, n_templates=n_templates,
            paraphrase_strength=paraphrase_strength, seed=seed)

        report = analyse(synth_result.prompts, radius=radius)

        t1_true = _tier_share_of_total(synth_result.tiers, "T1")
        t2_true = _tier_share_of_total(synth_result.tiers, "T2")
        t3_true = _tier_share_of_total(synth_result.tiers, "T3")
        t4_true_count = sum(1 for t in synth_result.tiers if t == "T4")

        # convert reported rates from "share of non-exact subset" to
        # "share of total traffic" — same conversion as report.py's
        # waterfall chart, needed for a fair comparison against ground
        # truth (which is already expressed as share of total)
        t1_reported = report.exact_repeat_rate
        sem_share = 1.0 - t1_reported
        t2_reported = report.would_hit_rate * sem_share
        t3_reported = report.cross_bucket_rate * sem_share

        t3_val = synth_result.t3_validation
        t3_pass_rate = (
            t3_val.get("passed", 0) / max(1, t3_val.get("passed", 0) + t3_val.get("fallback", 0))
            if t3_val else 0.0
        )

        points.append(SweepPoint(
            template_share=share,
            n=n,
            t1_true=t1_true, t2_true=t2_true, t3_true=t3_true,
            t4_true_count=t4_true_count,
            t1_reported=t1_reported, t2_reported=t2_reported,
            t3_reported=t3_reported,
            t4_reported_count=report.near_dup_count,
            t3_validation_pass_rate=t3_pass_rate,
        ))

    return SweepResult(points=points)