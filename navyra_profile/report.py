"""
navyra_profile.report — human-readable report generation.  [TO BUILD]

PROJECT TASK 3. Turn a TrafficReport into:
  1. a one-page HTML report built around the TIERED WATERFALL (see
     analyzer.py header): a waterfall/stacked chart showing
       T1 exact (already free) -> T2 same-bucket semantic (addressable
       today) -> T3 cross-bucket (future) -> remainder (unique traffic),
     plus hit-rate-by-bucket chart, cluster size distribution, cache
     warm-up curve over time, and a highlighted "near-duplicate risk"
     box showing T4 example pairs with the differing tokens marked —
     the finding that shows why answer-replay caching is unsafe on this
     traffic. Self-contained single HTML file (matplotlib or plotly).
  2. a terminal summary (rich or plain text)
  3. optionally a PDF export

Wording rule for the report: T2 is an UPPER-BOUND PROXY ("up to X% of
your traffic is engine-addressable; the free trial measures the true
rate inside the model"). Never present T2 as a promised saving.

Design goals: a GPU owner should be able to run one command against a
JSONL of their prompts and get a report they can forward to their boss.
Look at TrafficReport.summary() for the data available; extend the
analyzer if the charts need more (e.g. rolling hit rate needs per-index
hit/miss history — add it).
"""

def html_report(report, path: str) -> None:
    raise NotImplementedError("Project task 3 — see docstring")
