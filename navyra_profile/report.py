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

from __future__ import annotations
 
import base64
import difflib
import html
import io
from datetime import datetime, timezone
 
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from .analyser import TrafficReport
 

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def _waterfall_chart(report: TrafficReport) -> str:
    """Return a base64-encoded PNG of the waterfall chart."""
    t1 = report.exact_repeat_rate
    sem_share = 1.0 - t1 
    t2 = report.would_hit_rate * sem_share
    t3 = report.cross_bucket_rate * sem_share
    remainder = max(0.0, 1.0 - t1 - t2 - t3)

    
    tier_labels = [
    "T1 exact\nrepeats",
    "T2 same-bucket\nsemantic",
    "T3 cross-bucket\nsemantic",
    "Remainder\n(unique)",
    ]

    values = [t1,t2,t3,remainder]
    colors = ["#4C9A2A", "#2E7DAF", "#D9A441", "#999999"]

    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    running = 0.0

    for i, (label, val) in enumerate(zip(tier_labels, values)):
        ax.bar(label, val, bottom=running, color=colors[i], width=1.0)
        ax.text(i, running + val / 2, f"{val:.1%}", ha="center", va="center",
                 color="white", fontweight="bold")
        running += val

    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of total traffic")
    ax.set_title("Tiered waterfall: where your traffic actually goes")
    fig.tight_layout()
    return _fig_to_base64(fig)

def _hit_rate_by_bucket_chart(report: TrafficReport) -> str:
    if not report.hit_rate_by_bucket:
        return ""

    buckets = sorted(report.hit_rate_by_bucket)
    rates = [report.hit_rate_by_bucket[b][0] for b in buckets]
    counts = [report.hit_rate_by_bucket[b][1] for b in buckets]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([str(b) for b in buckets], rates, color="#2E7DAF")

    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(rates + [0.1]) * 1.25)
    ax.set_xlabel("Prompt length bucket (tokens)")
    ax.set_ylabel("Same-bucket hit rate")
    ax.set_title("Hit rate by length bucket")
    fig.tight_layout()
    return _fig_to_base64(fig)

def _cluster_size_chart(report: TrafficReport) -> str:
    if not report.cluster_sizes:
        return ""
    top = report.cluster_sizes[:20]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(top) + 1), top, color="#6C5B7B")
    ax.set_xlabel("Cluster rank (largest first)")
    ax.set_ylabel("Prompts in cluster")
    ax.set_title(
        f"Cluster size distribution "
        f"({report.n_clusters:,} clusters total, "
        f"top cluster = {report.top_cluster_share:.1%} of traffic)"
    )
    fig.tight_layout()
    return _fig_to_base64(fig)


def html_report(report, path: str) -> None:
    raise NotImplementedError("Project task 3 — see docstring")
