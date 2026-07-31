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
import difflib
import html as html_module
 
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from .analyser import TrafficReport
from matplotlib.backends.backend_pdf import PdfPages

_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 900px;
       margin: 40px auto; color: #222; line-height: 1.5; }
h1 { font-size: 1.6em; }
h2 { font-size: 1.2em; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
img { max-width: 100%; display: block; margin: 12px 0; }
.meta { color: #666; font-size: 0.9em; }
.risk-box { background: #fff6e5; border: 1px solid #e0b84b; border-radius: 6px;
            padding: 16px; margin-top: 12px; }
.near-dup-list { list-style: none; padding: 0; }
.near-dup-list li { border-top: 1px solid #eee; padding: 8px 0; font-family: monospace;
                     font-size: 0.9em; }
mark { background: #ffcccc; padding: 0 2px; }
pre.summary { background: #f6f6f6; padding: 12px; border-radius: 6px; overflow-x: auto; }
.note { font-size: 0.85em; color: #555; font-style: italic; }
"""
 

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def _build_waterfall_chart(report: TrafficReport) -> str:
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
    return fig 

def _waterfall_chart(report: TrafficReport) -> str:
    fig = _build_waterfall_chart(report)
    return _fig_to_base64(fig)

def _build_hit_rate_by_bucket_chart(report: TrafficReport) -> str:
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
    return fig

def _hit_rate_by_bucket_chart(report: TrafficReport) -> str:
    fig = _build_hit_rate_by_bucket_chart(report)
    if not fig:
        return ""
    return _fig_to_base64(fig)


def _build_cluster_size_chart(report: TrafficReport) -> str:
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
    return fig

def _cluster_size_chart(report: TrafficReport) -> str:
    fig = _build_cluster_size_chart(report)
    if not fig:
        return ""
    return _fig_to_base64(fig)


def _build_warmup_chart(report: TrafficReport) -> str:
    history = getattr(report, "warmup_history", [])
    if not history:
        return ""
    xs, ys = zip(*history)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, [y * 100 for y in ys], color="#2E7DAF", linewidth=2)
    ax.set_xlabel("Prompts processed")
    ax.set_ylabel("Cumulative T2 hit rate (%)")
    ax.set_title("Cache warm-up curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def _warmup_chart(report: TrafficReport) -> str:
    fig = _build_warmup_chart(report)
    if not fig:
        return ""
    return _fig_to_base64(fig)


def _highlight_diff(a: str, b: str) -> tuple[str, str]:
    a_words, b_words = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, a_words, b_words)
    out_a, out_b = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_chunk = html_module.escape(" ".join(a_words[i1:i2]))
        b_chunk = html_module.escape(" ".join(b_words[j1:j2]))
        if tag == "equal":
            out_a.append(a_chunk)
            out_b.append(b_chunk)
        else:
            if a_chunk:
                out_a.append(f"<mark>{a_chunk}</mark>")
            if b_chunk:
                out_b.append(f"<mark>{b_chunk}</mark>")
    return " ".join(out_a), " ".join(out_b)

def _near_dup_box(report: TrafficReport) -> str:
    if not report.near_dup_examples:
        return "<p>No near-duplicate risk pairs found in this sample.</p>"

    rows = []
    for a, b in report.near_dup_examples:
        a_hl, b_hl = _highlight_diff(a, b)
        rows.append(f"<li><div>{a_hl}</div><div>{b_hl}</div></li>")

    return (
        "<p>These prompt pairs are nearly identical in fingerprint but differ "
        "in specific details (highlighted). A whole-answer cache would risk "
        "serving one prompt's answer for the other — this is why answer-replay "
        "caching is unsafe on this traffic:</p>"
        f"<ul class='near-dup-list'>{''.join(rows)}</ul>"
    )

def html_report(report: TrafficReport, path: str) -> None:
    """Render a self-contained HTML report for `report` and write it to `path`."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    waterfall_img = _waterfall_chart(report)
    bucket_img = _hit_rate_by_bucket_chart(report)
    cluster_img = _cluster_size_chart(report)
    warmup_img = _warmup_chart(report)
    near_dup_html = _near_dup_box(report)

    parts = [
        f"<html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>",
        "<h1>Navyra Traffic Profile</h1>",
        f"<p class='meta'>Generated {generated} · {report.n_prompts:,} prompts analysed"
        f" · match radius {report.radius}</p>",

        "<h2>Tiered waterfall</h2>",
        f"<img src='data:image/png;base64,{waterfall_img}'>",
        "<p class='note'>T2 is an upper-bound proxy: up to this share of traffic is "
        "engine-addressable today. It is not a promised saving — the true hit rate "
        "is gated inside the model and measured in the free trial.</p>",
    ]

    if bucket_img:
        parts += ["<h2>Hit rate by length bucket</h2>",
                  f"<img src='data:image/png;base64,{bucket_img}'>"]
    if cluster_img:
        parts += ["<h2>Cluster size distribution</h2>",
                  f"<img src='data:image/png;base64,{cluster_img}'>"]
    if warmup_img:
        parts += ["<h2>Cache warm-up curve</h2>",
                  f"<img src='data:image/png;base64,{warmup_img}'>"]

    parts += [
        "<h2>Near-duplicate risk</h2>",
        f"<div class='risk-box'>{near_dup_html}</div>",

        "<h2>Full summary</h2>",
        f"<pre class='summary'>{html.escape(report.summary())}</pre>",

        "</body></html>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

def pdf_report(report: TrafficReport, path: str) -> None:
    with PdfPages(path) as pdf:
        # page 1: title + summary text
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.95, "Navyra Traffic Profile", ha="center", fontsize=18, weight="bold")
        ax.text(0.05, 0.85, report.summary(), fontsize=9, family="monospace", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # one page per chart
        for builder in (_build_waterfall_fig, _build_hit_rate_by_bucket_fig,
                        _build_cluster_size_fig, _build_warmup_fig):
            fig = builder(report)
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig) 