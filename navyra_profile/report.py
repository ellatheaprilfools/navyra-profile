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
    t1 = report.exact_repeat_rate
    sem_share = 1.0 - t1
    t2 = report.would_hit_rate * sem_share
    t3 = report.cross_bucket_rate * sem_share
    remainder = max(0.0, 1.0 - t1 - t2 - t3)

    tier_labels = ["T1 exact\nrepeats", "T2 same-bucket\nsemantic",
                   "T3 cross-bucket\nsemantic", "Remainder\n(unique)"]
    values = [t1, t2, t3, remainder]
    counts = [round(v * report.n_prompts) for v in values]
    colors = ["#4C9A2A", "#2E7DAF", "#D9A441", "#999999"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    running = 0.0
    MIN_VISIBLE = 0.015  # bars below this fraction get a minimum drawn height

    for i, (label, val, n) in enumerate(zip(tier_labels, values, counts)):
        draw_height = max(val, MIN_VISIBLE) if val > 0 else 0
        ax.bar(label, draw_height, bottom=running, color=colors[i], width=1.0)

        label_y = running + draw_height / 2
        text = f"{val:.1%}\n(n={n:,})" if val > 0 else "0.0%"
        text_color = "white" if draw_height > 0.03 else colors[i]
        va = "center" if draw_height > 0.03 else "bottom"
        label_y = label_y if draw_height > 0.03 else running + draw_height + 0.01

        ax.text(i, label_y, text, ha="center", va=va,
                 color=text_color, fontweight="bold", fontsize=9)
        running += val  # advance by the REAL value, not the padded draw_height

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of total traffic")
    ax.set_title("Tiered waterfall: where your traffic actually goes", fontsize=13)
    ax.text(0.5, 1.10, f"Based on {report.n_prompts:,} prompts · match radius {report.radius}",
             transform=ax.transAxes, ha="center", fontsize=9, color="#666")

    legend_text = (
        "T1 — identical prompts, already served free by prefix caching.\n"
        "T2 — different wording, same meaning, same length range: today's addressable "
        "opportunity (upper-bound estimate, not a guaranteed saving).\n"
        "T3 — same meaning, different length range: not addressable yet.\n"
        "Remainder — traffic with no detected repetition."
    )
    fig.text(0.02, -0.02, legend_text, fontsize=8, color="#444", va="top", wrap=True)

    fig.tight_layout()
    return fig


def _waterfall_chart(report: TrafficReport) -> str:
    fig = _build_waterfall_chart(report)
    if fig is None:
        return ""
    return _fig_to_base64(fig)

def _build_hit_rate_by_bucket_chart(report: TrafficReport) -> str:
    if not report.hit_rate_by_bucket:
        return None

    buckets = sorted(report.hit_rate_by_bucket)
    rates = [report.hit_rate_by_bucket[b][0] for b in buckets]
    counts = [report.hit_rate_by_bucket[b][1] for b in buckets]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([str(b) for b in buckets], rates, color="#2E7DAF")

    for bar, rate, n in zip(bars, rates, counts):
        label = f"{rate:.1%}\n(n={n:,})"
        # low-sample-size warning: a rate from very few prompts is noisy
        if n < 30:
            label += "\n⚠ small sample"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 label, ha="center", va="bottom", fontsize=8)

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, max(rates + [0.1]) * 1.35)
    ax.set_xlabel("Prompt length bucket (tokens)")
    ax.set_ylabel("Same-bucket hit rate")
    ax.set_title("Hit rate by length bucket", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    caption = (
        "Each bar shows what share of prompts in that length range found a "
        "semantic match (T2) elsewhere in the same range. 'n' is the number of "
        "prompts that length range actually contains — small n means the rate "
        "is based on limited data and may not generalise."
    )
    fig.text(0.02, -0.05, caption, fontsize=8, color="#444", va="top", wrap=True)
    fig.tight_layout()
    return fig

def _hit_rate_by_bucket_chart(report: TrafficReport) -> str:
    fig = _build_hit_rate_by_bucket_chart(report)
    if not fig:
        return ""
    return _fig_to_base64(_build_hit_rate_by_bucket_chart(report))


def _build_cluster_size_chart(report: TrafficReport) -> str:
    if not report.cluster_sizes:
        return None
    top = report.cluster_sizes[:20]
    shown_total = sum(top)
    all_total = sum(report.cluster_sizes)
    shown_share = shown_total / max(1, all_total)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(1, len(top) + 1), top, color="#6C5B7B")
    ax.bar(1, top[0], color="#4A3F57")  # top cluster gets a distinct shade

    ax.text(1, top[0], f"largest:\n{top[0]:,} prompts", ha="center", va="bottom",
             fontsize=8, fontweight="bold")

    ax.set_xlabel("Cluster rank (largest first)")
    ax.set_ylabel("Prompts in cluster")
    ax.set_title(
        f"Cluster size distribution — top 20 of {report.n_clusters:,} clusters total",
        fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    caption = (
        f"A cluster is a group of prompts that all matched each other semantically. "
        f"The 20 largest clusters shown here account for {shown_share:.1%} of all "
        f"traffic that formed any cluster ({shown_total:,} of {all_total:,} prompts). "
        f"The single largest cluster alone is {report.top_cluster_share:.1%} of your "
        f"total traffic."
    )
    fig.text(0.02, -0.06, caption, fontsize=8, color="#444", va="top", wrap=True)
    fig.tight_layout()
    return fig

def _cluster_size_chart(report: TrafficReport) -> str:
    fig = _build_cluster_size_chart(report)
    if fig is None:
        return ""
    return _fig_to_base64(fig)


def _build_warmup_chart(report: TrafficReport) -> str:
    history = getattr(report, "warmup_history", [])
    if not history:
        return None
    xs, ys = zip(*history)
    final_rate = report.would_hit_rate

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, [y * 100 for y in ys], color="#2E7DAF", linewidth=2, label="observed rate")
    ax.axhline(final_rate * 100, color="#999", linestyle="--", linewidth=1,
                label=f"final rate ({final_rate:.1%})")

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Prompts processed (in original order)")
    ax.set_ylabel("Cumulative T2 hit rate")
    ax.set_title("Cache warm-up curve", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    caption = (
        "This shows how the semantic hit rate changes as traffic streams in, "
        "simulating a cache starting empty. The rate is low early on (nothing to "
        "match against yet) and rises as more prompts accumulate in the bank. "
        "The dashed line marks the final overall rate for reference."
    )
    fig.text(0.02, -0.05, caption, fontsize=8, color="#444", va="top", wrap=True)
    fig.tight_layout()
    return fig


def _warmup_chart(report: TrafficReport) -> str:
    fig = _build_warmup_chart(report)
    if fig is None:
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

        for builder in (_build_waterfall_chart, _build_hit_rate_by_bucket_chart,
                        _build_cluster_size_chart, _build_warmup_chart):
            fig = builder(report)
            if fig is not None:         
                pdf.savefig(fig)
                plt.close(fig)