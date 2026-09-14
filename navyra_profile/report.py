"""
navyra_profile.report — human-readable report generation.

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


INK = "#14171F"
PAPER = "#FAFAF8"
HAIRLINE = "#D8D6CE"
MUTED = "#6B6E76"
BLUE = "#2A5C8A"     # T2 / primary
GREEN = "#4C7A3D"    # T1
AMBER = "#C7791F"    # T4 / risk
SLATE = "#8A8D93"    # T3
PURPLE = "#5B5E8A"   # clusters

_SERIF_HTML = "Georgia, 'Iowan Old Style', 'Palatino Linotype', serif"
_SANS_HTML = "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"
_MONO_HTML = "ui-monospace, 'SF Mono', 'Cascadia Code', Consolas, monospace"

_CSS = f"""
* {{ box-sizing: border-box; }}
body {{ font-family: {_SANS_HTML}; max-width: 820px; margin: 0 auto;
       padding: 56px 24px 80px; color: {INK}; background: {PAPER};
       line-height: 1.55; font-size: 15px; }}
h1 {{ font-family: {_SERIF_HTML}; font-size: 30px; font-weight: 400;
     margin: 0 0 4px; letter-spacing: -0.01em; }}
.meta {{ font-family: {_MONO_HTML}; font-size: 12.5px; color: {MUTED}; margin: 0 0 32px; }}
h2 {{ font-family: {_SERIF_HTML}; font-size: 18px; font-weight: 400;
     margin: 48px 0 4px; padding-bottom: 10px; border-bottom: 1px solid {HAIRLINE}; }}
img {{ max-width: 100%; display: block; margin: 18px 0 6px; }}
.specs {{ display: flex; border-top: 1px solid {INK}; border-bottom: 1px solid {INK}; margin: 28px 0 8px; }}
.spec {{ flex: 1; padding: 14px 16px; border-left: 1px solid {HAIRLINE}; }}
.spec:first-child {{ border-left: none; }}
.spec .label {{ font-family: {_SANS_HTML}; font-size: 11px; color: {MUTED}; margin-bottom: 6px; }}
.spec .value {{ font-family: {_MONO_HTML}; font-size: 26px; line-height: 1; }}
.spec .sub {{ font-family: {_SANS_HTML}; font-size: 11.5px; color: {MUTED}; margin-top: 6px; }}
.spec.t1 .value {{ color: {GREEN}; }}
.spec.t2 .value {{ color: {BLUE}; }}
.spec.t3 .value {{ color: {SLATE}; }}
.spec.t4 .value {{ color: {AMBER}; }}
.note {{ font-size: 13px; color: {MUTED}; font-style: italic; margin: 10px 0 0; }}
.risk-box {{ border: 1px solid {HAIRLINE}; border-left: 3px solid {AMBER};
            padding: 16px 20px; margin-top: 14px; background: transparent; }}
.risk-box > p {{ margin-top: 0; }}
.near-dup-list {{ list-style: none; padding: 0; margin: 12px 0 0; }}
.near-dup-list li {{ border-top: 1px solid {HAIRLINE}; padding: 10px 0;
                     font-family: {_MONO_HTML}; font-size: 12.5px; line-height: 1.6; }}
.near-dup-list li:first-child {{ border-top: none; }}
.near-dup-list li div {{ margin: 2px 0; }}
mark {{ background: transparent; color: {AMBER}; font-weight: 600; padding: 0; }}
pre.summary {{ font-family: {_MONO_HTML}; font-size: 12.5px; background: transparent;
              border: 1px solid {HAIRLINE}; padding: 18px 20px; overflow-x: auto; color: {INK}; }}
"""


def _fig_to_base64(fig) -> str:
    """Serialize a Matplotlib `fig` to a base64-encoded PNG string.

    The returned string can be embedded directly in an HTML `<img>` tag.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=PAPER)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _style_axes(ax, spines_to_hide=("top", "right")):
    """Shared chart styling: hairline spines, muted ticks — applied to
    every chart so they read as one consistent instrument, not four
    charts with four different defaults.
    """
    ax.set_facecolor(PAPER)
    for s in spines_to_hide:
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        if s.get_visible():
            s.set_color(HAIRLINE)
    ax.tick_params(colors=MUTED, labelsize=9)


def _build_waterfall_chart(report: TrafficReport):
    """Build a waterfall chart showing the tiered breakdown of traffic:
    T1 exact repeats (already free) -> T2 same-bucket semantic (addressable today) -> T3 cross-bucket semantic (future) -> remainder (unique traffic).
    Returns a matplotlib figure object.
    """
    t1 = report.exact_repeat_rate
    sem_share = 1.0 - t1
    t2 = report.would_hit_rate * sem_share
    t3 = report.cross_bucket_rate * sem_share
    remainder = max(0.0, 1.0 - t1 - t2 - t3)

    tier_labels = ["T1 exact\nrepeats", "T2 same-bucket\nsemantic",
                   "T3 cross-bucket\nsemantic", "Remainder\n(unique)"]
    values = [t1, t2, t3, remainder]
    counts = [round(v * report.n_prompts) for v in values]
    colors = [GREEN, BLUE, SLATE, "#C9C7BE"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(PAPER)
    running = 0.0
    MIN_VISIBLE = 0.015  # bars below this fraction get a minimum drawn height

    for i, (label, val, n) in enumerate(zip(tier_labels, values, counts)):
        draw_height = max(val, MIN_VISIBLE) if val > 0 else 0
        ax.bar(label, draw_height, bottom=running, color=colors[i], width=1.0,
               edgecolor=PAPER, linewidth=2)

        label_y = running + draw_height / 2
        text = f"{val:.1%}\n(n={n:,})" if val > 0 else "0.0%"
        text_color = "white" if draw_height > 0.03 else colors[i]
        va = "center" if draw_height > 0.03 else "bottom"
        label_y = label_y if draw_height > 0.03 else running + draw_height + 0.01

        ax.text(i, label_y, text, ha="center", va=va,
                 color=text_color, fontweight="bold", fontsize=9,
                 fontfamily="monospace")
        running += val  # advance by the REAL value, not the padded draw_height

    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_ylabel("Share of total traffic", color=MUTED, fontsize=10)
    ax.set_title("Tiered waterfall: where your traffic actually goes",
                 fontsize=13, fontfamily="serif", color=INK)
    ax.text(0.5, 1.10, f"Based on {report.n_prompts:,} prompts · match radius {report.radius}",
             transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    _style_axes(ax, spines_to_hide=("top", "right", "left"))
    ax.tick_params(axis="x", length=0)

    legend_text = (
        "T1 — identical prompts, already served free by prefix caching.\n"
        "T2 — different wording, same meaning, same length range: today's addressable "
        "opportunity (upper-bound estimate, not a guaranteed saving).\n"
        "T3 — same meaning, different length range: not addressable yet.\n"
        "Remainder — traffic with no detected repetition."
    )
    fig.text(0.02, -0.02, legend_text, fontsize=8, color=MUTED, va="top", wrap=True)

    fig.tight_layout()
    return fig


def _waterfall_chart(report: TrafficReport) -> str:
    """Return a base64 PNG string for the waterfall chart for `report`.

    Returns an empty string when the figure cannot be produced.
    """
    fig = _build_waterfall_chart(report)
    if fig is None:
        return ""
    return _fig_to_base64(fig)


def _build_hit_rate_by_bucket_chart(report: TrafficReport):
    """Build a matplotlib figure showing hit rate per length bucket.

    Returns a figure object, or `None` when there is no bucket data.
    """
    if not report.hit_rate_by_bucket:
        return None

    buckets = sorted(report.hit_rate_by_bucket)
    rates = [report.hit_rate_by_bucket[b][0] for b in buckets]
    counts = [report.hit_rate_by_bucket[b][1] for b in buckets]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PAPER)
    bars = ax.bar([str(b) for b in buckets], rates, color=BLUE, width=0.55)

    for bar, rate, n in zip(bars, rates, counts):
        label = f"{rate:.1%}\n(n={n:,})"
        # low-sample-size warning: a rate from very few prompts is noisy
        if n < 30:
            label += "\n⚠ small sample"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 label, ha="center", va="bottom", fontsize=8,
                 fontfamily="monospace", color=INK)

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, max(rates + [0.1]) * 1.35)
    ax.set_xlabel("Prompt length bucket (tokens)", color=MUTED, fontsize=10)
    ax.set_ylabel("Same-bucket hit rate", color=MUTED, fontsize=10)
    ax.set_title("Hit rate by length bucket", fontsize=13, fontfamily="serif", color=INK, pad=14)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.7)
    _style_axes(ax)
    ax.tick_params(axis="x", length=0)

    caption = (
        "Each bar shows what share of prompts in that length range found a "
        "semantic match (T2) elsewhere in the same range. 'n' is the number of "
        "prompts that length range actually contains — small n means the rate "
        "is based on limited data and may not generalise."
    )
    fig.text(0.02, -0.05, caption, fontsize=8, color=MUTED, va="top", wrap=True)
    fig.tight_layout()
    return fig


def _hit_rate_by_bucket_chart(report: TrafficReport) -> str:
    """Return a base64 PNG string for the hit-rate-by-bucket chart.

    Returns an empty string when there is no data to plot.
    """
    fig = _build_hit_rate_by_bucket_chart(report)
    if not fig:
        return ""
    return _fig_to_base64(fig)


def _build_cluster_size_chart(report: TrafficReport):
    """Build a matplotlib figure showing the distribution of cluster sizes.

    Returns a figure object, or `None` when there are no clusters.
    """
    if not report.cluster_sizes:
        return None
    top = report.cluster_sizes[:20]
    shown_total = sum(top)
    all_total = sum(report.cluster_sizes)
    shown_share = shown_total / max(1, all_total)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PAPER)
    ax.bar(range(1, len(top) + 1), top, color=PURPLE, width=0.6)
    ax.bar(1, top[0], color=INK)  # top cluster gets a distinct shade

    ax.text(1, top[0], f"largest:\n{top[0]:,} prompts", ha="center", va="bottom",
             fontsize=8, fontweight="bold", fontfamily="monospace", color=INK)

    ax.set_xlabel("Cluster rank (largest first)", color=MUTED, fontsize=10)
    ax.set_ylabel("Prompts in cluster", color=MUTED, fontsize=10)
    ax.set_title(
        f"Cluster size distribution — top 20 of {report.n_clusters:,} clusters total",
        fontsize=13, fontfamily="serif", color=INK, pad=14)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.7)
    _style_axes(ax)
    ax.tick_params(axis="x", length=0)

    caption = (
        f"A cluster is a group of prompts that all matched each other semantically. "
        f"The 20 largest clusters shown here account for {shown_share:.1%} of all "
        f"traffic that formed any cluster ({shown_total:,} of {all_total:,} prompts). "
        f"The single largest cluster alone is {report.top_cluster_share:.1%} of your "
        f"total traffic."
    )
    fig.text(0.02, -0.06, caption, fontsize=8, color=MUTED, va="top", wrap=True)
    fig.tight_layout()
    return fig


def _cluster_size_chart(report: TrafficReport) -> str:
    """Return a base64 PNG string for the cluster-size chart.

    Returns an empty string when there is no cluster data.
    """
    fig = _build_cluster_size_chart(report)
    if fig is None:
        return ""
    return _fig_to_base64(fig)


def _build_warmup_chart(report: TrafficReport):
    """Build a matplotlib figure showing the cache warm-up curve.

    The plot shows how the observed T2 hit rate evolves as prompts are
    processed in their original order. Returns a figure or `None` when
    no warmup history is present.
    """
    history = getattr(report, "warmup_history", [])
    if not history:
        return None
    xs, ys = zip(*history)
    final_rate = report.would_hit_rate

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PAPER)
    ax.plot(xs, [y * 100 for y in ys], color=BLUE, linewidth=2, label="observed rate")
    ax.axhline(final_rate * 100, color=MUTED, linestyle="--", linewidth=1,
                label=f"final rate ({final_rate:.1%})")

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Prompts processed (in original order)", color=MUTED, fontsize=10)
    ax.set_ylabel("Cumulative T2 hit rate", color=MUTED, fontsize=10)
    ax.set_title("Cache warm-up curve", fontsize=13, fontfamily="serif", color=INK, pad=14)
    ax.grid(color=HAIRLINE, linewidth=0.7)
    _style_axes(ax)
    legend = ax.legend(fontsize=8, loc="lower right", frameon=False)
    for text in legend.get_texts():
        text.set_color(MUTED)

    caption = (
        "This shows how the semantic hit rate changes as traffic streams in, "
        "simulating a cache starting empty. The rate is low early on (nothing to "
        "match against yet) and rises as more prompts accumulate in the bank. "
        "The dashed line marks the final overall rate for reference."
    )
    fig.text(0.02, -0.05, caption, fontsize=8, color=MUTED, va="top", wrap=True)
    fig.tight_layout()
    return fig


def _warmup_chart(report: TrafficReport) -> str:
    """Return a base64 PNG string for the warm-up chart, or empty string
    when no data is available.
    """
    fig = _build_warmup_chart(report)
    if fig is None:
        return ""
    return _fig_to_base64(fig)


def _highlight_diff(a: str, b: str) -> tuple[str, str]:
    """Return marked-up HTML snippets highlighting differences between
    token sequences `a` and `b`.
    """
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
    """Return HTML for the near-duplicate risk box, showing example
    near-duplicate prompt pairs with differing tokens highlighted.
    """
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


def _specs_strip(report: TrafficReport) -> str:
    """Headline datasheet strip — label over value, hairline dividers.
    A quick-glance summary of the four tiers, sitting above the full
    charts and explanatory captions further down the report.
    """
    return f"""
    <div class="specs">
      <div class="spec t1">
        <div class="label">T1 exact</div>
        <div class="value">{report.exact_repeat_rate:.1%}</div>
        <div class="sub">free via prefix caching</div>
      </div>
      <div class="spec t2">
        <div class="label">T2 same-bucket</div>
        <div class="value">{report.would_hit_rate:.1%}</div>
        <div class="sub">upper-bound proxy</div>
      </div>
      <div class="spec t3">
        <div class="label">T3 cross-bucket</div>
        <div class="value">{report.cross_bucket_rate:.1%}</div>
        <div class="sub">future-addressable</div>
      </div>
      <div class="spec t4">
        <div class="label">T4 flagged</div>
        <div class="value">{report.near_dup_count}</div>
        <div class="sub">answer-replay risk</div>
      </div>
    </div>
    """


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
        _specs_strip(report),

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
    """Render the same report content as a multi-page PDF."""
    with PdfPages(path) as pdf:
        # page 1: title, key wording rule, and full text summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor(PAPER)
        ax.axis("off")
        ax.text(0.5, 0.97, "Navyra Traffic Profile", ha="center",
                 fontsize=20, fontfamily="serif", color=INK)
        ax.text(0.5, 0.94,
                 f"{report.n_prompts:,} prompts analysed · match radius {report.radius}",
                 ha="center", fontsize=9, color=MUTED)
        ax.text(0.05, 0.88,
                 "Note: T2 (same-bucket semantic) is an upper-bound proxy — up to this "
                 "share of traffic is engine-addressable today. It is not a promised "
                 "saving; the true hit rate is gated inside the model and measured in "
                 "the free trial.",
                 fontsize=8, color=MUTED, style="italic", va="top", wrap=True)
        ax.text(0.05, 0.80, report.summary(), fontsize=9, family="monospace",
                 va="top", color=INK)
        pdf.savefig(fig, bbox_inches="tight", facecolor=PAPER)
        plt.close(fig)

        # one page per chart, each with its caption included
        for builder in (_build_waterfall_chart, _build_hit_rate_by_bucket_chart,
                        _build_cluster_size_chart, _build_warmup_chart):
            fig = builder(report)
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight", facecolor=PAPER)
                plt.close(fig)