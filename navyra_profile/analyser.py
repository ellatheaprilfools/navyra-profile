"""
navyra_profile.analyzer — turn fingerprints into traffic statistics.

Given a stream of prompts, answers the question a GPU owner actually has:
"how much of my traffic is semantically repeated — and therefore how much
of the computation is redundant?"

The report is a TIERED WATERFALL, not one number, because different
kinds of repetition are served by different mechanisms:

  Tier 1 — EXACT repeats (hash-identical text). Already served free by
           vLLM's prefix caching. Reported and DEDUCTED honestly.
  Tier 2 — SAME-BUCKET semantic hits (similar meaning, same token-length
           bucket, not exact). The class Navyra's engine addresses today.
           This is the headline number. Note: an embedding hit is an
           UPPER-BOUND PROXY for the engine's true hit rate (which is
           gated inside the model); the report must say so.
  Tier 3 — CROSS-BUCKET semantic hits (similar meaning, different length
           bucket). Future-addressable; reported separately.
  Tier 4 — NEAR-DUPLICATES: very high similarity but small critical
           edits (numbers, dates, names differ). Flagged as the class
           where whole-answer caches (GPTCache-style) silently serve
           WRONG answers — and where verified computation reuse is the
           safe alternative. This tier is reported as a risk finding,
           never added to any hit rate.

Everything here is measurement. No engine, no proprietary thresholds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .fingerprint import Fingerprinter, hamming


@dataclass
class TrafficReport:
    n_prompts: int = 0
    # tiered waterfall
    exact_repeat_rate: float = 0.0        # tier 1: prefix-caching territory
    would_hit_rate: float = 0.0           # tier 2: same-bucket semantic
    cross_bucket_rate: float = 0.0        # tier 3: future-addressable
    near_dup_count: int = 0               # tier 4: risk finding
    near_dup_examples: list = field(default_factory=list)
    # detail
    hit_rate_by_bucket: dict = field(default_factory=dict)
    n_clusters: int = 0
    top_cluster_share: float = 0.0
    cluster_sizes: list = field(default_factory=list)
    radius: int = 6
    notes: str = ""

    def summary(self) -> str:
        lines = [
            f"prompts analysed          : {self.n_prompts:,}",
            f"T1 exact repeats          : {self.exact_repeat_rate:.1%}  "
            f"(already free via prefix caching)",
            f"T2 same-bucket semantic   : {self.would_hit_rate:.1%}  "
            f"(engine-addressable today; upper-bound proxy)",
            f"T3 cross-bucket semantic  : {self.cross_bucket_rate:.1%}  "
            f"(future-addressable)",
            f"T4 near-duplicates flagged: {self.near_dup_count:,}  "
            f"(unsafe for answer-replay caches)",
            f"semantic clusters         : {self.n_clusters:,}",
            f"traffic in top cluster    : {self.top_cluster_share:.1%}",
        ]
        if self.hit_rate_by_bucket:
            lines.append("hit rate by length bucket:")
            for b, (hr, n) in sorted(self.hit_rate_by_bucket.items()):
                lines.append(f"  bucket {b:>4} tokens : {hr:6.1%}  (n={n:,})")
        return "\n".join(lines)


def _bucket(n_tokens: int, size: int = 8) -> int:
    """Round token count up to its length bucket."""
    return ((n_tokens + size - 1) // size) * size


def analyse(prompts: list[str],
            radius: int = 6,
            bucket_size: int = 8,
            fingerprinter: Fingerprinter | None = None) -> TrafficReport:
    """Streaming would-hit analysis. Simulates a cache warming as traffic
    arrives in order: each prompt either matches an earlier fingerprint
    (within `radius` Hamming bits) = HIT, or joins the bank = MISS.

    NOTE for productionisation: this reference implementation is O(n^2)
    in the worst case (each prompt compared against the growing bank).
    Fine to ~50k prompts. Making it scale to 1M+ (bucketed banks,
    vectorised chunk comparisons, or an LSH prefilter) is part of the
    project scope.
    """
    fp = fingerprinter or Fingerprinter()
    fps = fp.fingerprints(prompts)
    tok_counts = [max(1, len(p.split())) for p in prompts]   # rough tokens
    buckets = [_bucket(t, bucket_size) for t in tok_counts]

    # ---- tier 1: exact repeats (normalised hash) -------------------------
    import hashlib as _h
    def _norm(p): return " ".join(p.lower().split())
    seen_exact: set = set()
    exact = np.zeros(len(prompts), dtype=bool)
    for i, p in enumerate(prompts):
        hsh = _h.md5(_norm(p).encode()).hexdigest()
        if hsh in seen_exact:
            exact[i] = True
        seen_exact.add(hsh)

    bank: list[int] = []            # indices into fps
    bank_by_bucket: dict = {}
    hits = 0
    cross_hits = 0
    hit_by_bucket: Counter = Counter()
    n_by_bucket: Counter = Counter()
    cluster_of = np.full(len(prompts), -1, dtype=np.int64)
    next_cluster = 0
    near_dups: list = []            # (i, j, example pair) tier-4 findings
    NEAR_DUP_RADIUS = 2             # extremely close fingerprints

    for i in range(len(prompts)):
        b = buckets[i]
        n_by_bucket[b] += 1
        if exact[i]:
            # tier 1 — counted separately, excluded from semantic tiers
            cluster_of[i] = -2
            continue
        # tier 2: same-bucket semantic
        cand = bank_by_bucket.get(b, [])
        matched = False
        if cand:
            d = hamming(fps[np.asarray(cand)], np.uint64(fps[i]))
            j = int(np.argmin(d))
            if d[j] <= radius:
                matched = True
                hits += 1
                hit_by_bucket[b] += 1
                cluster_of[i] = cluster_of[cand[j]]
                # tier 4: near-duplicate flag — near-identical fingerprint
                # but text differs (exact[] already excluded identicals).
                # PROJECT TASK: refine with edit-distance + differing
                # digits/dates/names detection; keep <=5 example pairs.
                if d[j] <= NEAR_DUP_RADIUS and len(near_dups) < 5:
                    near_dups.append((prompts[cand[j]][:80],
                                      prompts[i][:80]))
        if not matched:
            # tier 3: would it have hit in ANY other bucket?
            others = [k for bb, ks in bank_by_bucket.items()
                      if bb != b for k in ks]
            if others:
                d2 = hamming(fps[np.asarray(others)], np.uint64(fps[i]))
                if int(d2.min()) <= radius:
                    cross_hits += 1
            cluster_of[i] = next_cluster
            next_cluster += 1
            bank.append(i)
            bank_by_bucket.setdefault(b, []).append(i)

    sizes = np.bincount(cluster_of[cluster_of >= 0])
    sizes_sorted = sorted(sizes.tolist(), reverse=True)

    n_sem = int((~exact).sum())     # prompts entering semantic tiers
    return TrafficReport(
        n_prompts=len(prompts),
        exact_repeat_rate=float(exact.mean()),
        cross_bucket_rate=cross_hits / max(1, n_sem),
        near_dup_count=len(near_dups),
        near_dup_examples=near_dups,
        would_hit_rate=hits / max(1, n_sem),
        hit_rate_by_bucket={
            b: (hit_by_bucket[b] / n_by_bucket[b], n_by_bucket[b])
            for b in n_by_bucket},
        n_clusters=next_cluster,
        top_cluster_share=(sizes_sorted[0] / len(prompts))
        if sizes_sorted else 0.0,
        cluster_sizes=sizes_sorted[:100],
        radius=radius,
    )
