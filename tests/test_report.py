"""Tests for navyra_profile.report."""

import base64
import binascii

import pytest

from navyra_profile.analyser import analyse, TrafficReport
from navyra_profile.report import (
    _waterfall_chart, _hit_rate_by_bucket_chart, _cluster_size_chart,
    _warmup_chart, _highlight_diff, _near_dup_box, html_report, pdf_report,
)


@pytest.fixture
def real_report():
    """A real TrafficReport from real analyse(), not a hand-built stub —
    exercises the actual data shapes report.py has to handle.
    """
    prompts = (["hello, how are you today?"] * 4
               + ["what is the current weather forecast?"] * 3
               + [f"a genuinely unique sentence number {i}" for i in range(15)])
    return analyse(prompts)


@pytest.fixture
def empty_report():
    """A report from a single prompt — the edge case where bucket/cluster/
    warmup data may be sparse or entirely absent.
    """
    return analyse(["just one prompt"])


def _is_valid_base64_png(s: str) -> bool:
    if not s:
        return False
    try:
        decoded = base64.b64decode(s, validate=True)
    except binascii.Error:
        return False
    return decoded[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG file signature


class TestWaterfallChart:
    def test_returns_valid_png(self, real_report):
        img = _waterfall_chart(real_report)
        assert _is_valid_base64_png(img)

    def test_never_returns_empty_even_with_sparse_data(self, empty_report):
        # waterfall always has data (T1 rate is always defined, even if 0)
        img = _waterfall_chart(empty_report)
        assert _is_valid_base64_png(img)


class TestHitRateByBucketChart:
    def test_returns_valid_png_when_data_exists(self, real_report):
        img = _hit_rate_by_bucket_chart(real_report)
        assert _is_valid_base64_png(img)

    def test_guard_clause_matches_actual_data_presence(self, real_report):
        img = _hit_rate_by_bucket_chart(real_report)
        if real_report.hit_rate_by_bucket:
            assert img != ""
        else:
            assert img == ""


class TestClusterSizeChart:
    def test_returns_valid_png_when_data_exists(self, real_report):
        img = _cluster_size_chart(real_report)
        if real_report.cluster_sizes:
            assert _is_valid_base64_png(img)
        else:
            assert img == ""


class TestWarmupChart:
    def test_returns_valid_png_when_history_exists(self, real_report):
        img = _warmup_chart(real_report)
        if getattr(real_report, "warmup_history", None):
            assert _is_valid_base64_png(img)
        else:
            assert img == ""

    def test_empty_history_returns_empty_string_not_crash(self):
        report = TrafficReport(n_prompts=1, warmup_history=[])
        assert _warmup_chart(report) == ""


class TestHighlightDiff:
    def test_identical_strings_produce_no_marks(self):
        a_hl, b_hl = _highlight_diff("hello world", "hello world")
        assert "<mark>" not in a_hl
        assert "<mark>" not in b_hl

    def test_differing_word_gets_marked_on_both_sides(self):
        a_hl, b_hl = _highlight_diff(
            "what is the status of order 12345",
            "what is the status of order 67890",
        )
        assert "<mark>12345</mark>" in a_hl
        assert "<mark>67890</mark>" in b_hl
        # the unchanged words should NOT be marked
        assert "<mark>what" not in a_hl.replace(" ", "")

    def test_html_special_characters_are_escaped(self):
        a_hl, b_hl = _highlight_diff("price < 5 & up", "price < 10 & up")
        # raw < and & must not appear unescaped, or the report's HTML breaks
        assert "<mark>5" not in a_hl or "&lt;" in a_hl
        assert "&amp;" in a_hl
        assert "&lt;" in a_hl

    def test_empty_strings_do_not_crash(self):
        a_hl, b_hl = _highlight_diff("", "")
        assert a_hl == ""
        assert b_hl == ""


class TestNearDupBox:
    def test_no_examples_gives_a_plain_message_not_empty(self):
        report = TrafficReport(n_prompts=10, near_dup_examples=[])
        html_out = _near_dup_box(report)
        assert "<p>" in html_out
        assert html_out != ""

    def test_examples_produce_one_list_item_each(self):
        report = TrafficReport(
            n_prompts=10,
            near_dup_examples=[
                ("order 123 status", "order 456 status"),
                ("meeting on monday", "meeting on tuesday"),
            ],
        )
        html_out = _near_dup_box(report)
        assert html_out.count("<li>") == 2


class TestHtmlReport:
    def test_writes_a_real_self_contained_file(self, real_report, tmp_path):
        out = tmp_path / "report.html"
        html_report(real_report, str(out))

        assert out.exists()
        content = out.read_text()
        assert "<html>" in content
        assert "</html>" in content
        assert "data:image/png;base64," in content
        # self-contained: no external file references
        assert "<link " not in content
        assert 'src="http' not in content

    def test_includes_the_wording_rule_language(self, real_report, tmp_path):
        # regression guard: T2 must never be presented as a promised
        # saving anywhere in the generated report
        out = tmp_path / "report.html"
        html_report(real_report, str(out))
        content = out.read_text().lower()
        assert "upper-bound" in content or "not a promised saving" in content

    def test_handles_report_with_no_near_dups(self, tmp_path):
        report = TrafficReport(n_prompts=5, near_dup_examples=[])
        out = tmp_path / "report.html"
        html_report(report, str(out))
        assert out.exists()


class TestPdfReport:
    def test_writes_a_real_pdf_file(self, real_report, tmp_path):
        out = tmp_path / "report.pdf"
        pdf_report(real_report, str(out))

        assert out.exists()
        # real PDFs start with this magic header
        assert out.read_bytes()[:5] == b"%PDF-"
        assert out.stat().st_size > 500

    def test_handles_sparse_report_without_crashing(self, empty_report, tmp_path):
        out = tmp_path / "sparse.pdf"
        pdf_report(empty_report, str(out))
        assert out.exists()