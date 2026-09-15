"""Tests for navyra_profile.cli."""

import json
import subprocess
import sys

import pytest


def _write_jsonl(path, prompts, field="prompt"):
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps({field: p}) + "\n")


def run_cli(*args, cwd=None):
    """Run the actual navyra-profile CLI as a subprocess, capturing
    stdout/stderr/returncode — tests the real, installed entry point,
    not just Python functions called directly.
    """
    result = subprocess.run(
        ["navyra-profile", *args],
        capture_output=True, text=True, cwd=cwd,
    )
    return result


class TestAnalyse:
    def test_analyse_runs_successfully(self, tmp_path):
        f = tmp_path / "prompts.jsonl"
        _write_jsonl(f, ["hello world"] * 5 + ["something else entirely"] * 3)

        result = run_cli("analyse", str(f))
        assert result.returncode == 0
        assert "T1 exact repeats" in result.stdout
        assert "T2 same-bucket semantic" in result.stdout

    def test_missing_file_gives_clean_error_not_traceback(self):
        result = run_cli("analyse", "/tmp/definitely_does_not_exist_12345.jsonl")
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "error: file not found" in result.stderr

    def test_malformed_json_gives_clean_error(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text("not valid json at all\n")

        result = run_cli("analyse", str(f))
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "not valid JSON" in result.stderr

    def test_missing_field_gives_clean_error(self, tmp_path):
        f = tmp_path / "wrong_field.jsonl"
        f.write_text('{"text": "hello"}\n')

        result = run_cli("analyse", str(f))
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "missing field" in result.stderr

    def test_empty_file_gives_clean_error(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")

        result = run_cli("analyse", str(f))
        assert result.returncode == 1
        assert "Traceback" not in result.stderr

    def test_custom_field_name_works(self, tmp_path):
        f = tmp_path / "custom.jsonl"
        _write_jsonl(f, ["hello"] * 3, field="text")

        result = run_cli("analyse", str(f), "--field", "text")
        assert result.returncode == 0

    def test_html_flag_writes_a_real_file(self, tmp_path):
        f = tmp_path / "prompts.jsonl"
        _write_jsonl(f, ["hello world"] * 5)
        out = tmp_path / "report.html"

        result = run_cli("analyse", str(f), "--html", str(out))
        assert result.returncode == 0
        assert out.exists()
        assert out.stat().st_size > 1000  # a real report, not an empty stub
        assert "<html>" in out.read_text()

    def test_pdf_flag_writes_a_real_file(self, tmp_path):
        f = tmp_path / "prompts.jsonl"
        _write_jsonl(f, ["hello world"] * 5)
        out = tmp_path / "report.pdf"

        result = run_cli("analyse", str(f), "--pdf", str(out))
        assert result.returncode == 0
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_radius_flag_is_accepted(self, tmp_path):
        f = tmp_path / "prompts.jsonl"
        _write_jsonl(f, ["hello world"] * 5)

        result = run_cli("analyse", str(f), "--radius", "2")
        assert result.returncode == 0


class TestSynth:
    def test_synth_writes_requested_number_of_prompts(self, tmp_path):
        out = tmp_path / "synth.jsonl"
        result = run_cli("synth", "--n", "50", "--template-share", "0.4",
                          "-o", str(out))
        assert result.returncode == 0
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 50
        # every line must be valid JSON with a "prompt" key
        for line in lines:
            obj = json.loads(line)
            assert "prompt" in obj

    def test_synth_missing_required_args_fails_cleanly(self):
        # --n and --template-share and -o are all required; omitting
        # them should be an argparse-level failure, not a crash deep
        # inside the program
        result = run_cli("synth")
        assert result.returncode != 0
        assert "Traceback" not in result.stderr

    def test_synth_reproducible_with_same_seed(self, tmp_path):
        out1 = tmp_path / "a.jsonl"
        out2 = tmp_path / "b.jsonl"
        run_cli("synth", "--n", "30", "--template-share", "0.4",
                 "--seed", "0", "-o", str(out1))
        run_cli("synth", "--n", "30", "--template-share", "0.4",
                 "--seed", "0", "-o", str(out2))
        assert out1.read_text() == out2.read_text()


class TestValidate:
    def test_validate_runs_and_prints_summary(self):
        result = run_cli("validate", "--n", "100", "--shares", "0.2,0.5")
        assert result.returncode == 0
        assert "Mean absolute error" in result.stdout


class TestNoSubcommand:
    def test_no_subcommand_shows_usage_not_crash(self):
        result = run_cli()
        assert result.returncode != 0
        assert "Traceback" not in result.stderr