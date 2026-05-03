"""Pass B integration tests — dry-run mode, no Milvus required."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent.parent
_SAMPLES = _ROOT / "data" / "samples"


def _dry_run(sample: str, doc_type: str) -> str:
    result = subprocess.run(
        [
            sys.executable, "scripts/run_ingest.py",
            "--input-file", str(_SAMPLES / sample),
            "--doc-type", doc_type,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    return result.stdout


def test_parameter_ref_dry_run():
    out = _dry_run("parameter_ref_sample.xlsx", "parameter_ref")
    assert "DRY RUN" in out
    assert "chunks" in out


def test_counter_ref_dry_run():
    out = _dry_run("counter_ref_sample.xlsx", "counter_ref")
    assert "DRY RUN" in out
    assert "chunks" in out


def test_alarm_ref_dry_run():
    out = _dry_run("alarm_excel_ref_sample.xlsx", "alarm_ref")
    assert "DRY RUN" in out
    assert "chunks" in out


def test_parameter_ref_chunk_preview():
    """dry-run output should include chunk_id= preview lines."""
    out = _dry_run("parameter_ref_sample.xlsx", "parameter_ref")
    assert "chunk_id=" in out


def test_alarm_ref_wrong_doc_type_fails():
    """Wrong doc_type for xlsx → non-zero exit code."""
    result = subprocess.run(
        [
            sys.executable, "scripts/run_ingest.py",
            "--input-file", str(_SAMPLES / "alarm_excel_ref_sample.xlsx"),
            "--doc-type", "spec",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode != 0
