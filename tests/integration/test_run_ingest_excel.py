"""Pass B integration tests — dry-run mode, no Milvus required."""
import subprocess
import sys
from pathlib import Path

# Locate the spar project root: contains scripts/run_ingest.py and data/.
# Works whether tests run from main repo or from a git worktree.
def _find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent.parent,          # main repo: tests/integration/../../..
                      here.parent.parent.parent / "spar"]:  # worktree: .worktrees/<b>/spar
        if (candidate / "scripts" / "run_ingest.py").exists():
            return candidate.resolve()
    raise RuntimeError(f"Cannot locate spar root from {here}")

_ROOT = _find_root()
_SAMPLES = _ROOT / "data" / "samples"
_PYTHON = sys.executable


def _dry_run(sample: str, doc_type: str) -> str:
    result = subprocess.run(
        [_PYTHON, "scripts/run_ingest.py",
         "--input-file", str(_SAMPLES / sample),
         "--doc-type", doc_type, "--dry-run"],
        capture_output=True, text=True, cwd=str(_ROOT),
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
    out = _dry_run("parameter_ref_sample.xlsx", "parameter_ref")
    assert "chunk_id=" in out


def test_alarm_ref_wrong_doc_type_fails():
    result = subprocess.run(
        [_PYTHON, "scripts/run_ingest.py",
         "--input-file", str(_SAMPLES / "alarm_excel_ref_sample.xlsx"),
         "--doc-type", "spec", "--dry-run"],
        capture_output=True, text=True, cwd=str(_ROOT),
    )
    assert result.returncode != 0
