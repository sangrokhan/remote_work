import subprocess
import sys
from pathlib import Path


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/run_ingest.py", *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_dry_run_md_file(tmp_path):
    md = tmp_path / "sample.md"
    md.write_text("# Section A\nHello world.\n# Section B\nMore content.\n")

    repo = Path(__file__).resolve().parents[2]  # spar/
    proc = _run(
        ["--input-file", str(md), "--doc-type", "spec", "--dry-run"],
        cwd=repo,
    )
    assert proc.returncode == 0, proc.stderr
    assert "DRY RUN" in proc.stdout
    assert "2 chunks" in proc.stdout  # Section A + B


def test_pdf_input_rejected(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    repo = Path(__file__).resolve().parents[2]
    proc = _run(
        ["--input-file", str(pdf), "--doc-type", "spec", "--dry-run"],
        cwd=repo,
    )
    assert proc.returncode != 0
    assert "convert_pdf_to_md.py" in proc.stderr
