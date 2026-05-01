import subprocess
import sys
from pathlib import Path


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/convert_pdf_to_md.py", *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_help_lists_doc_types():
    repo = Path(__file__).resolve().parents[2]
    proc = _run(["--help"], cwd=repo)
    assert proc.returncode == 0
    assert "--doc-type" in proc.stdout
    assert "spec" not in proc.stdout  # spec은 PDF 변환 대상 아님


def test_spec_doc_type_rejected(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    proc = _run(
        ["--input-file", str(pdf), "--doc-type", "spec",
         "--output-dir", str(tmp_path / "out")],
        cwd=repo,
    )
    assert proc.returncode != 0


def test_dry_run_announces_parser(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    proc = _run(
        ["--input-file", str(pdf), "--doc-type", "parameter_ref",
         "--output-dir", str(tmp_path / "out"), "--dry-run"],
        cwd=repo,
    )
    assert proc.returncode == 0
    assert "parameter_ref" in proc.stdout
    assert "[DRY RUN]" in proc.stdout


def test_directory_continues_on_failure(tmp_path):
    """parse_pdf NotImplementedError on one file shouldn't kill batch (non-dry-run)."""
    # Create 2 fake pdfs
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    out = tmp_path / "out"

    repo = Path(__file__).resolve().parents[2]
    proc = _run(
        ["--input-dir", str(tmp_path), "--doc-type", "parameter_ref",
         "--output-dir", str(out)],  # no --dry-run -> hits NotImplementedError
        cwd=repo,
    )
    # NotImplementedError caught per-file, batch continues, exit 0
    assert proc.returncode == 0, proc.stderr
    # Both files attempted
    assert "a.pdf" in proc.stdout
    assert "b.pdf" in proc.stdout
    # Both errors logged
    assert proc.stderr.count("NotImplementedError") == 2
