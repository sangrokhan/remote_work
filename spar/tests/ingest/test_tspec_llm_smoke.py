import subprocess
import sys
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tspec-llm" / "3GPP-clean" / "Rel-18"


@pytest.mark.skipif(not DATA_DIR.exists(),
                    reason="TSpec-LLM Rel-18 미다운로드 — fetch_tspec_llm.py 먼저 실행")
def test_first_md_file_dry_run():
    md_files = sorted(DATA_DIR.rglob("*.md"))
    assert md_files, "Rel-18 아래 md 없음"

    target = md_files[0]
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "scripts/run_ingest.py",
         "--input-file", str(target), "--doc-type", "spec", "--dry-run"],
        cwd=repo, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "DRY RUN" in proc.stdout
    # 3GPP spec 첫 파일은 헤더 풍부 — 최소 5청크 이상 기대
    assert "chunks" in proc.stdout
