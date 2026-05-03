import sys
from pathlib import Path
import pytest

_SCRIPTS = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_ingest_excel_file_exists():
    import run_ingest
    assert hasattr(run_ingest, "ingest_excel_file"), "ingest_excel_file not found in run_ingest"


def test_ingest_excel_file_wrong_doc_type_raises(tmp_path):
    import run_ingest
    fake_xlsx = tmp_path / "test.xlsx"
    fake_xlsx.write_bytes(b"")
    with pytest.raises(SystemExit, match="not supported"):
        run_ingest.ingest_excel_file(None, fake_xlsx, "spec", force=False, dry_run=True)
