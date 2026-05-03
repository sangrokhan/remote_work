import sys
from pathlib import Path
import pytest

_SCRIPTS = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_find_chunk_keywords_uses_entity_terms(monkeypatch):
    """_find_chunk_keywords must detect Samsung entity names."""
    import run_ingest
    monkeypatch.setattr(run_ingest, "_KEYWORDS", {"nrDlCellMaxTxPower", "HO"})
    result = run_ingest._find_chunk_keywords(
        "The parameter nrDlCellMaxTxPower controls downlink power", run_ingest._KEYWORDS
    )
    assert "nrDlCellMaxTxPower" in result


def test_find_chunk_keywords_signature_accepts_set(monkeypatch):
    import run_ingest
    monkeypatch.setattr(run_ingest, "_KEYWORDS", {"CA", "RACH"})
    result = run_ingest._find_chunk_keywords("CA and RACH procedures", run_ingest._KEYWORDS)
    assert "CA" in result
    assert "RACH" in result
