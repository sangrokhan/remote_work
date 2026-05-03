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


def test_keywords_global_is_set_type(monkeypatch):
    """_KEYWORDS module global must be a set (not dict), so ingest_file call is correct."""
    import run_ingest
    assert isinstance(run_ingest._KEYWORDS, set), (
        f"_KEYWORDS should be set[str], got {type(run_ingest._KEYWORDS)}"
    )


def test_find_chunk_keywords_with_module_level_keywords(monkeypatch):
    """_find_chunk_keywords called with _KEYWORDS (module-level) must detect patched terms."""
    import run_ingest
    # Simulate what ingest_file does: _find_chunk_keywords(c["text"], _KEYWORDS)
    patched = {"nrDlCellMaxTxPower", "pmRrcConnEstabAtt"}
    monkeypatch.setattr(run_ingest, "_KEYWORDS", patched)
    # Call exactly as ingest_file does at line ~158
    result = run_ingest._find_chunk_keywords(
        "Set nrDlCellMaxTxPower and monitor pmRrcConnEstabAtt counter",
        run_ingest._KEYWORDS,  # this is the patched set
    )
    assert "nrDlCellMaxTxPower" in result
    assert "pmRrcConnEstabAtt" in result
