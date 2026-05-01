from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


@pytest.fixture()
def qa_items():
    return [
        {"query_id": "Q0001", "query": "What is CA?", "answer": "ans", "type": "definition",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
        {"query_id": "Q0002", "query": "How to configure RACH?", "answer": "ans", "type": "procedural",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
        {"query_id": "Q0003", "query": "Why failure?", "answer": "ans", "type": "diagnostic",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
        {"query_id": "Q0004", "query": "Compare LTE NR", "answer": "ans", "type": "comparative",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
        {"query_id": "Q0005", "query": "Default maxTxPower?", "answer": "ans", "type": "lookup",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
    ]


def test_convert_type_mapping(qa_items):
    from gen_router_goldset import convert
    results, counts, skipped = convert(qa_items)
    assert len(results) == 5
    assert skipped == 0
    assert results[0]["expected_route"] == "definition_explain"
    assert results[1]["expected_route"] == "procedural"
    assert results[2]["expected_route"] == "diagnostic"
    assert results[3]["expected_route"] == "comparative"
    assert results[4]["expected_route"] == "structured_lookup"


def test_convert_query_id_format(qa_items):
    from gen_router_goldset import convert
    results, _, _ = convert(qa_items, start_id=1)
    assert results[0]["query_id"] == "RQ0001"
    assert results[4]["query_id"] == "RQ0005"


def test_convert_preserves_qa_query_id(qa_items):
    from gen_router_goldset import convert
    results, _, _ = convert(qa_items)
    assert results[0]["qa_query_id"] == "Q0001"
    assert results[4]["qa_query_id"] == "Q0005"


def test_convert_start_id_offset(qa_items):
    from gen_router_goldset import convert
    results, _, _ = convert(qa_items, start_id=10)
    assert results[0]["query_id"] == "RQ0010"
    assert results[4]["query_id"] == "RQ0014"


def test_convert_unknown_type_skipped():
    from gen_router_goldset import convert
    items = [
        {"query_id": "Q0001", "query": "q", "answer": "a", "type": "unknown_type",
         "source_doc": "x.md", "spec_number": "", "release": "Rel-18"},
    ]
    results, counts, skipped = convert(items)
    assert len(results) == 0
    assert skipped == 1


def test_convert_counts(qa_items):
    from gen_router_goldset import convert
    _, counts, _ = convert(qa_items)
    assert counts["definition_explain"] == 1
    assert counts["structured_lookup"] == 1


def test_cli_dry_run(tmp_path):
    from gen_router_goldset import convert
    input_path = tmp_path / "qa.jsonl"
    items = [
        {"query_id": "Q0001", "query": "What is CA?", "answer": "a", "type": "definition",
         "source_doc": "d.md", "spec_number": "29.502", "release": "Rel-18"},
    ]
    input_path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    output_path = tmp_path / "router.jsonl"
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/gen_router_goldset.py",
         "--input", str(input_path), "--output", str(output_path), "--dry-run"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert result.returncode == 0
    assert not output_path.exists()


def test_cli_writes_output(tmp_path):
    import subprocess
    input_path = tmp_path / "qa.jsonl"
    items = [
        {"query_id": "Q0001", "query": "What is CA?", "answer": "a", "type": "definition",
         "source_doc": "d.md", "spec_number": "29.502", "release": "Rel-18"},
    ]
    input_path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    output_path = tmp_path / "router.jsonl"
    result = subprocess.run(
        [sys.executable, "scripts/gen_router_goldset.py",
         "--input", str(input_path), "--output", str(output_path)],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert result.returncode == 0
    assert output_path.exists()
    lines = [json.loads(l) for l in output_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["expected_route"] == "definition_explain"
