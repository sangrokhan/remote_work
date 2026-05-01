from spar.retrieval.query_rewriter import (
    MAX_HISTORY_TURNS,
    build_context,
    extract_relevant_acronyms,
    format_history,
)

SAMPLE_ACRONYMS = {
    "global": {
        "PDCP": {"expansion": "Packet Data Convergence Protocol", "variants": []},
        "RLC": {"expansion": "Radio Link Control", "variants": ["Rlc"]},
    }
}


def test_format_history_empty():
    assert format_history([]) == ""


def test_format_history_labels():
    history = [
        {"role": "user", "content": "What is PDCP?"},
        {"role": "assistant", "content": "PDCP handles header compression."},
    ]
    result = format_history(history)
    assert "User: What is PDCP?" in result
    assert "Assistant: PDCP handles header compression." in result


def test_format_history_max_turns():
    history = [{"role": "user", "content": str(i)} for i in range(20)]
    result = format_history(history, max_turns=MAX_HISTORY_TURNS)
    lines = result.strip().splitlines()
    assert len(lines) == MAX_HISTORY_TURNS * 2


def test_extract_relevant_acronyms_match():
    result = extract_relevant_acronyms("explain PDCP timer", [], SAMPLE_ACRONYMS)
    assert "PDCP" in result
    assert result["PDCP"] == "Packet Data Convergence Protocol"


def test_extract_relevant_acronyms_variant_match():
    result = extract_relevant_acronyms("Rlc layer behavior", [], SAMPLE_ACRONYMS)
    assert "Rlc" in result


def test_extract_relevant_acronyms_no_match():
    result = extract_relevant_acronyms("explain timer", [], SAMPLE_ACRONYMS)
    assert result == {}


def test_extract_relevant_acronyms_from_history():
    history = [{"role": "user", "content": "What about RLC?"}]
    result = extract_relevant_acronyms("and that?", history, SAMPLE_ACRONYMS)
    assert "RLC" in result


def test_build_context_with_history_and_acronyms():
    history = [{"role": "user", "content": "What is PDCP?"}]
    result = build_context("how does it work in NR?", history, SAMPLE_ACRONYMS)
    assert "Conversation history:" in result
    assert "PDCP" in result
    assert "Relevant acronyms:" in result


def test_build_context_no_history():
    result = build_context("explain RLC", [], SAMPLE_ACRONYMS)
    assert "Conversation history:" not in result
    assert "RLC" in result


def test_build_context_no_match():
    result = build_context("explain timer", [], SAMPLE_ACRONYMS)
    assert result == ""
