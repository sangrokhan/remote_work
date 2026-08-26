"""export: a run's result as CSV.

Two CSVs:
  cwnd_csv     -- every congestion sample, one row per tick (the raw series,
                  meant to be plotted: snd_cwnd against t_ms is the picture).
  turns_csv    -- one row per turn: prompt size, request time, idle duration,
                  and the probe RTT samples collected during that turn's idle.
"""

from tcp_congestion import cwnd as cwndmon
from tcp_congestion import export


def _run_result():
    return {
        "label": "conversation",
        "host": "127.0.0.1",
        "port": 8888,
        "samples": [
            {"t_ms": 0.0, "local": "10.0.0.1:5000", "remote": "127.0.0.1:8888",
             "snd_cwnd": 10, "snd_ssthresh": 2147483647, "rtt_us": 500,
             "ca_state": "open"},
            {"t_ms": 10.0, "local": "10.0.0.1:5000", "remote": "127.0.0.1:8888",
             "snd_cwnd": 18, "snd_ssthresh": 2147483647, "rtt_us": 480,
             "ca_state": "open"},
        ],
        "turns": [
            {"turn": 0, "prompt_bytes": 200000, "request_ms": 1.4, "idle_ms": 2500},
            {"turn": 1, "prompt_bytes": 400100, "request_ms": 2.7, "idle_ms": 2500},
        ],
        "probes": [
            {"turn": 0, "samples": [{"ts": 1000.0, "rtt_ms": 20.1},
                                     {"ts": 1000.2, "rtt_ms": 19.8}]},
            {"turn": 1, "samples": [{"ts": 1003.0, "rtt_ms": 21.0}]},
        ],
        "idle_resets": 1,
        "peak_cwnd": 18,
        "final_cwnd": 10,
    }


def test_cwnd_csv_has_a_row_per_sample():
    text = export.cwnd_csv(_run_result())
    lines = text.strip().splitlines()
    header = lines[0].split(",")
    assert "snd_cwnd" in header
    assert "t_ms" in header
    assert len(lines) == 3  # header + 2 samples


def test_cwnd_csv_includes_label_and_host_columns():
    text = export.cwnd_csv(_run_result())
    header = text.strip().splitlines()[0].split(",")
    assert "label" in header
    assert "host" in header


def test_cwnd_csv_of_empty_run_is_header_only():
    text = export.cwnd_csv({"samples": []})
    assert len(text.strip().splitlines()) == 1


def test_cwnd_csv_field_order_matches_sample_fields():
    """SAMPLE_FIELDS from cwnd.py must all be present in the CSV header, so the
    raw series can be read against the field docs in that module."""
    header = export.cwnd_csv(_run_result()).strip().splitlines()[0].split(",")
    for field in cwndmon.SAMPLE_FIELDS:
        assert field in header


def test_turns_csv_has_a_row_per_turn():
    text = export.turns_csv(_run_result())
    lines = text.strip().splitlines()
    assert len(lines) == 3  # header + 2 turns


def test_turns_csv_includes_prompt_bytes_and_idle_ms():
    text = export.turns_csv(_run_result())
    header = text.strip().splitlines()[0].split(",")
    assert "prompt_bytes" in header
    assert "idle_ms" in header
    assert "request_ms" in header


def test_turns_csv_includes_probe_rtt_summary():
    """Each turn's row should summarise its probe RTTs (count + mean), so a
    reader does not have to cross-reference a separate JSON blob."""
    text = export.turns_csv(_run_result())
    header = text.strip().splitlines()[0].split(",")
    assert "probe_count" in header
    assert "probe_rtt_mean_ms" in header
    rows = text.strip().splitlines()[1:]
    row0 = dict(zip(header, rows[0].split(",")))
    assert row0["probe_count"] == "2"


def test_turns_csv_of_empty_run_is_header_only():
    text = export.turns_csv({"turns": [], "probes": []})
    assert len(text.strip().splitlines()) == 1
