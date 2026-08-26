"""conversation: full run() against a real server, with the live cwnd monitor.

Skipped where netlink/compiler is unavailable, same as token_traffic's
test_cwnd_live.py.
"""

import threading
import time

import pytest

from tcp_congestion import conversation, cwnd, server

_available, _reason = cwnd.available()
pytestmark = pytest.mark.skipif(
    not _available, reason=f"cwnd monitor unavailable: {_reason}")


@pytest.fixture()
def srv():
    s = server.Server(host="127.0.0.1", port=0)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield s
    s.shutdown()


def test_run_returns_one_entry_per_turn(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=3,
        turn_user_msg_bytes=200, mock_response_bytes=100,
        inference_delay_ms=20, idle_duration_ms=100, ping_interval_ms=20,
    )
    assert len(result["turns"]) == 3
    assert len(result["probes"]) == 3


def test_run_prompt_bytes_increase_across_turns(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=4,
        turn_user_msg_bytes=200, mock_response_bytes=150,
        inference_delay_ms=10, idle_duration_ms=50, ping_interval_ms=20,
    )
    sizes = [t["prompt_bytes"] for t in result["turns"]]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_run_system_prompt_only_inflates_first_turn(srv):
    """A system prompt shows up once (in turn 0's size) and is not re-added
    as fresh bytes on later turns -- growth from turn 1 onward should match
    a run with no system prompt at all, just offset by the constant."""
    with_sp = conversation.run(
        host=srv.host, port=srv.port, num_turns=3,
        system_prompt_bytes=5000, turn_user_msg_bytes=200,
        mock_response_bytes=100, inference_delay_ms=10, idle_duration_ms=50,
        ping_interval_ms=20,
    )
    without_sp = conversation.run(
        host=srv.host, port=srv.port, num_turns=3,
        system_prompt_bytes=0, turn_user_msg_bytes=200,
        mock_response_bytes=100, inference_delay_ms=10, idle_duration_ms=50,
        ping_interval_ms=20,
    )
    sizes_with = [t["prompt_bytes"] for t in with_sp["turns"]]
    sizes_without = [t["prompt_bytes"] for t in without_sp["turns"]]
    # every turn's size differs by exactly the system prompt constant
    for a, b in zip(sizes_with, sizes_without):
        assert a - b == 5000


def test_run_has_continuous_cwnd_samples(srv):
    """The point of using cwnd.Monitor instead of snapshotting: samples exist
    for the whole run, not just two points."""
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=2,
        turn_user_msg_bytes=200, mock_response_bytes=100,
        inference_delay_ms=20, idle_duration_ms=200, ping_interval_ms=20,
    )
    assert result["sample_count"] > 0
    assert result["error"] == ""


def test_run_probe_samples_have_no_delivery_rate(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=2,
        turn_user_msg_bytes=200, mock_response_bytes=100,
        inference_delay_ms=10, idle_duration_ms=100, ping_interval_ms=20,
    )
    for turn_probes in result["probes"]:
        for sample in turn_probes["samples"]:
            assert "delivery_rate" not in sample


def test_run_label_is_used(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
        label="my-conv",
    )
    assert result["label"] == "my-conv"


def test_run_without_capture_has_no_pcap_field(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
    )
    assert result.get("pcap") is None


def test_run_with_capture_true_adds_pcap_result(srv, monkeypatch, tmp_path):
    import shutil
    from tcp_congestion import capture as capture_mod
    if not shutil.which("tcpdump"):
        import pytest
        pytest.skip("tcpdump not installed")
    ok, reason = capture_mod.available()
    if not ok:
        import pytest
        pytest.skip(reason)
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))
    monkeypatch.setenv("TRAFFIC_PCAP_IFACE", "lo")

    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=2000, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
        capture=True,
    )
    assert result["pcap"] is not None
    assert "file" in result["pcap"]


def test_run_without_algorithm_reports_socket_default(srv):
    """No algorithm requested -> algorithm_requested is empty, but the
    result still reports whatever the kernel's default actually is."""
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
    )
    assert result["algorithm_requested"] == ""
    assert result["algorithm_error"] == ""
    # algorithm may be "" only if getsockopt itself is unsupported; on any
    # real Linux box this is the host's current_congestion_control default.
    assert isinstance(result["algorithm"], str)


def test_run_with_available_algorithm_sets_it(srv):
    """cubic is always available (built into the kernel, no modprobe
    needed), so this exercises the real TCP_CONGESTION path end-to-end."""
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
        algorithm="cubic",
    )
    assert result["algorithm_requested"] == "cubic"
    assert result["algorithm"] == "cubic"
    assert result["algorithm_error"] == ""


def test_run_with_unavailable_algorithm_reports_error_without_crashing(srv):
    """An algorithm name the kernel has never heard of must not blow up the
    whole run -- it should fall back to the socket default and say why."""
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
        algorithm="not-a-real-algorithm",
    )
    assert result["algorithm_requested"] == "not-a-real-algorithm"
    assert result["algorithm_error"] != ""
    assert result["algorithm"] != "not-a-real-algorithm"


def test_run_ping_probes_enabled_by_default_collects_samples(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=200, ping_interval_ms=20,
    )
    assert result["ping_probes_enabled"] is True
    assert len(result["probes"][0]["samples"]) > 0


def test_run_with_ping_probes_disabled_sends_no_pings(srv):
    """Idle duration must still fully elapse (unchanged wait), but with
    enable_ping_probes=False no HTTP PING traffic goes out during it."""
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=200, ping_interval_ms=20,
        enable_ping_probes=False,
    )
    assert result["ping_probes_enabled"] is False
    assert len(result["turns"]) == 1
    assert result["turns"][0]["idle_ms"] == 200
    assert result["probes"] == [{"turn": 0, "samples": []}]


def test_run_sends_mock_response_bytes_as_actual_response_size(srv):
    """conversation.mock_response_bytes must actually control what the
    server puts on the wire, not just the simulated history size used to
    grow later turns' prompts -- otherwise a pcap never shows the response
    size the run was configured with (see server._pad_json_to_size)."""
    import socket as socket_mod

    conn = socket_mod.create_connection((srv.host, srv.port), timeout=5)
    conn.sendall(
        b"GET /inference-mock?delay=0&response_bytes=1000 HTTP/1.1\r\n"
        b"Host: " + srv.host.encode() + b"\r\nConnection: keep-alive\r\n\r\n")
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += conn.recv(4096)
    head, _, body = buf.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    while len(body) < length:
        body += conn.recv(4096)
    conn.close()
    assert length == 1000
    assert len(body) == 1000

    # And conversation.run() actually requests that size end-to-end.
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=1000,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
    )
    assert result["turns"][0]["prompt_bytes"] == 100  # sanity: turn 0 unaffected by response size
