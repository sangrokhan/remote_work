"""app: request-param validation for the web frontend's /api/run.

fastapi/TestClient tests are skipped when fastapi is not installed, so this
module's pure logic (build_run_params) is tested separately and always runs.
"""

import pytest

from tcp_congestion import app as app_module


def test_build_run_params_applies_defaults():
    params = app_module.build_run_params({})
    assert params["num_turns"] == 20
    assert params["host"] == "server"
    assert params["port"] == 8888
    assert params["system_prompt_bytes"] == 20000
    assert params["turn_user_msg_bytes"] == 1000
    assert params["mock_response_bytes"] == 1000
    assert params["inference_delay_ms"] == 1000
    assert params["idle_duration_ms"] == 0
    assert params["ping_interval_ms"] == 1


def test_build_run_params_overrides_from_payload():
    params = app_module.build_run_params({
        "host": "127.0.0.1", "port": 9999, "num_turns": 10,
        "system_prompt_bytes": 3000, "turn_user_msg_bytes": 1000, "mock_response_bytes": 800,
        "inference_delay_ms": 100, "idle_duration_ms": 500,
        "ping_interval_ms": 25, "label": "test-run",
    })
    assert params["host"] == "127.0.0.1"
    assert params["port"] == 9999
    assert params["num_turns"] == 10
    assert params["label"] == "test-run"


def test_build_run_params_rejects_non_integer_num_turns():
    with pytest.raises(ValueError):
        app_module.build_run_params({"num_turns": "banana"})


def test_build_run_params_rejects_zero_num_turns():
    with pytest.raises(ValueError):
        app_module.build_run_params({"num_turns": 0})


def test_build_run_params_allows_zero_inference_delay():
    params = app_module.build_run_params({"inference_delay_ms": 0})
    assert params["inference_delay_ms"] == 0


def test_build_run_params_rejects_negative_idle_duration():
    with pytest.raises(ValueError):
        app_module.build_run_params({"idle_duration_ms": -10})


def test_build_run_params_defaults_capture_to_false():
    params = app_module.build_run_params({})
    assert params["capture"] is False


def test_build_run_params_reads_capture_true():
    params = app_module.build_run_params({"capture": True})
    assert params["capture"] is True


def test_build_run_params_algorithm_defaults_to_none():
    params = app_module.build_run_params({})
    assert params["algorithm"] is None


def test_build_run_params_normalizes_algorithm_case_and_whitespace():
    params = app_module.build_run_params({"algorithm": "  BBR  "})
    assert params["algorithm"] == "bbr"


def test_build_run_params_empty_string_algorithm_is_none():
    params = app_module.build_run_params({"algorithm": ""})
    assert params["algorithm"] is None


def test_build_run_params_enable_ping_probes_defaults_to_true():
    params = app_module.build_run_params({})
    assert params["enable_ping_probes"] is True


def test_build_run_params_reads_enable_ping_probes_false():
    params = app_module.build_run_params({"enable_ping_probes": False})
    assert params["enable_ping_probes"] is False


def test_build_run_params_ping_interval_zero_rejected_when_probes_enabled():
    """Checkbox ON (the default): interval 0 would busy-loop probe.run_probes
    with a zero-second sleep, so it stays invalid."""
    with pytest.raises(ValueError):
        app_module.build_run_params({"ping_interval_ms": 0})


def test_build_run_params_ping_interval_zero_allowed_when_probes_disabled():
    """Checkbox OFF: conversation.run() never reads ping_interval_ms in this
    case (no probe thread is started at all), so a leftover 0 in the form
    must not block the run."""
    params = app_module.build_run_params(
        {"ping_interval_ms": 0, "enable_ping_probes": False})
    assert params["ping_interval_ms"] == 0
    assert params["enable_ping_probes"] is False


try:
    from fastapi.testclient import TestClient
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

pytestmark_fastapi = pytest.mark.skipif(
    not _HAS_FASTAPI, reason="fastapi not installed")


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_index_page_serves_html():
    client = TestClient(app_module.create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_api_config_reports_cwnd_availability():
    client = TestClient(app_module.create_app())
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "cwnd_available" in data
    assert "cwnd_reason" in data
    assert "capture_available" in data
    assert "capture_reason" in data


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_api_config_reports_congestion_status():
    client = TestClient(app_module.create_app())
    resp = client.get("/api/config")
    assert resp.status_code == 200
    cc = resp.json()["congestion"]
    assert "ready" in cc
    assert "available" in cc
    assert "missing" in cc
    assert "qdisc" in cc
    assert "guidance" in cc


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_api_run_rejects_bad_num_turns():
    client = TestClient(app_module.create_app())
    resp = client.post("/api/run", json={"num_turns": 0})
    assert resp.status_code == 400
    assert "error" in resp.json()


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_api_run_reports_connection_failure_as_502():
    client = TestClient(app_module.create_app())
    resp = client.post("/api/run", json={
        "host": "127.0.0.1", "port": 1,  # nothing listens here
        "num_turns": 1, "inference_delay_ms": 0, "idle_duration_ms": 10,
    })
    assert resp.status_code == 502
    assert "error" in resp.json()


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_download_cwnd_csv_without_a_run_yet_is_404():
    client = TestClient(app_module.create_app())
    resp = client.get("/api/download/cwnd.csv")
    assert resp.status_code == 404


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_download_turns_csv_without_a_run_yet_is_404():
    client = TestClient(app_module.create_app())
    resp = client.get("/api/download/turns.csv")
    assert resp.status_code == 404


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_download_pcap_without_capture_is_404():
    client = TestClient(app_module.create_app())
    resp = client.get("/api/download/pcap")
    assert resp.status_code == 404


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_after_a_run_cwnd_csv_download_returns_csv(monkeypatch):
    import tcp_congestion.conversation as conv

    fake_result = {
        "label": "conversation", "host": "h", "port": 1, "samples": [],
        "turns": [], "probes": [], "pcap": None, "idle_resets": 0,
        "peak_cwnd": 0, "final_cwnd": 0, "sample_count": 0, "error": "",
    }
    monkeypatch.setattr(conv, "run", lambda **kw: fake_result)

    client = TestClient(app_module.create_app())
    run_resp = client.post("/api/run", json={"num_turns": 1})
    assert run_resp.status_code == 200

    csv_resp = client.get("/api/download/cwnd.csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]

    turns_resp = client.get("/api/download/turns.csv")
    assert turns_resp.status_code == 200
    assert "text/csv" in turns_resp.headers["content-type"]


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_download_bundle_zip_without_a_run_yet_is_404(monkeypatch):
    # _LAST_RESULT is process-global; other tests in this module populate it
    # via /api/run, so force the "no run yet" state this test is about.
    monkeypatch.setattr(app_module, "_LAST_RESULT", None)
    client = TestClient(app_module.create_app())
    resp = client.get("/api/download/bundle.zip")
    assert resp.status_code == 404


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_download_bundle_zip_contains_algorithm_in_filename(monkeypatch):
    import tcp_congestion.conversation as conv

    fake_result = {
        "label": "conversation", "host": "h", "port": 1, "samples": [],
        "turns": [], "probes": [], "pcap": None, "idle_resets": 0,
        "peak_cwnd": 0, "final_cwnd": 0, "sample_count": 0, "error": "",
        "algorithm_requested": "bbr", "algorithm": "bbr", "algorithm_error": "",
    }
    monkeypatch.setattr(conv, "run", lambda **kw: fake_result)

    client = TestClient(app_module.create_app())
    run_resp = client.post("/api/run", json={"num_turns": 1, "algorithm": "bbr"})
    assert run_resp.status_code == 200

    zip_resp = client.get("/api/download/bundle.zip")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"
    disposition = zip_resp.headers["content-disposition"]
    assert "bbr" in disposition


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_download_bundle_zip_contains_csv_entries(monkeypatch):
    import zipfile
    import io as io_mod
    import tcp_congestion.conversation as conv

    fake_result = {
        "label": "conversation", "host": "h", "port": 1,
        "samples": [{"t_ms": 0.0, "local": "1.2.3.4:1", "snd_cwnd": 10,
                     "ca_state": "open"}],
        "turns": [{"turn": 0, "prompt_bytes": 100, "request_ms": 1.0, "idle_ms": 10}],
        "probes": [{"turn": 0, "samples": []}],
        "pcap": None, "idle_resets": 0, "peak_cwnd": 10, "final_cwnd": 10,
        "sample_count": 1, "error": "",
        "algorithm_requested": "vegas", "algorithm": "vegas", "algorithm_error": "",
    }
    monkeypatch.setattr(conv, "run", lambda **kw: fake_result)

    client = TestClient(app_module.create_app())
    client.post("/api/run", json={"num_turns": 1, "algorithm": "vegas"})

    zip_resp = client.get("/api/download/bundle.zip")
    zf = zipfile.ZipFile(io_mod.BytesIO(zip_resp.content))
    names = zf.namelist()
    assert any("cwnd.csv" in n for n in names)
    assert any("turns.csv" in n for n in names)
    assert all(n.startswith("vegas_") for n in names)
