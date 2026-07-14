"""The probe's job is to tell three failures apart.

An expired token, a project without preview access, and an API that genuinely
refuses `model: gemini-*` all stop the request — but only the last one says
anything about the schema. Confusing them would send the whole experiment down
the wrong path, so the classifier is tested before anything is built on it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe


# --- classify --------------------------------------------------------------

def test_200_is_supported():
    assert probe.classify(200, "") == "supported"
    assert probe.classify(201, "") == "supported"


def test_400_invalid_argument_is_the_schema_signal():
    body = '{"error":{"code":400,"status":"INVALID_ARGUMENT",' \
           '"message":"Invalid value at \'model\'"}}'
    assert probe.classify(400, body) == "unsupported"


def test_404_means_the_resource_path_does_not_exist():
    # A regional interactions path that isn't served.
    assert probe.classify(404, "") == "unsupported"


def test_401_is_environment_not_schema():
    assert probe.classify(401, "UNAUTHENTICATED") == "environment"


def test_403_permission_denied_is_environment():
    body = '{"error":{"status":"PERMISSION_DENIED",' \
           '"message":"caller does not have permission"}}'
    assert probe.classify(403, body) == "environment"


def test_403_service_disabled_is_environment():
    body = "Vertex AI API has not been used in project 123 before or it is disabled"
    assert probe.classify(403, body) == "environment"


def test_403_allowlist_is_unavailable_not_environment():
    # The project is fine; it just isn't admitted to the preview. That is a
    # finding about the API, not a misconfiguration to go fix.
    body = '{"error":{"message":"Your project is not allowed to access this agent"}}'
    assert probe.classify(403, body) == "unavailable"


def test_400_with_bad_api_key_is_environment_not_schema():
    # The Developer API reports a bad key as 400, which would otherwise read as
    # "this field is unsupported".
    assert probe.classify(400, '{"error":{"message":"API key not valid"}}') == "environment"


def test_429_is_environment():
    assert probe.classify(429, "RESOURCE_EXHAUSTED") == "environment"


def test_500_is_error():
    assert probe.classify(500, "internal") == "error"


# --- in-stream errors ------------------------------------------------------
# A streamed interaction sends 200 headers before it knows whether it will
# succeed, then reports failure as an `error` event in the body. Judging on the
# HTTP status alone scored a depleted-billing failure as "supported" -- i.e. as
# evidence that the API runs a model it never ran.

_DEPLETED = {"event_type": "error",
             "error": {"code": "too_many_requests",
                       "message": "Your prepayment credits are depleted."}}


def test_error_event_is_found_in_the_stream():
    events = [{"event_type": "interaction.created"},
              {"event_type": "interaction.status_update"}, _DEPLETED]
    assert probe._stream_error(events)["code"] == "too_many_requests"


def test_a_clean_stream_has_no_error():
    assert probe._stream_error([{"event_type": "interaction.completed"}]) == {}


def test_depleted_credits_is_environment_not_supported():
    assert probe.classify_stream_error(_DEPLETED["error"]) == "environment"


def test_in_stream_invalid_argument_is_the_schema_signal():
    assert probe.classify_stream_error(
        {"code": "invalid_argument", "message": "Invalid value at 'model'"}) == "unsupported"


def test_in_stream_allowlist_message_is_unavailable():
    assert probe.classify_stream_error(
        {"code": "permission_denied",
         "message": "not allowed to access this agent"}) == "unavailable"


def test_usage_is_read_from_metadata_total_usage_too():
    # GEAP puts token counts on interaction.usage; the Developer API's streaming
    # docs describe metadata.total_usage. Read either.
    events = [{"event_type": "interaction.completed",
               "metadata": {"total_usage": {"total_tokens": 42}}}]
    assert probe._usage_from_events(events)["total_tokens"] == 42


def test_interaction_usage_still_wins_when_present():
    events = [{"interaction": {"usage": {"total_tokens": 7}}}]
    assert probe._usage_from_events(events)["total_tokens"] == 7


# --- _blank_id -------------------------------------------------------------

def test_identical_rejections_compare_equal_once_ids_are_blanked():
    a = 'Invalid value at "model": gemini-3-flash-preview'
    b = 'Invalid value at "model": definitely-not-a-real-model'
    assert probe._blank_id(a, "gemini-3-flash-preview") == \
           probe._blank_id(b, probe.BOGUS_MODEL)


def test_different_rejections_do_not_compare_equal():
    a = 'Model gemini-3-flash-preview is not enabled for interactions'
    b = 'Invalid value at "model": definitely-not-a-real-model'
    assert probe._blank_id(a, "gemini-3-flash-preview") != \
           probe._blank_id(b, probe.BOGUS_MODEL)


# --- system_instruction persistence ----------------------------------------
# The rule must be conditional and untriggered on turn 1. A model that merely
# copies the format of its own previous answer would otherwise be scored as
# "the server kept the system prompt" -- which is how the first version of this
# probe reported `persisted` for an API whose docs, and whose actual behaviour,
# say the opposite.

def _fake_calls(monkeypatch, texts):
    """Feed _call a scripted list of response texts, in order."""
    seq = list(texts)

    def fake(url, auth, body):
        t = seq.pop(0)
        return {"verdict": "supported", "status": 200, "text": t,
                "interaction_id": "i1", "usage": {}, "error": ""}

    monkeypatch.setattr(probe, "_call", fake)


def _sys(monkeypatch, texts):
    _fake_calls(monkeypatch, texts)
    return probe._probe_system_instruction("u", "apikey", "m")


def test_marker_on_turn_two_without_the_instruction_means_persisted(monkeypatch):
    # control fires, turn 1 is clean, turn 2 fires anyway -> nothing in the
    # history could have taught it the rule.
    r = _sys(monkeypatch, [probe.MARKER, "Hello!", probe.MARKER])
    assert r["verdict"] == "persisted"


def test_no_marker_on_turn_two_means_per_turn(monkeypatch):
    r = _sys(monkeypatch, [probe.MARKER, "Hello!", "BANANA! how fun"])
    assert r["verdict"] == "per_turn"


def test_a_model_that_ignores_the_rule_yields_no_verdict(monkeypatch):
    # Control didn't emit the marker: the rule was never obeyed, so turn 2's
    # silence would prove nothing.
    r = _sys(monkeypatch, ["I am a banana"])
    assert r["verdict"] == "inconclusive"


def test_a_marker_leaked_on_turn_one_yields_no_verdict(monkeypatch):
    # Turn 1 put the marker into the history, so turn 2 could just imitate it.
    r = _sys(monkeypatch, [probe.MARKER, f"Hello! {probe.MARKER}", probe.MARKER])
    assert r["verdict"] == "inconclusive"
    assert "imitate" in r["reason"]


# --- _target_verdict -------------------------------------------------------

def _models(stream, nonstream):
    return {"gemini-2.5-flash": {"interactions_stream": {"verdict": stream},
                                 "interactions_nonstream": {"verdict": nonstream}}}


def test_a_target_whose_calls_all_failed_on_billing_is_not_unsupported():
    # Every call died on depleted credits. The API never judged the body.
    assert probe._target_verdict(_models("environment", "environment"), None) == "environment"


def test_a_genuine_refusal_outranks_an_environment_failure():
    assert probe._target_verdict(_models("unsupported", "environment"), None) == "unsupported"


def test_allowlist_beats_environment_but_not_a_refusal():
    assert probe._target_verdict(_models("unavailable", "environment"), None) == "unavailable"


def test_any_success_makes_the_target_supported():
    assert probe._target_verdict(_models("supported", "environment"),
                                 "gemini-2.5-flash") == "supported"


# --- _conclude -------------------------------------------------------------

def _target(name, model=None, verdict="unsupported", sysv="skipped"):
    return {"target": name, "verdict": verdict, "supported_model": model,
            "checks": {"system_instruction": {"verdict": sysv}}}


def test_devapi_support_settles_the_comparison():
    c = probe._conclude([_target("devapi", "gemini-3-flash-preview", "supported",
                                 sysv="per_turn")])
    assert c["next_step"] == "compare_on_devapi"
    assert c["host"] == "devapi"
    assert c["system_instruction"] == "per_turn"


def test_no_model_interaction_is_the_finding():
    c = probe._conclude([_target("devapi")])
    assert c["next_step"] == "no_comparison_possible"


def test_a_skipped_host_is_never_a_schema_conclusion():
    # No API key: nothing was measured. Calling that "unsupported" would turn a
    # missing credential into a claim about the API.
    c = probe._conclude([_target("devapi", verdict="skipped")])
    assert c["next_step"] == "fix_environment"


def test_environment_failure_never_reports_a_schema_conclusion():
    # A 401 must not be read as "gemini models are unsupported".
    c = probe._conclude([_target("devapi", verdict="environment")])
    assert c["next_step"] == "fix_environment"
    assert "devapi" in c["blocked"]


# --- one host, one auth ----------------------------------------------------

def test_the_only_target_is_the_developer_api():
    """Vertex was probed here and never served a plain-model interaction. Every arm
    runs on one host now -- otherwise the latency numbers compare network paths."""
    assert [t["name"] for t in probe._targets()] == ["devapi"]
    assert all("generativelanguage" in t["url"] for t in probe._targets())


def test_there_is_no_adc_auth_left():
    assert probe._headers("apikey")["x-goog-api-key"] is not None
    with pytest.raises(ValueError):
        probe._headers("adc")


# --- mock ------------------------------------------------------------------

def test_mock_runs_without_credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = probe.probe_interactions()
    assert out["mock"] is True
    assert {t["target"] for t in out["targets"]} == {"devapi"}
    assert out["conclusion"]["next_step"] == "compare_on_devapi"
