"""The probe's job is to tell three failures apart.

An expired token, a project without preview access, and an API that genuinely
refuses `model: gemini-*` all stop the request — but only the last one says
anything about the schema. Confusing them would send the whole experiment down
the wrong path, so the classifier is tested before anything is built on it.
"""

import sys
from pathlib import Path

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


# --- _conclude -------------------------------------------------------------

def _target(name, model=None, verdict="unsupported", sysv="skipped"):
    return {"target": name, "verdict": verdict, "supported_model": model,
            "checks": {"system_instruction": {"verdict": sysv}}}


def test_vertex_support_keeps_every_arm_on_vertex():
    c = probe._conclude([_target("vertex-global", "gemini-3-flash-preview", "supported"),
                         _target("devapi", "gemini-3-flash-preview", "supported")])
    assert c["next_step"] == "compare_on_vertex"
    assert c["host"] == "vertex-global"


def test_devapi_only_support_moves_the_comparison():
    c = probe._conclude([_target("vertex-global"),
                         _target("devapi", "gemini-3-flash-preview", "supported",
                                 sysv="per_turn")])
    assert c["next_step"] == "compare_on_devapi"
    assert c["system_instruction"] == "per_turn"


def test_no_host_supports_a_model_interaction():
    c = probe._conclude([_target("vertex-global"), _target("devapi")])
    assert c["next_step"] == "no_comparison_possible"


def test_all_targets_skipped_is_never_a_schema_conclusion():
    # No project, no API key: nothing was measured. Calling that "unsupported"
    # would turn a missing credential into a claim about the API.
    c = probe._conclude([_target("vertex-global", verdict="skipped"),
                         _target("devapi", verdict="skipped")])
    assert c["next_step"] == "fix_environment"


def test_a_skipped_host_is_named_when_the_others_say_unsupported():
    c = probe._conclude([_target("vertex-global"), _target("devapi", verdict="skipped")])
    assert c["next_step"] == "no_comparison_possible"
    assert c["unprobed"] == ["devapi"]


def test_environment_failure_never_reports_a_schema_conclusion():
    # A 401 everywhere must not be read as "gemini models are unsupported".
    c = probe._conclude([_target("vertex-global", verdict="environment"),
                         _target("devapi", verdict="environment")])
    assert c["next_step"] == "fix_environment"
    assert "vertex-global" in c["blocked"]


# --- mock ------------------------------------------------------------------

def test_mock_runs_without_credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = probe.probe_interactions()
    assert out["mock"] is True
    assert {t["target"] for t in out["targets"]} == {"vertex-global", "devapi"}
    assert out["conclusion"]["next_step"] == "compare_on_devapi"
