from python_backend.app.sse_envelope import build_sse_envelope, validate_envelope, REQUIRED_ENVELOPE_FIELDS


def test_build_sse_envelope_defaults():
    envelope = build_sse_envelope(
        {
            "runId": "run-abc",
            "threadId": "thread-1",
            "eventSeq": 1,
            "eventType": "run_started",
        }
    )

    assert isinstance(envelope["eventId"], str)
    assert envelope["runId"] == "run-abc"
    assert envelope["threadId"] == "thread-1"
    assert envelope["eventSeq"] == 1
    assert envelope["eventType"] == "run_started"
    assert envelope["checkpoint"] == {}
    assert envelope["payload"] == {}
    assert isinstance(envelope["issuedAt"], str)


def test_validate_envelope_success_and_required_fields():
    envelope = build_sse_envelope(
        {
            "runId": "run-abc",
            "threadId": "thread-1",
            "eventSeq": 1,
            "eventType": "run_started",
        }
    )

    assert validate_envelope(envelope) is True

    for field in REQUIRED_ENVELOPE_FIELDS:
        assert field in envelope


def test_build_invalid_inputs():
    try:
        build_sse_envelope(
            {
                "runId": "run-abc",
                "threadId": "thread-1",
                "eventSeq": 0,
                "eventType": "run_started",
            }
        )
    except Exception as error:
        assert type(error).__name__ == "TypeError"
    else:
        raise AssertionError("expected TypeError")

    try:
        build_sse_envelope(
            {
                "runId": "run-abc",
                "threadId": "thread-1",
                "eventSeq": 1,
                "eventType": "",
            }
        )
    except Exception as error:
        assert type(error).__name__ == "TypeError"
    else:
        raise AssertionError("expected TypeError")

    try:
        validate_envelope(
            {
                "eventId": "",
                "runId": "run-abc",
                "threadId": "thread-1",
                "eventSeq": 1,
                "eventType": "run_started",
                "payload": {},
                "checkpoint": {},
                "issuedAt": "2026-01-01T00:00:00Z",
            }
        )
    except Exception as error:
        assert type(error).__name__ == "TypeError"
    else:
        raise AssertionError("expected TypeError")
