"""What summarize() must never do: mix two providers' arms, fold prep into a total,
hide a failed call, or lose which measure mode produced the bytes."""

import pytest

from core.metrics import key_of, summarize


def rec(provider, arm, turn, *, phase="steady", measure="both", sent=1000,
        recv=500, tokens=(100, 0, 20, 0), marks=(10, 100, 200, 900, 2700),
        error=""):
    req_sent_ms, ttfb, ttft, ttlt, turn_end = marks
    input_t, cached_t, output_t, reasoning_t = tokens
    return {
        "schema_version": 1,
        "provider": provider, "arm": arm, "phase": phase, "turn": turn,
        "measure": measure,
        "wire_sent": sent, "wire_recv": recv,
        "req_sent_ms": req_sent_ms, "ttfb_ms": ttfb, "ttft_ms": ttft,
        "ttlt_ms": ttlt, "turn_end_ms": turn_end,
        "store_tail_ms": turn_end - ttlt,
        "elapsed_ms": turn_end,
        "input_tokens": input_t, "cached_tokens": cached_t,
        "output_tokens": output_t, "reasoning_tokens": reasoning_t,
        "error": error,
    }


def run_of(records, measure="both", **params):
    return {"params": {"measure": measure, **params}, "records": records}


def test_series_are_keyed_by_provider_and_arm():
    # Both providers ship an arm the operator calls "stateless". They are different
    # machinery and must never land in the same bucket.
    s = summarize(run_of([
        rec("gemini", "stateless", 1),
        rec("openai", "chat_stateless", 1),
    ]))
    assert s["keys"] == ["gemini:stateless", "openai:chat_stateless"]
    assert set(s["series"]) == set(s["totals"]) == set(s["keys"])


def test_the_same_arm_name_under_two_providers_does_not_collide():
    s = summarize(run_of([
        rec("gemini", "stateless", 1, sent=1000),
        rec("openai", "stateless", 1, sent=7000),
    ]))
    assert s["totals"]["gemini:stateless"]["wire_sent"] == 1000
    assert s["totals"]["openai:stateless"]["wire_sent"] == 7000


def test_provider_and_arm_are_carried_as_fields_so_a_ui_can_group_either_way():
    s = summarize(run_of([rec("gemini", "cached", 1)]))
    k = key_of("gemini", "cached")
    for section in ("series", "totals"):
        assert s[section][k]["provider"] == "gemini"
        assert s[section][k]["arm"] == "cached"


def test_per_turn_and_cumulative_series():
    s = summarize(run_of([
        rec("gemini", "stateless", 1, sent=1000, recv=500),
        rec("gemini", "stateless", 2, sent=2000, recv=500),
        rec("gemini", "stateless", 3, sent=3000, recv=500),
    ]))["series"]["gemini:stateless"]
    assert s["turns"] == [1, 2, 3]
    assert s["per_turn_wire_sent"] == [1000, 2000, 3000]
    assert s["cum_wire_sent"] == [1000, 3000, 6000]
    assert s["per_turn_wire"] == [1500, 2500, 3500]
    assert s["cum_wire"] == [1500, 4000, 7500]
    assert s["cum_wire_recv"] == [500, 1000, 1500]


def test_cumulative_counts_stay_integers():
    # A tooltip reading "1500.0 bytes" claims a precision the measurement does not
    # have.
    s = summarize(run_of([rec("gemini", "stateless", 1)]))["series"]["gemini:stateless"]
    assert all(isinstance(v, int) for v in s["cum_wire"] + s["cum_input_tokens"])


def test_records_arrive_out_of_order_and_the_series_is_still_in_turn_order():
    s = summarize(run_of([
        rec("gemini", "stateless", 3, sent=3000),
        rec("gemini", "stateless", 1, sent=1000),
        rec("gemini", "stateless", 2, sent=2000),
    ]))["series"]["gemini:stateless"]
    assert s["turns"] == [1, 2, 3]
    assert s["per_turn_wire_sent"] == [1000, 2000, 3000]


def test_mark_stats_for_all_five_marks_and_the_store_tail():
    s = summarize(run_of([
        rec("gemini", "interaction", 1, marks=(10, 100, 200, 900, 2700)),
        rec("gemini", "interaction", 2, marks=(20, 200, 400, 1100, 2900)),
    ]))["totals"]["gemini:interaction"]["marks"]
    for m in ("req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms", "turn_end_ms",
              "store_tail_ms"):
        assert set(s[m]) >= {"mean", "median", "min", "max"}
    assert s["ttft_ms"] == {"mean": 300.0, "median": 300.0, "min": 200,
                            "max": 400, "n": 2}
    # The tail a stored interaction pays after the last token: 1800 both turns.
    assert s["store_tail_ms"]["mean"] == 1800.0
    assert s["store_tail_ms"]["max"] == 1800


def test_store_tail_is_derived_when_the_record_does_not_carry_it():
    r = rec("gemini", "interaction", 1, marks=(10, 100, 200, 900, 2700))
    del r["store_tail_ms"]
    s = summarize(run_of([r]))["totals"]["gemini:interaction"]["marks"]
    assert s["store_tail_ms"]["mean"] == 1800


def test_token_totals():
    s = summarize(run_of([
        rec("openai", "responses_stateful", 1, tokens=(100, 40, 20, 5)),
        rec("openai", "responses_stateful", 2, tokens=(200, 80, 30, 5)),
    ]))["totals"]["openai:responses_stateful"]
    assert s["input_tokens"] == 300
    assert s["cached_tokens"] == 120
    assert s["output_tokens"] == 50
    assert s["reasoning_tokens"] == 10


# --- prep is setup, not traffic ------------------------------------------------

def test_prep_phases_are_kept_out_of_the_totals():
    # A cache build re-sends the whole system prompt. Folded into the total it would
    # drown every measured turn -- 90 kB of setup against 3 kB of traffic.
    s = summarize(run_of([
        rec("gemini", "cached", 0, phase="cachegen", sent=90000, recv=200,
            tokens=(9000, 0, 0, 0)),
        rec("gemini", "cached", 1, sent=1000, recv=500, tokens=(100, 0, 20, 0)),
        rec("gemini", "cached", 2, sent=1000, recv=500, tokens=(100, 0, 20, 0)),
    ]))
    t = s["totals"]["gemini:cached"]
    assert t["wire_sent"] == 2000
    assert t["input_tokens"] == 200
    assert t["turns"] == 2
    assert s["series"]["gemini:cached"]["turns"] == [1, 2]


def test_prep_is_reported_separately_rather_than_hidden():
    s = summarize(run_of([
        rec("gemini", "cached", 0, phase="cachegen", sent=90000, recv=200,
            tokens=(9000, 0, 0, 0)),
        rec("gemini", "cached", 1),
    ]))
    p = s["prep"]["gemini:cached"]
    assert p["phases"] == ["cachegen"]
    assert p["calls"] == 1
    assert p["wire_sent"] == 90000
    assert p["wire"] == 90200
    assert p["input_tokens"] == 9000
    assert p["provider"] == "gemini" and p["arm"] == "cached"


def test_any_phase_that_is_not_steady_counts_as_prep():
    # OpenAI's conversation create is called "setup", not "cachegen". The rule is the
    # phase name being anything other than steady, not a list of known prep names.
    s = summarize(run_of([
        rec("openai", "responses_stateful", 0, phase="setup", sent=21000),
        rec("openai", "responses_stateful", 1, sent=900),
    ]))
    assert s["totals"]["openai:responses_stateful"]["wire_sent"] == 900
    assert s["prep"]["openai:responses_stateful"]["phases"] == ["setup"]


def test_an_arm_with_no_prep_has_no_prep_entry():
    s = summarize(run_of([rec("gemini", "stateless", 1)]))
    assert "gemini:stateless" not in s["prep"]


# --- failures ------------------------------------------------------------------

def test_failures_are_named_by_provider_arm_and_turn():
    # A run with a broken arm still produces plausible-looking numbers. The failing
    # calls have to be nameable, or a zero reads as a measurement.
    s = summarize(run_of([
        rec("gemini", "cached", 1),
        rec("gemini", "cached", 2, sent=0, recv=0, tokens=(0, 0, 0, 0),
            error="429 RESOURCE_EXHAUSTED"),
        rec("openai", "chat_stateless", 1),
    ]))
    assert s["failures"] == [{
        "provider": "gemini", "arm": "cached", "key": "gemini:cached",
        "phase": "steady", "turn": 2, "error": "429 RESOURCE_EXHAUSTED",
    }]
    assert s["totals"]["gemini:cached"]["errors"] == 1
    assert s["totals"]["openai:chat_stateless"]["errors"] == 0


def test_a_failed_prep_call_is_a_failure_too():
    s = summarize(run_of([
        rec("gemini", "cached", 0, phase="cachegen", error="cache create failed"),
        rec("gemini", "cached", 1),
    ]))
    assert [f["phase"] for f in s["failures"]] == ["cachegen"]


def test_a_clean_run_names_nothing():
    assert summarize(run_of([rec("gemini", "stateless", 1)]))["failures"] == []


# --- measure mode --------------------------------------------------------------

@pytest.mark.parametrize("measure", ["bytes", "latency", "both"])
def test_the_measure_mode_is_visible_in_the_summary(measure):
    # Bytes off a streamed pass and bytes off a blocking pass are different
    # measurements. A summary that does not say which cannot be read.
    s = summarize(run_of([rec("gemini", "stateless", 1, measure=measure)],
                         measure=measure))
    assert s["measure"] == measure
    assert s["series"]["gemini:stateless"]["measure"] == measure
    assert s["totals"]["gemini:stateless"]["measure"] == measure


def test_no_cost_estimate_anywhere():
    # A dollar figure built on a guessed per-token rate is not evidence.
    s = summarize(run_of([rec("gemini", "stateless", 1)]))
    flat = repr(s)
    assert "cost" not in flat and "usd" not in flat.lower()


def test_an_empty_run_summarizes_to_nothing_rather_than_raising():
    s = summarize(run_of([]))
    assert s == {"measure": "both", "keys": [], "series": {}, "totals": {},
                 "prep": {}, "failures": []}


def test_call_ms_comes_off_turn_end_which_is_what_a_record_actually_carries():
    # core.record has no elapsed_ms field; turn_end_ms is the call's duration. A
    # clock that read the missing field would report a confident zero.
    r = rec("gemini", "stateless", 1, marks=(10, 100, 200, 900, 2700))
    del r["elapsed_ms"]
    assert summarize(run_of([r]))["totals"]["gemini:stateless"]["call_ms"] == 2700


def test_wall_ms_is_read_per_key():
    s = summarize({"params": {"measure": "bytes"},
                   "wall_ms": {"gemini:stateless": 4200},
                   "records": [rec("gemini", "stateless", 1)]})
    assert s["totals"]["gemini:stateless"]["wall_ms"] == 4200
