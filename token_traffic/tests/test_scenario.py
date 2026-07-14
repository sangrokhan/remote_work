"""The fixture is the only thing every arm has in common. If it is wrong, every number
in the run is wrong together, and consistently -- which is the hardest kind of wrong to
notice."""

from __future__ import annotations

import pytest

from core import scenario


class TestLoad:
    def test_the_default_fixture_loads_and_is_a_conversation(self):
        f = scenario.load()
        assert f["steps"] and all(isinstance(s, str) and s for s in f["steps"])
        assert isinstance(f["system"], str)

    def test_the_system_prompt_is_big_enough_to_cache(self):
        # The perf fixture exists to make caching engage: Gemini's implicit cache needs
        # a prompt over ~4k tokens, and a smaller one would make the cached arm look
        # identical to the stateless one for reasons that have nothing to do with the
        # arms.
        assert len(scenario.load()["system"]) > 4096

    def test_a_step_object_is_unwrapped_to_its_text(self):
        # The perf fixture writes steps as {"text": ...}. A provider takes list[str],
        # and a dict that survives this function becomes {"text": {"text": ...}} in a
        # live request body: a part the API accepts the shape of and reads nothing from.
        assert scenario._text_of({"text": "hi"}, 0) == "hi"
        assert scenario._text_of("hi", 0) == "hi"

    def test_a_step_of_an_unusable_shape_is_refused_by_index(self):
        with pytest.raises(ValueError, match="step 3"):
            scenario._text_of({"prompt": "hi"}, 3)

    def test_turns_truncates_rather_than_repeats(self):
        full = scenario.load()
        cut = scenario.load(turns=3)
        assert cut["steps"] == full["steps"][:3]

    def test_asking_for_more_turns_than_the_fixture_has_is_refused(self):
        # Silently cycling the questions would answer the second pass from context and
        # cost nothing like a new turn -- an arm would look cheap because it was asked
        # something it had already been told.
        n = len(scenario.load()["steps"])
        with pytest.raises(ValueError, match="Add steps"):
            scenario.load(turns=n + 1)

    def test_unknown_fixture_names_the_ones_that_exist(self):
        with pytest.raises(KeyError, match="perf"):
            scenario.load("nonesuch")
