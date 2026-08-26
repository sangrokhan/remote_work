"""aipt.gateway.profiles -- preset value definitions and env parsing
(DESIGN.md 4.7 B9)."""

import os

import pytest

from aipt.gateway import profiles


def test_preset_names_match_design_dropdown():
    # DESIGN.md 4.7 B11: clean/broadband/3g/satellite/lossy/custom, in that order.
    assert profiles.PRESET_NAMES == ("clean", "broadband", "3g", "satellite", "lossy", "custom")


def test_all_non_custom_presets_are_defined():
    for name in profiles.PRESET_NAMES:
        if name == "custom":
            continue
        assert name in profiles.PRESETS


def test_clean_preset_has_no_impairment():
    p = profiles.PRESETS["clean"]
    assert p.delay_ms == 0
    assert p.jitter_ms == 0
    assert p.loss_pct == 0.0
    assert p.reorder_pct == 0.0


@pytest.mark.parametrize("name", ["broadband", "3g", "satellite", "lossy"])
def test_impaired_presets_have_positive_delay_or_loss(name):
    p = profiles.PRESETS[name]
    assert p.delay_ms > 0 or p.loss_pct > 0 or p.reorder_pct > 0


def test_satellite_has_the_largest_delay():
    # Satellite (GEO) RTT dominates every other preset -- the whole point
    # of the preset is to model that.
    sat = profiles.PRESETS["satellite"].delay_ms
    for name in ("clean", "broadband", "3g", "lossy"):
        assert sat > profiles.PRESETS[name].delay_ms


def test_lossy_has_the_highest_loss_and_reorder():
    lossy = profiles.PRESETS["lossy"]
    for name in ("clean", "broadband", "3g", "satellite"):
        other = profiles.PRESETS[name]
        assert lossy.loss_pct >= other.loss_pct
        assert lossy.reorder_pct >= other.reorder_pct


def test_get_preset_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        profiles.get_preset("nope")


def test_get_preset_custom_raises_keyerror():
    # "custom" has no fixed values -- must go through custom_profile()/resolve().
    with pytest.raises(KeyError):
        profiles.get_preset("custom")


def test_custom_profile_builds_from_values():
    p = profiles.custom_profile(delay_ms=42, jitter_ms=7, loss_pct=1.5, reorder_pct=0.2)
    assert p.name == "custom"
    assert p.delay_ms == 42
    assert p.jitter_ms == 7
    assert p.loss_pct == 1.5
    assert p.reorder_pct == 0.2


def test_custom_profile_clamps_negative_values():
    p = profiles.custom_profile(delay_ms=-5, jitter_ms=-1, loss_pct=-0.5, reorder_pct=-1)
    assert p.delay_ms == 0
    assert p.jitter_ms == 0
    assert p.loss_pct == 0.0
    assert p.reorder_pct == 0.0


def test_resolve_preset_ignores_overrides():
    p = profiles.resolve("3g", delay_ms=999)
    assert p.name == "3g"
    assert p.delay_ms == profiles.PRESETS["3g"].delay_ms


def test_resolve_custom_uses_overrides():
    p = profiles.resolve("custom", delay_ms=10, jitter_ms=2, loss_pct=0.1, reorder_pct=0.0)
    assert p.name == "custom"
    assert p.delay_ms == 10


def test_profile_as_dict_shape():
    p = profiles.PRESETS["broadband"]
    d = p.as_dict()
    assert set(d.keys()) == {"profile", "delay_ms", "jitter_ms", "loss_pct", "reorder_pct"}
    assert d["profile"] == "broadband"


class TestFromEnv:
    def _clear(self, monkeypatch):
        for var in (
            "GATEWAY_PROFILE",
            "GATEWAY_DELAY_MS",
            "GATEWAY_JITTER_MS",
            "GATEWAY_LOSS_PCT",
            "GATEWAY_REORDER_PCT",
            "CLIENT_NETEM_DELAY_MS",
            "SERVER_NETEM_DELAY_MS",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_defaults_to_clean(self, monkeypatch):
        self._clear(monkeypatch)
        p = profiles.from_env()
        assert p.name == "clean"

    def test_gateway_profile_selects_named_preset(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("GATEWAY_PROFILE", "3g")
        p = profiles.from_env()
        assert p.name == "3g"
        assert p.delay_ms == profiles.PRESETS["3g"].delay_ms

    def test_gateway_profile_is_case_insensitive(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("GATEWAY_PROFILE", "SATELLITE")
        p = profiles.from_env()
        assert p.name == "satellite"

    def test_gateway_knobs_build_custom_profile(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("GATEWAY_DELAY_MS", "80")
        monkeypatch.setenv("GATEWAY_JITTER_MS", "10")
        monkeypatch.setenv("GATEWAY_LOSS_PCT", "2.5")
        monkeypatch.setenv("GATEWAY_REORDER_PCT", "1")
        p = profiles.from_env()
        assert p.name == "custom"
        assert p.delay_ms == 80
        assert p.jitter_ms == 10
        assert p.loss_pct == 2.5
        assert p.reorder_pct == 1.0

    def test_deprecated_client_netem_delay_ms_alias(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("CLIENT_NETEM_DELAY_MS", "50")
        p = profiles.from_env()
        assert p.name == "custom"
        assert p.delay_ms == 50

    def test_deprecated_server_netem_delay_ms_alias(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("SERVER_NETEM_DELAY_MS", "60")
        p = profiles.from_env()
        assert p.delay_ms == 60

    def test_canonical_var_wins_over_deprecated_alias(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("GATEWAY_DELAY_MS", "10")
        monkeypatch.setenv("CLIENT_NETEM_DELAY_MS", "999")
        p = profiles.from_env()
        assert p.delay_ms == 10

    def test_gateway_profile_custom_falls_through_to_knobs(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("GATEWAY_PROFILE", "custom")
        monkeypatch.setenv("GATEWAY_DELAY_MS", "33")
        p = profiles.from_env()
        assert p.name == "custom"
        assert p.delay_ms == 33
