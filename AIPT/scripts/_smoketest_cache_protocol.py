"""Unit tests for aipt.core.cache_protocol -- run standalone with plain
assert (not pytest) since this is a smoke-test script, matching the repo's
tests/ dir being pytest-based but this being a quick verification pass
during initial implementation (2026-09-01)."""
import copy
import sys

sys.path.insert(0, ".")
from aipt.core import cache_protocol as cp


def test_path_label_roundtrip():
    path = ("messages", 0, "content")
    label = cp.path_to_label(path)
    assert label == '"messages".0."content"', label
    assert cp.parse_label(label) == path


def test_path_label_roundtrip_special_chars():
    path = ("weird key", 3, "a.b")
    label = cp.path_to_label(path)
    assert cp.parse_label(label) == path


def test_encode_first_appearance_unchanged():
    cache = cp.SessionCache()
    long_text = "x" * 300
    body = {"messages": [{"role": "user", "content": long_text}]}
    out = cp.encode_body(body, cache)
    assert out["messages"][0]["content"] == long_text  # unchanged, first time
    assert cp.CACHE_MAP_FIELD not in out
    # but cache learned it
    assert cache.hash_for(long_text) is not None


def test_encode_second_appearance_hashed():
    cache = cp.SessionCache()
    long_text = "y" * 300
    body1 = {"messages": [{"role": "user", "content": long_text}]}
    cp.encode_body(body1, cache)  # learns it

    body2 = {"messages": [
        {"role": "user", "content": long_text},
        {"role": "user", "content": "new short text"},
    ]}
    out = cp.encode_body(body2, cache)
    h = cache.hash_for(long_text)
    assert out["messages"][0]["content"] == h
    assert out["messages"][1]["content"] == "new short text"
    assert cp.CACHE_MAP_FIELD in out
    assert out[cp.CACHE_MAP_FIELD] == {"hashed_0": '"messages".0."content"'}


def test_encode_below_threshold_never_touched():
    cache = cp.SessionCache()
    short_text = "hi"
    body = {"messages": [{"role": "user", "content": short_text}]}
    out1 = cp.encode_body(body, cache)
    out2 = cp.encode_body(copy.deepcopy(body), cache)
    # even sent twice, short text never gets hashed
    assert out1["messages"][0]["content"] == short_text
    assert out2["messages"][0]["content"] == short_text
    assert cp.CACHE_MAP_FIELD not in out2


def test_original_body_never_mutated():
    cache = cp.SessionCache()
    long_text = "z" * 300
    body1 = {"messages": [{"role": "user", "content": long_text}]}
    original_snapshot = copy.deepcopy(body1)
    cp.encode_body(body1, cache)
    out = cp.encode_body(body1, cache)  # second call, same original body dict
    assert body1 == original_snapshot  # caller's dict untouched
    assert out["messages"][0]["content"] != long_text  # but output is hashed


def test_roundtrip_encode_decode_symmetric():
    client_cache = cp.SessionCache()
    server_cache = cp.SessionCache()
    long_text = "w" * 300

    # Turn 1: first appearance
    body1 = {"messages": [{"role": "user", "content": long_text}]}
    wire1 = cp.encode_body(body1, client_cache)
    decoded1 = cp.decode_body(wire1, server_cache)
    assert decoded1["messages"][0]["content"] == long_text
    assert cp.CACHE_MAP_FIELD not in decoded1

    # Turn 2: client re-sends same content among a fresh turn
    body2 = {"messages": [
        {"role": "user", "content": long_text},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "follow up question"},
    ]}
    wire2 = cp.encode_body(body2, client_cache)
    assert wire2["messages"][0]["content"] != long_text  # hashed on wire
    decoded2 = cp.decode_body(wire2, server_cache)
    assert decoded2["messages"][0]["content"] == long_text  # restored
    assert cp.CACHE_MAP_FIELD not in decoded2  # bookkeeping field stripped
    assert decoded2["messages"][2]["content"] == "follow up question"


def test_decode_cache_miss_raises_with_paths():
    server_cache = cp.SessionCache()  # empty -- server "forgot" everything
    wire = {
        "messages": [{"role": "user", "content": "deadbeefdeadbeefdead"}],
        cp.CACHE_MAP_FIELD: {"hashed_0": '"messages".0."content"'},
    }
    try:
        cp.decode_body(wire, server_cache)
        assert False, "expected CacheMiss"
    except cp.CacheMiss as exc:
        assert exc.missing_paths == ['"messages".0."content"']


def test_decode_never_partially_mutates_on_miss():
    server_cache = cp.SessionCache()
    wire = {
        "messages": [{"role": "user", "content": "aaaaaaaaaaaaaaaaaaaa"}],
        cp.CACHE_MAP_FIELD: {"hashed_0": '"messages".0."content"'},
    }
    wire_snapshot = copy.deepcopy(wire)
    try:
        cp.decode_body(wire, server_cache)
    except cp.CacheMiss:
        pass
    assert wire == wire_snapshot  # untouched


def test_cache_map_field_itself_never_walked_as_leaf():
    cache = cp.SessionCache()
    body = {
        "messages": [{"role": "user", "content": "x" * 300}],
        cp.CACHE_MAP_FIELD: {"hashed_0": "should not be walked"},
    }
    # Should not raise / should not try to hash the cache map's own values
    out = cp.encode_body(body, cache)
    assert isinstance(out, dict)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
