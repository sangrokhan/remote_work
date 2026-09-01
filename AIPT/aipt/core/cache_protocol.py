"""aipt.core.cache_protocol -- request-body leaf-hash dedup protocol shared
by the client side (``aipt/backends/local_llm/gateway.py``, runs inside
``web``) and the server side (``docker/engine_gateway.py``, runs inside the
``local-llm`` container). See ``docs/engine_gateway_caching_seed.md`` for
the full design (motivation, wire format, opt-in header, session model).

Deliberately stdlib-only (``hashlib``/``json``-adjacent, no ``aipt.core``
siblings that pull in ``requests`` etc.) so it can be copied into the
``local-llm`` image's minimal ``aipt`` slice the same way
``aipt/core/idle_reset.py`` already is (see docker/Dockerfile.local_llm).

Not a caching layer for LLM *responses* -- see this module's sibling
concept, ``aipt.backends.local_llm.gateway.Gateway``'s ``on_request``/
``on_response`` hooks (currently no-op, a different and orthogonal design
originally mis-scoped as "cache the answer" before the 2026-09-01 Slack
design discussion corrected it to "dedup what the client re-sends").  This
module is the *request* dedup layer: never touches response bytes.

Wire format (see the Seed doc's "5. 와이어 포맷" section for worked
examples):

  * Opt-in header: ``X-AIPT-Cache: enable`` on both the request and (when
    the header was present on the request) reflected in error responses.
  * Only STRING leaf values whose UTF-8 byte length is >= a threshold
    (default 200 bytes) are dedup candidates -- short leaves (``role``,
    etc.) are left alone even when this feature is on.
  * A leaf's path is a tuple of string keys / int indices, e.g.
    ``("messages", 0, "content")``, rendered as the label
    ``"messages".0."content"`` (string keys quoted, int indices bare) for
    the ``$aipt_cache_map`` field.
  * First time a session sees a given leaf VALUE, it is sent verbatim and
    both sides independently learn ``hash(value) -> value`` for this
    session -- no round trip needed to agree on that.
  * Every later occurrence of an already-known value is replaced in place
    by its hash (the leaf's JSON value becomes the hash string) and its
    path is listed under ``$aipt_cache_map`` (map key is a human-readable
    ``hashed_N`` label; map value is the path label -- the substituted hash
    itself is already sitting at that path in the body, so the map only
    needs to say "go look here").
  * Session = the underlying keep-alive TCP connection. No explicit
    session id, no TTL: a fresh connection starts with an empty cache on
    both sides.
  * Cache miss on the server (a path in ``$aipt_cache_map`` whose hash the
    server's session store does not know -- e.g. a reused connection whose
    server-side process restarted) is reported as HTTP 409 with
    ``{"error": "cache_miss", "missing_paths": [...]}"``; the client is
    expected to resend with just those paths' original values restored.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

#: Opt-in header both sides must send/recognise for this protocol to run
#: at all. Its absence means "behave exactly as if this module didn't
#: exist" on both client and server.
CACHE_HEADER = "X-AIPT-Cache"
CACHE_HEADER_VALUE = "enable"

#: Field added at the top level of the request body listing which leaf
#: paths currently hold a hash rather than their real value.
CACHE_MAP_FIELD = "$aipt_cache_map"

#: sha256 hexdigest truncated to 20 chars (80 bits) -- see
#: docs/engine_gateway_caching_seed.md section 8.1 for the collision-odds
#: calculation (~4e-19 at 1,000 cached leaves/session) behind this choice.
HASH_LEN = 20

#: Leaf values shorter than this (UTF-8 byte length) are never dedup
#: candidates, matching short structural fields like "role" out for free
#: without an explicit field whitelist.
DEFAULT_THRESHOLD_BYTES = 200


def compute_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:HASH_LEN]


# --------------------------------------------------------------------------
# JSON path <-> label
# --------------------------------------------------------------------------

def path_to_label(path: tuple) -> str:
    """``("messages", 0, "content")`` -> ``'"messages".0."content"'``."""
    parts = []
    for elem in path:
        if isinstance(elem, int):
            parts.append(str(elem))
        else:
            parts.append('"' + str(elem).replace('"', '\\"') + '"')
    return ".".join(parts)


_LABEL_TOKEN_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|(\d+)')


def parse_label(label: str) -> tuple:
    """Inverse of :func:`path_to_label`. Raises ValueError on malformed
    input (a label that doesn't fully tokenize) rather than silently
    returning a partial path -- a wrong path here means writing to the
    wrong place in someone's request body."""
    path = []
    pos = 0
    for m in _LABEL_TOKEN_RE.finditer(label):
        if m.start() != pos:
            raise ValueError(f"malformed cache-map path label: {label!r}")
        if m.group(1) is not None:
            path.append(m.group(1).replace('\\"', '"'))
        else:
            path.append(int(m.group(2)))
        pos = m.end()
        if pos < len(label) and label[pos] == ".":
            pos += 1
    if pos != len(label):
        raise ValueError(f"malformed cache-map path label: {label!r}")
    return tuple(path)


def get_at_path(root: Any, path: tuple) -> Any:
    node = root
    for elem in path:
        if isinstance(elem, int):
            node = node[elem]
        else:
            node = node[elem]
    return node


def set_at_path(root: Any, path: tuple, value: Any) -> None:
    node = root
    for elem in path[:-1]:
        node = node[elem]
    node[path[-1]] = value


def iter_string_leaves(obj: Any, path: tuple = ()):
    """Yield (path, value) for every STRING leaf reachable by walking dicts
    (by key) and lists (by index). Non-string leaves (numbers, bools,
    None) are never dedup candidates -- see module docstring."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == CACHE_MAP_FIELD:
                continue  # never recurse into our own bookkeeping field
            yield from iter_string_leaves(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_string_leaves(v, path + (i,))
    elif isinstance(obj, str):
        yield path, obj


# --------------------------------------------------------------------------
# Session-scoped cache store
# --------------------------------------------------------------------------

class SessionCache:
    """One side's (client's or server's) view of "what has this session
    already seen", keyed by hash. Deliberately dumb (plain dicts, no TTL,
    no eviction) -- lifetime is the TCP connection's, matching the design
    doc's "session = keep-alive connection, no explicit id, no TTL"
    decision. Not thread-safe; callers own one instance per connection.
    """

    def __init__(self) -> None:
        self._hash_to_value: dict[str, str] = {}
        self._value_to_hash: dict[str, str] = {}

    def hash_for(self, value: str) -> str | None:
        """Existing hash for a value this session has already recorded,
        or None if this is the first time this exact value is seen."""
        return self._value_to_hash.get(value)

    def record(self, value: str) -> str:
        """Learn (or re-confirm) this value, returning its hash. Cheap to
        call redundantly -- callers do not need to pre-check ``hash_for``
        before calling this."""
        h = self._value_to_hash.get(value)
        if h is not None:
            return h
        h = compute_hash(value)
        self._value_to_hash[value] = h
        self._hash_to_value[h] = value
        return h

    def value_for(self, h: str) -> str | None:
        return self._hash_to_value.get(h)


# --------------------------------------------------------------------------
# Encode (client: plain body -> hash-substituted body) / decode (server:
# hash-substituted body -> plain body, or a list of misses).
# --------------------------------------------------------------------------

def encode_body(body: dict, cache: SessionCache,
                 threshold_bytes: int = DEFAULT_THRESHOLD_BYTES) -> dict:
    """Returns a NEW dict (deep-copies only what it touches; leaves the
    caller's ``body`` completely unmutated -- important because the
    caller's ``messages`` list is also its own multi-turn history, which
    must keep holding plain text for future turns, not hashes).

    Any candidate leaf whose value this session has already recorded is
    replaced by its hash and listed in ``$aipt_cache_map``. Any candidate
    leaf seen for the first time is left as-is AND recorded, so it is
    eligible for substitution starting next call. Sub-threshold leaves are
    never touched or recorded (matching the design's "short fields opt out
    for free" property).
    """
    import copy
    new_body = copy.deepcopy(body)
    cache_map: dict[str, str] = {}
    n = 0
    for path, value in list(iter_string_leaves(new_body)):
        if len(value.encode("utf-8")) < threshold_bytes:
            continue
        existing = cache.hash_for(value)
        if existing is not None:
            set_at_path(new_body, path, existing)
            cache_map[f"hashed_{n}"] = path_to_label(path)
            n += 1
        else:
            cache.record(value)
    if cache_map:
        new_body[CACHE_MAP_FIELD] = cache_map
    return new_body


class CacheMiss(Exception):
    """Raised by :func:`decode_body` when one or more paths listed in
    ``$aipt_cache_map`` reference a hash this session's store does not
    know. ``missing_paths`` carries the exact labels (as they appeared in
    the map) so the caller can build a 409 response or, on the client
    side, know exactly which paths to resend verbatim."""

    def __init__(self, missing_paths: list[str]) -> None:
        super().__init__(f"cache miss for paths: {missing_paths}")
        self.missing_paths = missing_paths


def decode_body(body: dict, cache: SessionCache,
                 threshold_bytes: int = DEFAULT_THRESHOLD_BYTES) -> dict:
    """Inverse of :func:`encode_body`, run on the receiving side. Returns a
    NEW dict with every ``$aipt_cache_map``-listed path restored to its
    real value, and the map field removed (upstream engines must never see
    this project's bookkeeping field). Also learns any sub-threshold-free
    candidate leaves that arrived as plain text (the "first appearance, or
    a post-409 resend" case) so this session's store stays in sync with
    the client's.

    Raises :class:`CacheMiss` (no partial mutation observable to the
    caller -- checked BEFORE any restoration happens) if any mapped path's
    hash is unknown.
    """
    import copy
    cache_map = body.get(CACHE_MAP_FIELD) or {}
    missing = []
    for label in cache_map.values():
        path = parse_label(label)
        h = get_at_path(body, path)
        if cache.value_for(h) is None:
            missing.append(label)
    if missing:
        raise CacheMiss(missing)

    new_body = copy.deepcopy(body)
    new_body.pop(CACHE_MAP_FIELD, None)
    mapped_paths = {parse_label(label) for label in cache_map.values()}
    for label in cache_map.values():
        path = parse_label(label)
        h = get_at_path(new_body, path)
        # `h` here is still the hash (we deep-copied body verbatim first).
        original = cache.value_for(h)
        set_at_path(new_body, path, original)
    # Symmetric learning pass: any string leaf NOT listed in the map is
    # either below threshold (record() below is a cheap no-op-ish path for
    # those too -- harmless to call, but skip on purpose to match the
    # client's threshold gate exactly) or a plain-text leaf the client
    # sent because this session had not seen it yet (or is resending after
    # a 409). Either way this side should learn it now, mirroring the
    # client's own "record on first send" behaviour.
    for path, value in list(iter_string_leaves(new_body)):
        if path in mapped_paths:
            continue
        if len(value.encode("utf-8")) < threshold_bytes:
            continue
        cache.record(value)
    return new_body


__all__ = [
    "CACHE_HEADER",
    "CACHE_HEADER_VALUE",
    "CACHE_MAP_FIELD",
    "HASH_LEN",
    "DEFAULT_THRESHOLD_BYTES",
    "compute_hash",
    "path_to_label",
    "parse_label",
    "get_at_path",
    "set_at_path",
    "iter_string_leaves",
    "SessionCache",
    "CacheMiss",
    "encode_body",
    "decode_body",
]
