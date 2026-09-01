#!/usr/bin/env python3
"""Verifies the cache-miss (409) recovery path: kill the engine_gateway
server and restart it (simulating a server-side session loss while the
client's TCP connection object still thinks it's the same session -- the
client's requests library will open a fresh TCP connection transparently,
but for this test we simulate the more interesting case: the *client's*
cache still has the hash while the server genuinely never learned it, by
constructing the scenario directly against cache_protocol rather than a
live restart (a live restart also changes the TCP connection, which
degrades to the trivial "fresh session, no map sent yet" case -- not the
409 path this test wants to exercise)."""
import sys
sys.path.insert(0, ".")
from aipt.core import cache_protocol as cp

client_cache = cp.SessionCache()
server_cache = cp.SessionCache()  # deliberately never learns anything

long_text = "z" * 300
body = {"messages": [{"role": "user", "content": long_text}]}

# Client believes it already sent this once and can hash it (simulating:
# client sent it on a previous request over the same TCP connection, but
# the server's in-process cache was somehow lost -- e.g. a bug, or this
# test's deliberate setup).
client_cache.record(long_text)
wire_body = cp.encode_body(body, client_cache)
assert wire_body["messages"][0]["content"] != long_text  # confirms it's hashed

# Server tries to decode and MUST raise CacheMiss (it never learned the hash)
missing = None
try:
    cp.decode_body(wire_body, server_cache)
    assert False, "expected CacheMiss"
except cp.CacheMiss as exc:
    missing = exc.missing_paths
    print("Server correctly raised CacheMiss for:", missing)
assert missing is not None

# Client's recovery: revert exactly the missing paths to plaintext using
# ITS OWN cache (mirroring gateway.py's _revert_missing_paths), then resend.
import copy
recovered = copy.deepcopy(wire_body)
cache_map = dict(recovered.get(cp.CACHE_MAP_FIELD) or {})
reverse_map = {v: k for k, v in cache_map.items()}
for label in missing:
    path = cp.parse_label(label)
    h = cp.get_at_path(recovered, path)
    original = client_cache.value_for(h)
    cp.set_at_path(recovered, path, original)
    cache_map.pop(reverse_map.get(label), None)
if cache_map:
    recovered[cp.CACHE_MAP_FIELD] = cache_map
else:
    recovered.pop(cp.CACHE_MAP_FIELD, None)

print("Recovered (resend) body:", recovered)
assert recovered["messages"][0]["content"] == long_text

# Server processes the resend -- should succeed this time and LEARN it.
decoded = cp.decode_body(recovered, server_cache)
assert decoded["messages"][0]["content"] == long_text
assert cp.CACHE_MAP_FIELD not in decoded
print("Server successfully decoded the resend and learned the value.")
print("\nCACHE-MISS RECOVERY TEST: PASS")
