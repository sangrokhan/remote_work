#!/usr/bin/env python3
"""Verifies Gateway.send()'s ACTUAL 409-retry branch (aipt/backends/
local_llm/gateway.py's cache-miss recovery block) fires and recovers
correctly against the real running engine_gateway.py server.

Technique: prime the Gateway's client-side cache with a value+hash pair
the SERVER never saw (bypassing send() so the server genuinely never
learned it), then call send() with a message that will get hashed using
that value. The server must 409, and Gateway.send()'s own retry logic
(not test code) must catch it, revert, resend, and return 200.
"""
import os
import sys

sys.path.insert(0, ".")
os.environ["LOCAL_LLM_ENGINE_URL"] = "http://127.0.0.1:40079"

from aipt.backends.local_llm.engine_adapter import EngineAdapter
from aipt.backends.local_llm.gateway import Gateway
from aipt.core import wire

adapter = EngineAdapter(base_url="http://127.0.0.1:40079", model="local-model")
gw = Gateway(adapter, cache_enabled=True, cache_threshold_bytes=50)
wire.reset_session()

phantom_value = "phantom-value-server-never-saw " * 5  # > 50 bytes
assert gw._cache is not None
phantom_hash = gw._cache.record(phantom_value)  # client "knows" it; server does NOT
print(f"Client cache primed with a hash the server never learned: {phantom_hash}")

messages = [{"role": "user", "content": phantom_value}]
result = gw.send(messages)

assert result.status == 200, (
    f"expected Gateway.send()'s 409-retry to recover and return 200, "
    f"got status={result.status} error={result.error}"
)
echoed = result.response_body.get("_echo_messages")
assert echoed[0]["content"] == phantom_value, (
    f"upstream should have received the real (recovered) value, got: "
    f"{echoed[0]['content'][:50]!r}"
)
# The request that actually succeeded (visible via request_body, which
# GatewayResult always carries for the LAST _send_once call) must have
# been sent in plaintext, not carrying a cache map for this leaf anymore --
# proving the retry path reverted it rather than the server somehow
# guessing the plaintext.
assert result.request_body["messages"][0]["content"] == phantom_value, (
    "the recovered/resent request should have sent the real value in plaintext"
)
print("Gateway.send() correctly caught the 409, reverted the missing path via "
      "its own cache, resent, and got 200 with the correct content restored upstream.")
print("\nGATEWAY 409-RETRY BRANCH TEST: PASS")
