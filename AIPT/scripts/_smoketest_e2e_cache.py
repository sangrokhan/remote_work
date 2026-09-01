#!/usr/bin/env python3
"""End-to-end smoke test: real Gateway.send() client (with cache_enabled)
against the real engine_gateway.py server (with cache decode) against a
fake echo upstream -- verifies:
  1. The upstream ALWAYS receives the real, uncorrupted content (proven via
     the echo's _echo_messages field) even when the wire carried a hash.
  2. Once a value has been seen, later turns actually replace it with a
     short hash on the wire (proven by inspecting result.request_body,
     which is exactly what was serialized and sent).
  3. The wire payload for a repeated long value is dramatically smaller
     than sending that value again in full would have been.
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

long_system = "You are a helpful assistant. " * 10  # 300 bytes, repeats every turn
assert len(long_system.encode()) > 50

messages = [{"role": "system", "content": long_system}]
wire.reset_session()

# Turn 0: first appearance -- system prompt must go out VERBATIM.
messages.append({"role": "user", "content": "q0"})
r0 = gw.send(messages)
assert r0.status == 200, (r0.status, r0.error)
assert r0.request_body["messages"][0]["content"] == long_system, (
    "turn 0 should send the system prompt verbatim (first appearance)"
)
assert r0.response_body["_echo_messages"][0]["content"] == long_system, (
    "upstream should have received the real system prompt on turn 0"
)
print("turn 0 OK: system prompt sent verbatim, upstream received it correctly")

messages.append({"role": "assistant", "content": "ack"})

# Turn 1: same system prompt repeats -- must be replaced by a short hash
# on the wire, but the upstream must still see the REAL text (proving the
# server decoded it before forwarding).
messages.append({"role": "user", "content": "q1"})
r1 = gw.send(messages)
assert r1.status == 200, (r1.status, r1.error)
wire_system_value = r1.request_body["messages"][0]["content"]
assert wire_system_value != long_system, (
    "turn 1 should have replaced the repeated system prompt with a hash on the wire"
)
assert len(wire_system_value) < len(long_system), "hash should be much shorter than original"
assert "$aipt_cache_map" in r1.request_body, "cache map field should be present on turn 1's wire body"
echoed_system = r1.response_body["_echo_messages"][0]["content"]
assert echoed_system == long_system, (
    f"upstream must have received the REAL system prompt (server decoded the hash), "
    f"got: {echoed_system[:50]!r}"
)
print(f"turn 1 OK: wire carried hash {wire_system_value!r} (was {len(long_system)} bytes), "
      f"upstream correctly received the real {len(long_system)}-byte prompt")

# Turn 2: same again -- same behaviour, and the SAVED bytes vs "what it
# would have cost to send verbatim again" should be substantial.
messages.append({"role": "assistant", "content": "ack"})
messages.append({"role": "user", "content": "q2"})
r2 = gw.send(messages)
assert r2.status == 200, (r2.status, r2.error)
assert r2.request_body["messages"][0]["content"] != long_system
saved_bytes = len(long_system.encode()) - len(wire_system_value.encode())
assert saved_bytes > 200, f"expected >200 bytes saved per repeated occurrence, got {saved_bytes}"
echoed_system2 = r2.response_body["_echo_messages"][0]["content"]
assert echoed_system2 == long_system
print(f"turn 2 OK: {saved_bytes} bytes saved on this repeated leaf vs sending verbatim, "
      f"upstream still correctly received the real prompt")

print("\nE2E CACHE PROTOCOL TEST: PASS")
print("  - request-body leaf-hash dedup works over the real HTTP wire")
print("  - engine Gateway correctly restores hashed leaves before forwarding upstream")
print("  - dedup only kicks in from the 2nd occurrence onward, as designed")
