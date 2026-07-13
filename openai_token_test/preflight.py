"""One cheap turn on each arm, to prove the key, the model, and all three
endpoints work before spending a real run on them.

Costs a few hundredths of a cent. Run this first.
"""

from __future__ import annotations

import sys

import fixture as fixture_mod
import openai_client as oc


def main() -> int:
    fx = fixture_mod.load("perf")
    model = oc.DEFAULT_MODEL
    print(f"model={model} base={oc.base_url()} system={fx.system_chars} chars\n")

    ok = True
    conversation = None

    for arm in oc.ARMS:
        try:
            if arm == "responses_stateful":
                conversation, setup = oc.create_conversation(fx.system)
                print(f"  conversations/create   -> {conversation} "
                      f"({setup.req_payload_bytes:,}B up)")
            res = oc.call(arm, model=model, system=fx.system, history=[],
                          question=fx.steps[0], turn=1, conversation=conversation)
            print(f"  {arm:<20} OK  up={res.req_payload_bytes:,}B  "
                  f"in_tok={res.input_tokens:,} cached={res.cached_tokens:,} "
                  f"out_tok={res.output_tokens:,} reasoning={res.reasoning_tokens} "
                  f"{res.latency_ms}ms")
            if res.reasoning_tokens:
                print(f"    ! reasoning_tokens={res.reasoning_tokens} — set "
                      f"OPENAI_REASONING_EFFORT=none or the byte measurement gets noisy")
            if not res.text:
                print("    ! empty completion — check max_output_tokens")
        except Exception as e:
            ok = False
            print(f"  {arm:<20} FAIL  {e}")

    print("\nready" if ok else "\nnot ready — fix the failures above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
