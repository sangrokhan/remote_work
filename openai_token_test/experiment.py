"""Run the three arms over the same fixture and record every call.

Each arm replays the identical turn sequence against the identical model. The
only thing that differs is who remembers the conversation:

  chat_stateless / responses_stateless -> the client does, and re-uploads it
  responses_stateful                   -> the server does, and the client doesn't

Arms are run one after another (single-threaded), so the socket byte tally in
wire.py is unambiguous: only ever one request in flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import fixture as fixture_mod
import openai_client as oc
import wire


@dataclass
class ArmRun:
    arm: str
    repeat: int
    setup: dict | None = None          # conversation-create call, stateful arm only
    turns: list[dict] = field(default_factory=list)


def cache_key_for(arm: str, repeat: int) -> str:
    """Stable within one arm-run, distinct across arms and across repeats.

    Stable, because prompt-cache routing only sticks if the key does not change
    turn to turn. Distinct across arms, because a shared key would let whichever
    arm ran first warm the cache for the one that ran second, and the second arm
    would look cheaper for no reason but its position. Distinct across repeats,
    because each repeat is supposed to be an independent conversation that starts
    cold — reusing the key would hand repeats 2..N a warm first turn and make the
    averaged cold-start cost a fiction.
    """
    return f"otst-{arm}-r{repeat}"


def run_arm(arm: str, fx: fixture_mod.Fixture, *, model: str, turns: int,
            repeat: int, stream: bool = False, on_turn=None) -> ArmRun:
    run = ArmRun(arm=arm, repeat=repeat)
    history: list[dict] = []
    conversation = None
    cache_key = cache_key_for(arm, repeat)

    if arm == "responses_stateful":
        conversation, setup = oc.create_conversation(fx.system)
        run.setup = setup.as_dict(bodies=True)

    for k, question in enumerate(fx.head(turns), start=1):
        res = oc.call(
            arm,
            model=model,
            system=fx.system,
            history=history,
            question=question,
            turn=k,
            conversation=conversation,
            cache_key=cache_key,
            stream=stream,
        )
        run.turns.append(res.as_dict(bodies=True))

        # The stateless arms must carry the transcript forward themselves; that
        # growing list is exactly what they re-upload next turn. The stateful arm
        # keeps no history — the server appended both messages for it.
        if arm != "responses_stateful":
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": res.text})

        if on_turn:
            on_turn(arm, k, turns, res)

    return run


def run_experiment(*, fixture_name: str = "perf", model: str = oc.DEFAULT_MODEL,
                   turns: int = 10, repeats: int = 3,
                   arms: tuple[str, ...] = oc.ARMS, capture: bool = False,
                   stream: bool = False, on_turn=None) -> dict:
    """Run every arm over the fixture.

    `capture` wraps each arm in its own tcpdump, so the pcap for an arm contains
    only that arm's packets. One pcap for the whole run would put all three arms
    in one file and the whole point — comparing their traffic — would be lost.

    `stream` is what makes TTFT measurable at all: a first token only exists in a
    stream. It comes at a price. Upload bytes stay comparable (the stream flags
    add ~66 B), but DOWNLOAD bytes do not: SSE framing, the per-chunk envelope,
    and the full Response object repeated in created/completed all ride along. So
    a streamed run answers "how fast", and a plain run answers "how many bytes
    came back". Both answer "how many bytes went up", which is the thesis.

    The socket byte tally is global and single-threaded, so arms must not overlap;
    they run strictly one after another.
    """
    fx = fixture_mod.load(fixture_name)
    turns = min(turns, len(fx.steps))

    runs: list[dict] = []
    captures: list[dict] = []

    for repeat in range(1, repeats + 1):
        for arm in arms:
            if capture:
                import capture as cap_mod
                from datetime import datetime, timezone
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                # a fresh socket per arm, so each pcap opens on a real TCP
                # handshake instead of mid-stream on a pooled connection
                wire.reset_session()
                with cap_mod.Capture(ts, arm=arm) as cap:
                    run = run_arm(arm, fx, model=model, turns=turns,
                                  repeat=repeat, stream=stream, on_turn=on_turn)
                wire.reset_session()
                res = cap.result()
                res["repeat"] = repeat
                captures.append(res)
            else:
                run = run_arm(arm, fx, model=model, turns=turns, repeat=repeat,
                              stream=stream, on_turn=on_turn)

            runs.append({"arm": run.arm, "repeat": run.repeat,
                         "setup": run.setup, "turns": run.turns})

    return {
        "config": {
            "model": model,
            "fixture": fx.name,
            "turns": turns,
            "repeats": repeats,
            "arms": list(arms),
            "system_chars": fx.system_chars,
            "max_output_tokens": oc.DEFAULT_MAX_OUTPUT_TOKENS,
            "reasoning_effort": oc.DEFAULT_REASONING_EFFORT,
            "stream": stream,
            "capture": capture,
        },
        "runs": runs,
        "captures": captures,
    }


def _progress(arm: str, k: int, n: int, res: oc.CallResult) -> None:
    timing = (f"ttft={res.ttft_ms:>5}ms ttlt={res.ttlt_ms:>6}ms"
              if res.streamed else f"{res.latency_ms:>6}ms")
    print(f"  {arm:<20} turn {k:>2}/{n}  "
          f"up={res.req_payload_bytes:>7}B  "
          f"in_tok={res.input_tokens:>6} (cached {res.cached_tokens:>6})  "
          f"{timing}",
          flush=True)


def main() -> None:
    import argparse

    import metrics
    import store

    ap = argparse.ArgumentParser(description="OpenAI stateless-vs-stateful traffic experiment")
    ap.add_argument("--fixture", default="perf")
    ap.add_argument("--model", default=oc.DEFAULT_MODEL)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--arms", default=",".join(oc.ARMS))
    ap.add_argument("--capture", action="store_true",
                    help="run tcpdump around each arm and keep the .pcap")
    ap.add_argument("--stream", action="store_true",
                    help="stream the responses, so TTFT/TTLT can be measured. "
                         "Upload bytes stay comparable; download bytes do not "
                         "(SSE framing rides along).")
    args = ap.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    print(f"model={args.model} fixture={args.fixture} turns={args.turns} "
          f"repeats={args.repeats} arms={list(arms)}", flush=True)

    if args.capture:
        import capture as cap_mod
        ok, why = cap_mod.available()
        print(f"capture: {'ready' if ok else 'UNAVAILABLE — ' + why}")
        if not ok:
            print("continuing without packet capture\n")
        args.capture = ok
    print(flush=True)

    exp = run_experiment(fixture_name=args.fixture, model=args.model,
                         turns=args.turns, repeats=args.repeats, arms=arms,
                         capture=args.capture, stream=args.stream,
                         on_turn=_progress)

    summary = metrics.summarize(exp)
    exec_id = store.new_exec_id()
    manifest = store.save_run(exec_id, exp, summary, exp.get("captures"))

    metrics.print_summary(summary)
    print(f"\nsaved {manifest['dir']}")
    print(f"  run.json · summary.csv · charts.png · {manifest['bodies']} raw bodies"
          + (f" · {manifest['captures']} pcaps" if manifest["captures"] else ""))


if __name__ == "__main__":
    main()
