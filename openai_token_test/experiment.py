"""Run the three arms over the same fixture and record every call.

Each arm replays the identical turn sequence against the identical model. The
only thing that differs is who remembers the conversation:

  chat_stateless / responses_stateless -> the client does, and re-uploads it
  responses_stateful                   -> the server does, and the client doesn't

Arms are run one after another (single-threaded), so the socket byte tally in
wire.py is unambiguous: only ever one request in flight.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import fixture as fixture_mod
import openai_client as oc

RESULTS_DIR = Path(__file__).parent / "results"


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
            repeat: int, on_turn=None) -> ArmRun:
    run = ArmRun(arm=arm, repeat=repeat)
    history: list[dict] = []
    conversation = None
    cache_key = cache_key_for(arm, repeat)

    if arm == "responses_stateful":
        conversation, setup = oc.create_conversation(fx.system)
        run.setup = setup.as_dict()

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
        )
        run.turns.append(res.as_dict())

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
                   arms: tuple[str, ...] = oc.ARMS, on_turn=None) -> dict:
    fx = fixture_mod.load(fixture_name)
    turns = min(turns, len(fx.steps))

    runs: list[dict] = []
    for repeat in range(1, repeats + 1):
        for arm in arms:
            run = run_arm(arm, fx, model=model, turns=turns, repeat=repeat,
                          on_turn=on_turn)
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
            "stream": False,
        },
        "runs": runs,
    }


def save(experiment: dict, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(experiment, indent=2))
    return path


def _progress(arm: str, k: int, n: int, res: oc.CallResult) -> None:
    print(f"  {arm:<20} turn {k:>2}/{n}  "
          f"up={res.req_payload_bytes:>7}B  "
          f"in_tok={res.input_tokens:>6} (cached {res.cached_tokens:>6})  "
          f"{res.latency_ms:>5}ms",
          flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="OpenAI stateless-vs-stateful traffic experiment")
    ap.add_argument("--fixture", default="perf")
    ap.add_argument("--model", default=oc.DEFAULT_MODEL)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--arms", default=",".join(oc.ARMS))
    ap.add_argument("--name", default="run")
    args = ap.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    print(f"model={args.model} fixture={args.fixture} turns={args.turns} "
          f"repeats={args.repeats} arms={list(arms)}\n", flush=True)

    exp = run_experiment(fixture_name=args.fixture, model=args.model,
                         turns=args.turns, repeats=args.repeats, arms=arms,
                         on_turn=_progress)
    path = save(exp, args.name)
    print(f"\nsaved {path}")

    import metrics
    metrics.print_summary(metrics.summarize(exp))


if __name__ == "__main__":
    main()
