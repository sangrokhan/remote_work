"""aipt.backends.quic_mock.experiment -- throughput/latency A/B measurement
for IdleProbeCongestionControl vs plain Reno, through the real Network
Gateway (aipt/gateway/).

This is DESIGN.md section 7's explicit "next step 2": the cwnd-trajectory
spike (spike_runner.py) confirmed idle_probe *reacts* to measured RTT
growth by shrinking cwnd before the next transmission -- it did NOT show
whether that reaction is actually a net win in throughput or latency. A
smaller cwnd after idle means fewer bytes can be in flight during the
next turn's slow-start ramp, which could just as easily make each turn
*slower* if the shrink was too aggressive or the RTT growth was transient
noise rather than sustained congestion. This module measures that
directly.

Methodology:
  * Payload size is deliberately larger than the initial congestion
    window (10 * max_datagram_size, ~12000 bytes) so a turn actually
    takes multiple round trips to complete -- with a payload smaller than
    cwnd, the whole thing goes out in one flight regardless of cwnd size,
    and no difference between algorithms could show up at all.
  * Per turn: wall-clock time from send to the *full* echoed payload
    being received back (fixed from spike_runner.py's ProbeAwareProtocol,
    which only waited for the first fragment -- fine for cwnd-trajectory
    logging, wrong for a real latency measurement of a multi-RTT
    transfer).
  * N repetitions of the whole conversation are run for each congestion
    control, back-to-back under the same Gateway profile, to average out
    per-connection noise (netem's own jitter, scheduling jitter) rather
    than trusting a single run.
  * Reports, per congestion control: mean/stdev of per-turn latency
    (post-idle turns only -- turn 0 has no prior idle gap to react to),
    and aggregate goodput (total payload bytes / total wall time across
    all repetitions).

Run from inside the `web` container:

    docker compose exec web python -m aipt.backends.quic_mock.experiment \\
        --profile 3g --turns 6 --think-time 1.0 --repeats 5 \\
        --payload-bytes 30000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time

import requests

from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import PingAcknowledged, QuicEvent, StreamDataReceived

from aipt.backends.quic_mock import congestion as idle_probe_cc  # noqa: F401
from aipt.backends.quic_mock.spike_runner import set_gateway_profile

log = logging.getLogger("quic_experiment")

DEFAULT_QUIC_HOST = os.environ.get("QUIC_MOCK_SERVER_HOST", "127.0.0.1")
DEFAULT_QUIC_PORT = int(os.environ.get("QUIC_MOCK_SERVER_PORT", "4433"))
DEFAULT_GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "gateway")
DEFAULT_GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8080"))


class ThroughputProtocol(QuicConnectionProtocol):
    """Like spike_runner.ProbeAwareProtocol, but send_turn() correctly
    waits for the *complete* echoed payload (accumulates fragments until
    end_stream=True) instead of resolving on the first STREAM frame --
    the bug that would have made any multi-RTT latency measurement
    meaningless (a large payload arrives as several separate
    StreamDataReceived events, one per received frame/flight)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ping_rtt_event: asyncio.Event = asyncio.Event()
        self._reply_waiters: dict = {}
        self._reply_buffers: dict = {}

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, StreamDataReceived):
            buf = self._reply_buffers.setdefault(event.stream_id, bytearray())
            buf.extend(event.data)
            if event.end_stream:
                waiter = self._reply_waiters.pop(event.stream_id, None)
                if waiter and not waiter.done():
                    waiter.set_result(bytes(buf))
                del self._reply_buffers[event.stream_id]
        elif isinstance(event, PingAcknowledged):
            self.ping_rtt_event.set()

    async def send_turn(self, payload: bytes) -> bytes:
        stream_id = self._quic.get_next_available_stream_id()
        waiter = asyncio.get_event_loop().create_future()
        self._reply_waiters[stream_id] = waiter
        self._quic.send_stream_data(stream_id, payload, end_stream=True)
        self.transmit()
        return await waiter

    async def idle_probe(self, uid: int, timeout: float = 3.0) -> None:
        self.ping_rtt_event.clear()
        self._quic.send_ping(uid)
        self.transmit()
        try:
            await asyncio.wait_for(self.ping_rtt_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("idle probe %d: no ack within %.1fs", uid, timeout)


async def run_one_conversation(
    *, host: str, port: int, cc_name: str, num_turns: int, think_time: float,
    use_idle_probe: bool, payload_bytes: int,
) -> dict:
    config = QuicConfiguration(is_client=True, congestion_control_algorithm=cc_name)
    config.verify_mode = False  # spike: self-signed cert, no real CA chain

    turn_latencies_ms: list[float] = []
    total_bytes = 0
    t_start = time.monotonic()

    async with connect(
        host, port, configuration=config, create_protocol=ThroughputProtocol
    ) as client:
        cc = client._quic._loss._cc
        last_known_rtt = client._quic._loss._rtt_smoothed or 0.05

        for turn in range(num_turns):
            payload = bytes(f"turn-{turn}-", "ascii") + b"x" * payload_bytes
            t0 = time.monotonic()
            reply = await client.send_turn(payload)
            t1 = time.monotonic()

            latency_ms = (t1 - t0) * 1000
            # turn 0 has no preceding idle gap -- exclude it from the
            # "post-idle" comparison this experiment cares about, but
            # still record it for completeness.
            turn_latencies_ms.append(latency_ms)
            total_bytes += len(payload) + len(reply)

            if use_idle_probe and hasattr(cc, "mark_idle_probe_sent"):
                await asyncio.sleep(think_time / 2)
                rtt_before = client._quic._loss._rtt_smoothed or last_known_rtt
                cc.mark_idle_probe_sent(rtt_before)
                await client.idle_probe(uid=turn)
                await asyncio.sleep(think_time / 2)
            else:
                await asyncio.sleep(think_time)

            last_known_rtt = client._quic._loss._rtt_smoothed or last_known_rtt

    t_end = time.monotonic()
    wall_time_s = t_end - t_start - think_time * num_turns  # exclude idle sleeps
    return {
        "turn_latencies_ms": turn_latencies_ms,
        "total_bytes": total_bytes,
        "active_wall_time_s": wall_time_s,
    }


async def run_experiment(
    *, host: str, port: int, cc_name: str, use_idle_probe: bool,
    num_turns: int, think_time: float, payload_bytes: int, repeats: int,
) -> dict:
    runs = []
    for r in range(repeats):
        log.info("  repeat %d/%d (%s)...", r + 1, repeats, cc_name)
        runs.append(await run_one_conversation(
            host=host, port=port, cc_name=cc_name, num_turns=num_turns,
            think_time=think_time, use_idle_probe=use_idle_probe,
            payload_bytes=payload_bytes,
        ))

    # Post-idle turns = every turn except the first (turn 0 has no prior
    # idle gap for the probe to have reacted to).
    post_idle_latencies = [
        lat for run in runs for lat in run["turn_latencies_ms"][1:]
    ]
    total_bytes = sum(run["total_bytes"] for run in runs)
    total_active_time = sum(run["active_wall_time_s"] for run in runs)

    return {
        "cc_name": cc_name,
        "use_idle_probe": use_idle_probe,
        "repeats": repeats,
        "runs": runs,
        "post_idle_latency_ms_mean": round(statistics.mean(post_idle_latencies), 2),
        "post_idle_latency_ms_stdev": (
            round(statistics.stdev(post_idle_latencies), 2)
            if len(post_idle_latencies) > 1 else 0.0
        ),
        "post_idle_latency_ms_max": round(max(post_idle_latencies), 2),
        "goodput_bps": round(total_bytes / total_active_time, 1) if total_active_time > 0 else 0,
        "total_bytes": total_bytes,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_QUIC_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_QUIC_PORT)
    parser.add_argument("--gateway-host", default=DEFAULT_GATEWAY_HOST)
    parser.add_argument("--gateway-port", type=int, default=DEFAULT_GATEWAY_PORT)
    parser.add_argument("--profile", default="3g",
                         help="Gateway netem preset: clean/broadband/3g/satellite/lossy")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--think-time", type=float, default=1.0)
    parser.add_argument("--payload-bytes", type=int, default=30000,
                         help="Per-turn payload size; must exceed initial cwnd "
                              "(~12000 bytes) for cwnd differences to matter at all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="quic_experiment_result.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    log.info("setting Gateway profile to %r", args.profile)
    profile_result = set_gateway_profile(args.gateway_host, args.gateway_port, args.profile)
    log.info("Gateway profile response: %s", profile_result)

    log.info("=== baseline: reno on QUIC, no idle probing (%d repeats) ===", args.repeats)
    baseline = await run_experiment(
        host=args.host, port=args.port, cc_name="reno", use_idle_probe=False,
        num_turns=args.turns, think_time=args.think_time,
        payload_bytes=args.payload_bytes, repeats=args.repeats,
    )

    log.info("=== idle_probe cc, active RTT probing (%d repeats) ===", args.repeats)
    probed = await run_experiment(
        host=args.host, port=args.port, cc_name="idle_probe", use_idle_probe=True,
        num_turns=args.turns, think_time=args.think_time,
        payload_bytes=args.payload_bytes, repeats=args.repeats,
    )

    result = {
        "gateway_profile": args.profile,
        "gateway_profile_response": profile_result,
        "turns": args.turns,
        "think_time_s": args.think_time,
        "payload_bytes": args.payload_bytes,
        "repeats": args.repeats,
        "baseline": baseline,
        "idle_probe": probed,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    latency_delta_pct = (
        (probed["post_idle_latency_ms_mean"] - baseline["post_idle_latency_ms_mean"])
        / baseline["post_idle_latency_ms_mean"] * 100
        if baseline["post_idle_latency_ms_mean"] else 0.0
    )
    goodput_delta_pct = (
        (probed["goodput_bps"] - baseline["goodput_bps"]) / baseline["goodput_bps"] * 100
        if baseline["goodput_bps"] else 0.0
    )

    print(f"\nResult written to {args.out}\n")
    print(f"{'':20} {'baseline (reno)':>20} {'idle_probe':>15} {'delta':>10}")
    print(f"{'post-idle latency':20} {baseline['post_idle_latency_ms_mean']:>17.1f}ms "
          f"{probed['post_idle_latency_ms_mean']:>13.1f}ms {latency_delta_pct:>+9.1f}%")
    print(f"{'  (stdev)':20} {baseline['post_idle_latency_ms_stdev']:>17.1f}ms "
          f"{probed['post_idle_latency_ms_stdev']:>13.1f}ms")
    print(f"{'  (max)':20} {baseline['post_idle_latency_ms_max']:>17.1f}ms "
          f"{probed['post_idle_latency_ms_max']:>13.1f}ms")
    print(f"{'goodput':20} {baseline['goodput_bps']:>17.0f}bps "
          f"{probed['goodput_bps']:>13.0f}bps {goodput_delta_pct:>+9.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
