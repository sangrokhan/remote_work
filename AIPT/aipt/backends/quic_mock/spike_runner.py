"""aipt.backends.quic_mock.spike_runner -- drives the QUIC idle-probe
experiment through the real Network Gateway (aipt/gateway/), not
loopback (unlike the throwaway spike this module replaces).

Run from inside the `web` container (it has the route to `gateway` and
transitively to `quic-mock-server` already set up by
docker/entrypoint_web.py -- see docker-compose.yml's QUIC_MOCK_SERVER_HOST/
QUIC_MOCK_SERVER_PORT env vars):

    docker compose exec web python -m aipt.backends.quic_mock.spike_runner \
        --profile 3g --turns 8 --think-time 0.3

What it does:
  1. Sets the Gateway's netem profile via ``POST /gateway/profile`` (same
     API aipt/web's own experiment form would use) -- so both congestion
     control variants run under an *actual* injected delay/jitter/loss
     path, not just measurement noise on loopback.
  2. Runs the mock conversation pattern (turn -> idle "think time" gap ->
     turn, N times) twice against ``quic-mock-server``: once with plain
     ``reno`` (baseline, no idle probing at all -- aioquic has no
     idle-restart logic of its own) and once with the custom
     ``idle_probe`` congestion control
     (aipt.backends.quic_mock.congestion), firing a PING mid-gap each
     time.
  3. Reports the cwnd/RTT trajectory for both runs.

Deliberately NOT wired into ``aipt/web/routes_run.py``/``RunRequest`` yet
-- this is a standalone measurement tool for the spike, matching the
user's explicit staging: (1) confirm the mechanism holds under a real
impaired path in Mock, (2) THEN decide on a UI checkbox / Backend
integration / HTTP-3-for-local_llm follow-up.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass

import requests

from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import PingAcknowledged, QuicEvent, StreamDataReceived

# Importing this registers the "idle_probe" congestion control algorithm
# name with aioquic (aipt/backends/quic_mock/congestion.py's module-level
# register_congestion_control() call).
from aipt.backends.quic_mock import congestion as idle_probe_cc  # noqa: F401

log = logging.getLogger("quic_spike_runner")

DEFAULT_QUIC_HOST = os.environ.get("QUIC_MOCK_SERVER_HOST", "127.0.0.1")
DEFAULT_QUIC_PORT = int(os.environ.get("QUIC_MOCK_SERVER_PORT", "4433"))
DEFAULT_GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "gateway")
DEFAULT_GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8080"))


def set_gateway_profile(gateway_host: str, gateway_port: int, profile: str) -> dict:
    """Same POST /gateway/profile call aipt/web's experiment form makes
    (aipt/gateway/app.py) -- switches both of Gateway's interfaces to the
    named preset (clean/broadband/3g/satellite/lossy)."""
    url = f"http://{gateway_host}:{gateway_port}/gateway/profile"
    resp = requests.post(url, json={"profile": profile}, timeout=10)
    resp.raise_for_status()
    return resp.json()


class ProbeAwareProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ping_rtt_event: asyncio.Event = asyncio.Event()
        self._reply_waiters: dict = {}

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, StreamDataReceived):
            waiter = self._reply_waiters.pop(event.stream_id, None)
            if waiter and not waiter.done():
                waiter.set_result(event.data)
        elif isinstance(event, PingAcknowledged):
            self.ping_rtt_event.set()

    async def send_turn(self, payload: bytes) -> bytes:
        stream_id = self._quic.get_next_available_stream_id()
        waiter = asyncio.get_event_loop().create_future()
        self._reply_waiters[stream_id] = waiter
        self._quic.send_stream_data(stream_id, payload, end_stream=False)
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


async def run_conversation(
    *, host: str, port: int, cc_name: str, num_turns: int, think_time: float,
    use_idle_probe: bool, cafile: str | None = None,
) -> list[dict]:
    config = QuicConfiguration(is_client=True, congestion_control_algorithm=cc_name)
    if cafile:
        config.load_verify_locations(cafile)
    config.verify_mode = False  # spike: self-signed cert, no real CA chain

    samples: list[dict] = []

    async with connect(
        host, port, configuration=config, create_protocol=ProbeAwareProtocol
    ) as client:
        cc = client._quic._loss._cc  # aioquic internal, spike-only introspection
        last_known_rtt = client._quic._loss._rtt_smoothed or 0.05

        for turn in range(num_turns):
            payload = f"turn-{turn}-{'x' * 500}".encode()
            t0 = time.monotonic()
            await client.send_turn(payload)
            t1 = time.monotonic()

            samples.append({
                "turn": turn, "phase": "turn_sent", "t": round(t1, 3),
                "cwnd": cc.congestion_window, "req_ms": round((t1 - t0) * 1000, 2),
            })

            if use_idle_probe and hasattr(cc, "mark_idle_probe_sent"):
                await asyncio.sleep(think_time / 2)
                rtt_before = client._quic._loss._rtt_smoothed or last_known_rtt
                cc.mark_idle_probe_sent(rtt_before)
                await client.idle_probe(uid=turn)
                await asyncio.sleep(think_time / 2)
            else:
                await asyncio.sleep(think_time)

            last_known_rtt = client._quic._loss._rtt_smoothed or last_known_rtt
            samples.append({
                "turn": turn, "phase": "after_idle", "t": round(time.monotonic(), 3),
                "cwnd": cc.congestion_window,
                "srtt_ms": round(last_known_rtt * 1000, 3),
            })

        if hasattr(cc, "idle_adjustments"):
            for adj in cc.idle_adjustments:
                log.info("idle adjustment: %s", adj)

    return samples


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_QUIC_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_QUIC_PORT)
    parser.add_argument("--gateway-host", default=DEFAULT_GATEWAY_HOST)
    parser.add_argument("--gateway-port", type=int, default=DEFAULT_GATEWAY_PORT)
    parser.add_argument("--profile", default="clean",
                         help="Gateway netem preset: clean/broadband/3g/satellite/lossy")
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--think-time", type=float, default=0.3,
                         help="Idle gap between turns, seconds (LLM 'thinking' time)")
    parser.add_argument("--out", default="quic_spike_result.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    log.info("setting Gateway profile to %r", args.profile)
    profile_result = set_gateway_profile(args.gateway_host, args.gateway_port, args.profile)
    log.info("Gateway profile response: %s", profile_result)

    log.info("=== baseline: reno on QUIC, no idle probing ===")
    baseline = await run_conversation(
        host=args.host, port=args.port, cc_name="reno",
        num_turns=args.turns, think_time=args.think_time, use_idle_probe=False,
    )

    log.info("=== idle_probe cc, active RTT probing during idle gaps ===")
    probed = await run_conversation(
        host=args.host, port=args.port, cc_name="idle_probe",
        num_turns=args.turns, think_time=args.think_time, use_idle_probe=True,
    )

    result = {
        "gateway_profile": args.profile,
        "gateway_profile_response": profile_result,
        "turns": args.turns,
        "think_time_s": args.think_time,
        "baseline": baseline,
        "idle_probe": probed,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult written to {args.out}")
    print("\n--- baseline (reno, no probing) cwnd trajectory ---")
    for s in baseline:
        print(s)
    print("\n--- idle_probe cc cwnd trajectory ---")
    for s in probed:
        print(s)


if __name__ == "__main__":
    asyncio.run(main())
