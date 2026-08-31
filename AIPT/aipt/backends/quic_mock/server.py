"""aipt.backends.quic_mock.server -- QUIC echo server standing in for the
HTTP/1.1 mock server (aipt.backends.mock.server) but over QUIC/UDP.

Deliberately minimal (echo, not a real inference-mock endpoint) -- this
spike's question is "does idle-probing move cwnd the way we expect
through the real Gateway netem path", not "does the QUIC mock server
replicate every aipt.backends.mock.server feature". A full-featured QUIC
mock server (record replay, response padding, etc.) is a follow-up once
the spike result justifies promoting this to a real ``Backend`` protocol
implementation.
"""
from __future__ import annotations

import asyncio
import logging

from aioquic.asyncio import serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived

log = logging.getLogger(__name__)


class EchoProtocol(QuicConnectionProtocol):
    """Echoes every stream's bytes straight back on the same stream --
    just enough traffic shape (request in, response out, over one
    connection, across many turns) for the idle-probe cwnd experiment."""

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, StreamDataReceived):
            self._quic.send_stream_data(event.stream_id, event.data, event.end_stream)
            self.transmit()


async def run_server(host: str, port: int, cert_path: str, key_path: str, *,
                      create_protocol=EchoProtocol):
    """Starts the QUIC server and returns the running ``QuicServer`` (call
    ``.close()`` on it to shut down). ``create_protocol`` defaults to the
    plain ``EchoProtocol`` above (this spike's original traffic-shape-only
    server); the standalone ``quic-mock-server`` Docker service instead
    passes ``aipt.backends.quic_mock.backend._MockEchoProtocol`` so it
    speaks the same length-prefixed request protocol
    ``aipt.backends.quic_mock.backend.QuicMockBackend`` (the real
    Backend-protocol client) uses -- letting the web UI's Mock card, when
    pointed at this container via ``MOCK_SERVER_HOST``/``QUIC_MOCK_SERVER_HOST``,
    traverse the actual Gateway-routed L3 topology (DESIGN.md 4.7) instead
    of spawning its own loopback server, the same way ``LocalLLMBackend``
    always has."""
    config = QuicConfiguration(is_client=False)
    config.load_cert_chain(cert_path, key_path)
    server = await serve(host, port, configuration=config, create_protocol=create_protocol)
    log.info("quic_mock server listening on %s:%d", host, port)
    return server


async def _main() -> None:
    import argparse
    import signal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4433)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = await run_server(args.host, args.port, args.cert, args.key)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    server.close()


if __name__ == "__main__":
    asyncio.run(_main())
