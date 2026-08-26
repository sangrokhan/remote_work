#!/usr/bin/env python3
"""Server container entrypoint.

1. Apply tc netem delay (NETEM_DELAY_MS, NETEM_IFACE) if nonzero.
2. Apply NIC offload toggle (NIC_OFFLOAD_DISABLE, NIC_OFFLOAD_IFACE) if set.
3. Start the inference-mock HTTP server.
"""
import os
import sys
sys.path.insert(0, "/app")

from tcp_congestion.netem import from_env as netem_env, apply as netem_apply
from tcp_congestion.offload import from_env as offload_env, apply as offload_apply
from tcp_congestion.server import Server

cfg = netem_env()
if cfg["delay_ms"] > 0:
    print(f"[server] applying netem delay {cfg['delay_ms']}ms on {cfg['iface']}")
    try:
        netem_apply(**cfg)
    except Exception as e:
        print(f"[server] netem failed (continuing without): {e}")

ocfg = offload_env()
if ocfg["disable"]:
    print(f"[server] disabling NIC offload (tso/gso/sg/gro/lro) on {ocfg['iface']}")
    try:
        offload_apply(**ocfg)
    except Exception as e:
        print(f"[server] offload toggle failed (continuing without): {e}")

host = os.environ.get("SERVER_HOST", "0.0.0.0")
port = int(os.environ.get("SERVER_PORT", "8888"))
print(f"[server] listening on {host}:{port}")

srv = Server(host=host, port=port)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    srv.shutdown()
