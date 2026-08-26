#!/usr/bin/env python3
"""Client container entrypoint: applies netem, then serves the web UI.

Runs are triggered from the browser (POST /api/run), not automatically on
startup -- the whole point of the frontend is to let the operator set turn
count / payload growth / idle duration and watch the cwnd trace per run.
"""
import os
import sys
sys.path.insert(0, "/app")

from tcp_congestion.netem import from_env as netem_env, apply as netem_apply
from tcp_congestion.offload import from_env as offload_env, apply as offload_apply

cfg = netem_env()
if cfg["delay_ms"] > 0:
    print(f"[client] applying netem delay {cfg['delay_ms']}ms on {cfg['iface']}")
    try:
        netem_apply(**cfg)
    except Exception as e:
        print(f"[client] netem failed (continuing without): {e}")

ocfg = offload_env()
if ocfg["disable"]:
    print(f"[client] disabling NIC offload (tso/gso/sg/gro/lro) on {ocfg['iface']}")
    try:
        offload_apply(**ocfg)
    except Exception as e:
        print(f"[client] offload toggle failed (continuing without): {e}")

import uvicorn
from tcp_congestion.app import create_app

host = os.environ.get("WEB_HOST", "0.0.0.0")
port = int(os.environ.get("WEB_PORT", "10000"))
print(f"[client] web UI on {host}:{port}")

uvicorn.run(create_app(), host=host, port=port)
