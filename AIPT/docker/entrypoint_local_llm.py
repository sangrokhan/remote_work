#!/usr/bin/env python3
"""docker/entrypoint_local_llm.py -- container entrypoint for the AIPT
local-llm image (docker/Dockerfile.local_llm, mirrors
docker/entrypoint_mockserver.py's routing contract).

This container wraps the upstream ``ghcr.io/ggml-org/llama.cpp:server``
image (a real llama.cpp `llama-server` OpenAI-compatible HTTP server) --
it does not reimplement inference, matching
``aipt/backends/local_llm/engine_adapter.py``'s explicit "do not
reimplement inference" stance. This script only does two things before
handing off to the real ``llama-server`` binary:

1. Same L3-routing fix-up as ``entrypoint_mockserver.py``/``entrypoint_web.py``
   (DESIGN.md 4.7 확정 설계 1): ``local-llm`` lives on ``net-backend`` only,
   so without an explicit ``ip route add <net-client subnet> via <gateway's
   net-backend IP>`` its replies to ``web`` would have no route back (or
   could take a shortcut that bypasses ``gateway``, defeating netem
   injection). Needs NET_ADMIN (docker-compose.yml: `cap_add: [NET_ADMIN]`
   on `local-llm`) -- failures are logged and swallowed, never fatal.
2. ``exec``s the real ``llama-server`` binary (``/app/llama-server`` in the
   upstream image) with host/port/model args from env vars, so the final
   PID 1 is the actual engine (signals/healthcheck behave exactly like the
   upstream image).

Env vars:
  LLAMA_HOST        default "0.0.0.0"
  LLAMA_PORT        default "40080" (AIPT convention: local_llm lab uses the
                    40000s to avoid clashing with the 8080/8888/10000 ports
                    the other services already use)
  MODEL_REPO        "<hf-repo>:<quant>" shorthand llama-server resolves via
                    the HF Hub (default: the same small smoke-test model
                    scripts/run_local_llm_engine.sh uses)
  CTX_SIZE          default "4096"
  GATEWAY_PEER_SUBNET -- net-client CIDR to route via gateway
  GATEWAY_ROUTE_VIA   -- gateway's own IP address on net-backend
  IDLE_RESET_ADMIN_PORT -- default "40081" (idle_reset_admin.py sidecar,
                    see that module's docstring for why local-llm needs a
                    separate admin server rather than an in-process route
                    the way mock-server has).
"""
import os
import subprocess
import sys

PEER_SUBNET = os.environ.get("GATEWAY_PEER_SUBNET", "").strip()
ROUTE_VIA = os.environ.get("GATEWAY_ROUTE_VIA", "").strip()


def _add_route() -> None:
    if not PEER_SUBNET or not ROUTE_VIA:
        print(
            "[entrypoint_local_llm] GATEWAY_PEER_SUBNET/GATEWAY_ROUTE_VIA not "
            "set -- skipping explicit route via gateway (fine for "
            "standalone/dev runs outside the DESIGN.md 4.7 Docker topology)."
        )
        return
    argv = ["ip", "route", "add", PEER_SUBNET, "via", ROUTE_VIA]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        print("[entrypoint_local_llm] `ip` (iproute2) not installed -- cannot add route, continuing anyway.")
        return
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[entrypoint_local_llm] route setup failed: {exc} -- continuing anyway.")
        return
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if "File exists" in err:
            print(f"[entrypoint_local_llm] route to {PEER_SUBNET} via {ROUTE_VIA} already present, skipping.")
            return
        print(
            f"[entrypoint_local_llm] `{' '.join(argv)}` exited {proc.returncode}: {err[:200]} "
            "-- likely missing NET_ADMIN (docker-compose: cap_add: [NET_ADMIN] on `local-llm`). "
            "Continuing anyway; response traffic may not traverse gateway correctly."
        )
    else:
        print(f"[entrypoint_local_llm] route added: {PEER_SUBNET} via {ROUTE_VIA}")


_add_route()

# Spawn the idle-reset admin sidecar as its own DETACHED PROCESS, not an
# in-process thread. This matters: os.execvp() below replaces this entire
# process image (that's the whole point -- PID 1 becomes the real
# llama-server, exactly as documented above), which destroys every thread
# this process was running, including a daemon thread. A first version of
# this file started the admin server as a thread via
# idle_reset_admin.start_background() and it silently vanished the moment
# execvp() ran -- found during the 2026-09-01 ooo end-to-end verification
# (GET /admin/idle-reset via the web proxy returned "connection refused"
# even though the code looked correct and the import worked fine under
# `docker compose exec`). subprocess.Popen here creates a genuinely
# separate OS process that survives this parent's exec unaffected.
_ADMIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idle_reset_admin.py")
try:
    subprocess.Popen([sys.executable, _ADMIN_SCRIPT])
    print(f"[entrypoint_local_llm] idle-reset admin sidecar spawned ({_ADMIN_SCRIPT})")
except Exception as exc:  # pragma: no cover - defensive
    print(f"[entrypoint_local_llm] failed to spawn idle-reset admin sidecar: {exc} -- continuing anyway.")

host = os.environ.get("LLAMA_HOST", "0.0.0.0")
port = os.environ.get("LLAMA_PORT", "40080")
model_repo = os.environ.get("MODEL_REPO", "bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M")
ctx_size = os.environ.get("CTX_SIZE", "4096")

print(f"[local-llm] starting llama-server on {host}:{port}, model={model_repo}")

argv = [
    "/app/llama-server",
    "-hf", model_repo,
    "--host", host,
    "--port", port,
    "-c", ctx_size,
]
os.execvp(argv[0], argv)
