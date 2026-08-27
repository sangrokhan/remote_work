#!/usr/bin/env bash
# scripts/run_local_llm_engine.sh -- stand up a llama.cpp OpenAI-compatible
# engine for aipt.backends.local_llm to talk to (DESIGN.md 4.5/B4 explicitly
# leaves "actually starting an engine process" out of the adapter itself --
# this script is that follow-up operational concern).
#
# What it does:
#   1. Downloads a prebuilt llama.cpp release (linux x64 CPU build) into
#      $LLAMA_CPP_DIR if llama-server isn't already there.
#   2. Runs `llama-server -hf <repo:quant>` which pulls the GGUF straight
#      from the Hugging Face Hub (cached under ~/.cache/llama.cpp) and
#      serves it on POST /v1/chat/completions.
#
# Usage:
#   ./scripts/run_local_llm_engine.sh                 # defaults below
#   MODEL_REPO=bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M \
#     PORT=40090 ./scripts/run_local_llm_engine.sh
#
# Then point the local_llm backend at it:
#   export LOCAL_LLM_ENGINE_URL="http://127.0.0.1:${PORT:-40080}"
#   export LOCAL_LLM_ENGINE_KIND=llama_cpp
#
# Env vars (all optional):
#   LLAMA_CPP_DIR   Where to install/find the llama.cpp release build.
#                   Default: ~/opt/llama.cpp
#   LLAMA_CPP_TAG   GitHub release tag to fetch if llama-server is missing.
#                   Default: latest
#   MODEL_REPO      "<hf-repo>:<quant>" shorthand llama-server resolves via
#                   the HF Hub. Default: the smallest practical instruct
#                   model, ~400MB on disk / ~500MB RSS at 4k context --
#                   good for a quick API smoke test.
#                   Default: bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
#   PORT            Port to bind. Default: 40080 (this project's local_llm
#                   convention -- deliberately NOT llama-server's own
#                   classic default of 8080, which collides with AIPT's own
#                   `gateway` service; see engine_adapter.py's
#                   DEFAULT_ENGINE_URL). Pick a different free port with
#                   `ss -tlnp | grep <port>` first if 40080 is already used
#                   on your box.
#   HOST            Bind address. Default: 127.0.0.1
#   CTX_SIZE        -c context size passed to llama-server. Default: 4096
#
# Memory rule of thumb (see AIPT DESIGN.md's local_llm section): resident
# memory once loaded is roughly the GGUF file size plus a few hundred MB
# for the KV cache/runtime, e.g. ~500MB for the 0.5B Q4_K_M default here.

set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/opt/llama.cpp}"
LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-latest}"
MODEL_REPO="${MODEL_REPO:-bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M}"
PORT="${PORT:-40080}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-4096}"

mkdir -p "$LLAMA_CPP_DIR"

find_server_bin() {
    find "$LLAMA_CPP_DIR" -maxdepth 2 -type f -name "llama-server" 2>/dev/null | head -1
}

SERVER_BIN="$(find_server_bin)"

if [[ -z "$SERVER_BIN" ]]; then
    echo "== llama-server not found under $LLAMA_CPP_DIR -- downloading a release =="

    if [[ "$LLAMA_CPP_TAG" == "latest" ]]; then
        RELEASE_JSON_URL="https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1"
    else
        RELEASE_JSON_URL="https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${LLAMA_CPP_TAG}"
    fi

    TAG="$(curl -sL "$RELEASE_JSON_URL" | python3 -c '
import json, sys
d = json.load(sys.stdin)
if isinstance(d, list):
    d = d[0]
print(d["tag_name"])
')"

    ARCHIVE="llama-${TAG}-bin-ubuntu-x64.tar.gz"
    URL="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/${ARCHIVE}"

    echo "Fetching ${URL}"
    TMP_TAR="$(mktemp --suffix=.tar.gz)"
    curl -sL -o "$TMP_TAR" "$URL"
    mkdir -p "$LLAMA_CPP_DIR/$TAG"
    tar xzf "$TMP_TAR" -C "$LLAMA_CPP_DIR/$TAG"
    rm -f "$TMP_TAR"

    SERVER_BIN="$(find_server_bin)"
    if [[ -z "$SERVER_BIN" ]]; then
        echo "ERROR: llama-server not found after extracting ${ARCHIVE}" >&2
        exit 1
    fi
fi

BIN_DIR="$(dirname "$SERVER_BIN")"
echo "== Using llama-server: $SERVER_BIN =="
echo "== Model (pulled from HF Hub on first run, cached after): $MODEL_REPO =="
echo "== Listening on http://${HOST}:${PORT} =="
echo "   export LOCAL_LLM_ENGINE_URL=\"http://${HOST}:${PORT}\""
echo "   export LOCAL_LLM_ENGINE_KIND=llama_cpp"

exec env LD_LIBRARY_PATH="$BIN_DIR" "$SERVER_BIN" \
    -hf "$MODEL_REPO" \
    --host "$HOST" \
    --port "$PORT" \
    -c "$CTX_SIZE"
