#!/usr/bin/env bash
# scripts/ensure_env.sh -- make sure AIPT/.env exists before `docker compose`
# reads it for variable substitution.
#
# Why this has to be a *pre*-step and can't live inside a Dockerfile:
# docker compose resolves `${VAR:-default}` substitutions in
# docker-compose.yml from the *host's* .env file (project-root convention)
# before it even starts building any image -- by the time a Dockerfile
# or container entrypoint runs, that substitution has already happened.
# So "copy .env.example to .env if missing" has to run on the host, before
# `docker compose build`/`up`, not inside the build. This script is meant
# to be that one step -- call it directly, or via `make up`/`make build`
# (see Makefile) which already do.
#
# Idempotent: if .env already exists, it is left untouched (never
# overwritten -- a developer's real secrets/overrides always win over the
# template). Only creates .env when it is completely absent.
#
# Usage:
#   ./scripts/ensure_env.sh
#   ./scripts/ensure_env.sh && docker compose up --build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"

if [[ -f "$ENV_FILE" ]]; then
    echo "== .env already exists at $ENV_FILE -- leaving it untouched =="
    exit 0
fi

if [[ ! -f "$ENV_EXAMPLE" ]]; then
    echo "ERROR: neither .env nor .env.example found under $PROJECT_ROOT" >&2
    exit 1
fi

cp "$ENV_EXAMPLE" "$ENV_FILE"
echo "== .env not found -- created from .env.example (defaults: GATEWAY_PROFILE=custom, GATEWAY_DELAY_MS=20ms, etc.) =="
echo "== Edit $ENV_FILE to set GEMINI_API_KEY/OPENAI_API_KEY or override any other default before running docker compose. =="
