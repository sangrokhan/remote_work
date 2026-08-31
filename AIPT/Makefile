# AIPT Makefile -- thin wrapper around `docker compose` that guarantees
# .env exists (created from .env.example when missing, see
# scripts/ensure_env.sh) before compose ever reads it for ${VAR:-default}
# substitution. Plain `docker compose up --build` still works fine if you
# already ran scripts/ensure_env.sh or have a .env of your own -- these
# targets just make "I forgot to cp .env.example .env" impossible.

.PHONY: env up build down logs ps

# .env is a real target (not .PHONY): make only (re)runs ensure_env.sh when
# .env doesn't exist yet, and never touches an existing one.
.env:
	@./scripts/ensure_env.sh

env: .env

up: .env
	docker compose up --build

build: .env
	docker compose build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps
