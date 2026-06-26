# Gemini API Traffic Experiment

Proves stateless full-history resend costs far more tokens + network traffic than
delta-only sending. See [`PROJECT_GOAL.md`](PROJECT_GOAL.md) for the why.

## Run with Docker (recommended)

```bash
export GEMINI_API_KEY=your_key_here      # real key -> live calls
# or, to try without quota:
# export GEMINI_MOCK=1

docker compose up --build
```

Open http://localhost:8000 → set turns/message-size → **Start experiment**.

Runs persist to `./data/runs/` (mounted into the container).

## Run locally (no Docker)

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here      # or export GEMINI_MOCK=1
python app.py
# http://localhost:8000
```

## Mock mode

No key? `GEMINI_MOCK=1` returns synthetic responses with token counts derived from
input length, so the full UI, charts, and persistence work without spending quota.

## Tests

```bash
python -m pytest tests/        # or: python -m unittest discover tests
```

Pure metric-math tests (no network): confirms stateless cumulative tokens grow
quadratically vs delta linear, ratio, and cost math.

## What's measured per call

- Tokens — from response `usageMetadata`.
- Wire bytes — raw bytes through the TLS socket (real on-wire), cross-checked vs
  JSON payload size.
- Everything tagged `mode = stateless | delta`; one API key is enough.

## Config (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `GEMINI_API_KEY` | — | API key; live calls when set |
| `GEMINI_MOCK` | `0` | `1` = synthetic, no quota |
| `GEMINI_PRICE_PER_TOKEN` | `0.0000001` | cost-estimate rate (USD/token) |
| `PORT` | `8000` | server port |
| `GEMINI_DATA_DIR` | `data/runs` | run history dir |
