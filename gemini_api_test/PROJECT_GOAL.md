# Project Goal — Gemini API Traffic Experiment

## Goal

Demonstrate, with self-measured numbers, that **stateless cumulative API usage**
(resending the full conversation history on every turn) costs far more token and
network traffic than a **delta-only** approach (sending only the newest turn) —
and, since Gemini bills on **all** tokens fed to inference, that this directly
inflates spend.

This is a **traffic measurement experiment**, not a chat-quality one. Delta mode
deliberately drops context; we compare only bytes and tokens, never answer
correctness.

## The two modes

| Mode | What each turn sends | Cumulative input tokens |
|------|----------------------|-------------------------|
| **stateless** | full history (turns 1..k) | **O(N²)** — grows every turn |
| **delta** | only the current turn k | **O(N)** — flat per turn |

## Hypothesis

For an N-turn conversation the stateless/delta token ratio rises roughly linearly
with N. By turn 10 stateless already sends several times more tokens per call; the
cumulative gap is large and grows. Charts should show stateless curving upward
(quadratic) while delta stays a straight line.

## How we measure (no account split needed)

Every single call is self-measured and tagged `mode`:

- **Tokens** — from each response's `usageMetadata`
  (`promptTokenCount` / `candidatesTokenCount` / `totalTokenCount`).
- **Wire bytes** — a socket wrapper counts raw bytes sent/received over the TLS
  connection (real on-wire bytes, no tcpdump / NET_ADMIN). Cross-checked against
  JSON payload sizes.

Because attribution is per-request in our own code, **one API key / account is
enough**. We do not depend on Google's aggregate billing dashboard. (Optional
later verification: run the two modes in separate time windows and compare against
the billing page.)

## What the tool shows

- **Start button** + params (turns, message size, model).
- Two cumulative charts: **tokens** and **wire bytes**, stateless vs delta.
- Per-turn detail table and a summary (totals, **ratio**, **cost estimate**).
- **Background collection:** every run is saved to `data/runs/*.json`; the history
  panel aggregates total tokens + traffic per endpoint
  (`generativelanguage.googleapis.com:443`) across all runs.

## The proof

The summary's `token_ratio` and `wire_ratio` (stateless ÷ delta) quantify the
penalty; the charts make the quadratic-vs-linear divergence visible; the cost
estimate converts it to dollars. Together they are the clear proof that
cumulative stateless usage is the bigger-traffic, bigger-cost pattern.

## Run

See `README.md`. TL;DR — Docker: `docker compose up --build` with
`GEMINI_API_KEY` exported (or `GEMINI_MOCK=1` to try it without quota), then open
http://localhost:8000.

## Expected result (mock, defaults: 10 turns, 500 chars)

Stateless cumulative tokens ≈ several× delta; ratio grows with turn count. Re-run
with more turns to widen the gap and watch the curve steepen.
