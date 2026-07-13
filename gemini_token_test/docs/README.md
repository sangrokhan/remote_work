# Docs

## Rendered web UI

![Gemini Traffic Experiment UI](gemini-ui-render.png)

`gemini-ui-render.png` — full-page render of the live Flask dashboard after one
mock run (10 turns, 500 chars). Shows:

- **Status bar** — MOCK badge, Vertex endpoint, history store (local JSON / Firestore).
- **Result summary** — stateless vs delta tokens, token & wire ratios, cost estimate.
- **Cumulative tokens** chart — stateless (red) curves upward (O(N²)) vs delta (green) linear.
- **Cumulative wire bytes** chart — same divergence on real on-wire bytes.
- **Per-turn detail** table and **run history** (background collection) panel.
- **`interaction_stateless`** arm — `POST /interactions` with `store: false`,
  no `previous_interaction_id`; the client resends the whole conversation as
  `input` every turn. Live-verified 2026-07-13: `wire_sent` and `input_tokens`
  grow turn over turn and track `stateless` within noise. Its gap to
  `interaction` is exactly what `previous_interaction_id` buys.

Regenerate: run `GEMINI_MOCK=1 python app.py`, open the page, click **Start
experiment**, screenshot the full page.

See also: [`../PROJECT_GOAL.md`](../PROJECT_GOAL.md),
[design spec](superpowers/specs/2026-06-26-gemini-traffic-test-design.md).
