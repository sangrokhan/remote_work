# Docs

- [`call-flow.md`](call-flow.md) — the six arms, what each one puts on the wire, and
  where the numbers come from.
- [`interactions-api-fields.md`](interactions-api-fields.md) — what the Interactions
  API actually stores, and the measured cost of `store:true`.
- [`devapi-endpoints.md`](devapi-endpoints.md) — the Developer API endpoints in use.

## Historical render

![Gemini Traffic Experiment UI](gemini-ui-render.png)

`gemini-ui-render.png` is a **June 2026 render and no longer describes the app.** It
predates the current experiment: it shows a Vertex endpoint, a Firestore history
store, and a two-way "stateless vs delta" comparison. Today there is one host (the
Gemini Developer API, API-key auth), no Firestore, and six arms — `stateless`,
`nocontext`, `cached`, `interaction`, `interaction_inline`, `interaction_stateless` —
each reporting wire bytes in both directions and five per-turn latency marks. The
screenshot is kept only as a record of what the UI used to look like; run the app for
the current one.
