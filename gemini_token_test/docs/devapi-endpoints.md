# Developer API — endpoint formats for the four arms

Host: `https://generativelanguage.googleapis.com/v1beta`. Auth header
`x-goog-api-key: <key>` (this project uses the header, not the `?key=` query
param). Model fixed at `gemini-3.1-flash-lite`.

Sources: the `generateContent`, `caching`, and `interactions` REST references and
the caching guide on ai.google.dev, indexed 2026-07-13. Where a field was
observed live by the probe, it is marked **(measured)**.

---

## 1. `generateContent` — used by `stateless`, `cached`, `nocontext`

```
POST /v1beta/models/gemini-3.1-flash-lite:generateContent
```

### Request body (`GenerateContentRequest`)

| Field | Type | What goes in it |
|-------|------|-----------------|
| `contents[]` | Content[] | The turns. Each `{role: "user"\|"model", parts:[{text}]}`. This is where each arm's history policy lives. |
| `systemInstruction` | Content | `{parts:[{text}]}`. The 12K system prompt. |
| `generationConfig` | object | `maxOutputTokens`, `temperature`, `topP`, `seed`, `stopSequences`, `thinkingConfig`. Not pinned beyond nothing here except the run defaults. |
| `cachedContent` | string | Name of an explicit cache to use: `cachedContents/{id}`. Only the `cached` arm sets it. |

### Response — `usageMetadata`

```json
"usageMetadata": {
  "promptTokenCount": 3050,        // total effective prompt, INCLUDING cached part
  "cachedContentTokenCount": 3000, // the cached slice of the prompt
  "candidatesTokenCount": 40,
  "thoughtsTokenCount": 0,
  "totalTokenCount": 3090
}
```

Note: `promptTokenCount` already includes `cachedContentTokenCount`. Billable
non-cached input = `promptTokenCount - cachedContentTokenCount`.

---

## 2. `cachedContents` — the `cached` arm's setup

Explicit caching is a `generateContent` feature. **It is not available on the
Interactions API** (see §4).

### Create

```
POST /v1beta/cachedContents
```

| Field | Type | What goes in it |
|-------|------|-----------------|
| `model` | string | **`models/gemini-3.1-flash-lite`** — note the `models/` prefix, unlike generateContent's path. Required, immutable. |
| `contents[]` | Content[] | The prefix to cache (history so far). Immutable. |
| `systemInstruction` | Content | The system prompt to cache. |
| `ttl` | duration | `"1800s"`. Input only. Alternatively `expireTime` (RFC3339). |
| `displayName` | string | Optional label, ≤128 chars. |

Response carries `name: "cachedContents/{id}"`, echoed `expireTime`, and a
`usageMetadata.totalTokenCount` for what was cached.

### Use

Pass `cachedContent: "cachedContents/{id}"` in a `generateContent` request.

### Delete

```
DELETE /v1beta/cachedContents/{id}
```

### Minimum token count (this is a hard gate)

Explicit cache creation is rejected below a per-model minimum:

| Model | Min tokens |
| --- | --- |
| Gemini 3.5 Flash | 4096 |
| Gemini 3.1 Pro Preview | 4096 |
| Gemini 2.5 Flash | 2048 |
| Gemini 2.5 Pro | 2048 |

**`gemini-3.1-flash-lite` is not in the published table.** Its minimum is
unknown. The 12K system prompt is ~3,000 tokens. If flash-lite's floor is 4096
(like the other 3.x models), a system-prompt-only cache would be **rejected**,
and the `cached` arm can only cache once the accumulated history pushes the prefix
over the floor. This must be checked with one probe before the arm is trusted —
not assumed. `gemini_client.MIN_CACHE_TOKENS = 2048` is a stale Vertex constant
and is not authoritative here.

---

## 3. `interactions` — the `interaction` arm

Already documented in `interactions-api-fields.md`. Recap of what this arm sends
every turn (all measured):

```
POST /v1beta/interactions
{
  "model": "gemini-3.1-flash-lite",
  "stream": false,
  "store": true,
  "system_instruction": "<the 12K system prompt>",   // re-sent EVERY turn
  "previous_interaction_id": "<id from prev turn>",   // absent on turn 1
  "input": [{"type":"user_input","content":[{"type":"text","text":"<qk>"}]}],
  "generation_config": {"max_output_tokens": N}
}
```

Response usage (non-stream, top-level `usage`):
`total_input_tokens`, `total_cached_tokens`, `total_output_tokens`,
`total_thought_tokens`, `total_tokens`.

---

## 4. Implicit vs explicit caching — the finding that reshapes the arms

From the caching guide (2026-07-07):

> Implicit caching is enabled by default for all Gemini 2.5 and newer models. It
> is supported for both stateful (`previous_interaction_id`) and stateless
> conversation modes. Cost savings pass through automatically. **Explicit caching
> is not supported in the Interactions API.**

Consequences:

1. **`stateless` and `interaction` both get implicit caching for free.** Sending a
   repeated prefix (the 12K system prompt, and history) in a short window can
   produce cache hits with no cache object at all. So `total_cached_tokens` may be
   nonzero on arms that never touched `cachedContents`. This is realistic and
   should be recorded, not suppressed.

2. **The `cached` arm (explicit cache) is a `generateContent`-only construct.** It
   cannot be layered onto `interactions`. So the theoretical "interaction + explicit
   cache" fifth arm the earlier spec floated **does not exist** — the API refuses
   it. Drop that idea.

3. **The comparison is now partly about implicit-vs-explicit, not stateful-vs-
   stateless alone.** The honest framing:
   - `stateless` — full history every turn, implicit caching may help on the wire-repeated prefix.
   - `cached` — explicit cache holds the prefix; only the new question is sent as `contents`.
   - `interaction` — server holds history; system prompt re-sent every turn; implicit caching may help.
   - `nocontext` — new question only; lower bound.

   The wire-bytes axis is unaffected by implicit caching (bytes still cross the
   socket). The token axis is where implicit caching shows up, via
   `cached_tokens`. Keeping the two axes separate (already in the spec) is exactly
   what makes this legible.
