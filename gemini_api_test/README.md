# Gemini API Traffic Experiment (Vertex AI + Cloud Run)

Proves stateless full-history resend costs far more tokens + network traffic than
delta-only sending. Calls run against **Vertex AI**; run history is written to
**Firestore** (and local JSON). See [`PROJECT_GOAL.md`](PROJECT_GOAL.md) for why.

## Auth model (important)

- **Vertex AI** and **Firestore** both use **ADC** (Application Default
  Credentials) — no API key.
- On **Cloud Run**: the attached **service account** + metadata server supply
  creds automatically. Nothing to set besides project/location.
- **Locally**: `gcloud auth application-default login` (or a service-account JSON
  via `GOOGLE_APPLICATION_CREDENTIALS`).
- **No creds / no quota?** `GEMINI_MOCK=1` runs synthetic data end-to-end.

## Quick local run (mock — no GCP needed)

```bash
pip install -r requirements.txt
GEMINI_MOCK=1 python app.py        # http://localhost:8080
```

## Local run against real Vertex

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project
export VERTEX_LOCATION=us-central1
python app.py                      # http://localhost:8080
```

Open the page → set turns/message-size → **Start experiment** → charts + tables.

## Docker

```bash
cp .env.example .env                       # fill in, or just set GEMINI_MOCK=1
docker compose up --build                  # http://localhost:8080 (auto-loads .env)
```

See [`.env.example`](.env.example) for all parameters (Vertex, Firestore,
capture, inspector). `.env` is gitignored; compose auto-loads it.

### Build behind a mirror (offline / corporate registry)

Override the base image registry and pip index via build args (defaults = Docker
Hub + pypi.org). With compose, set them in `.env`:

```bash
BASE_IMAGE=registry.example.com/python:3.12-slim
PIP_INDEX_URL=https://mirror.example.com/pypi/simple
PIP_TRUSTED_HOST=mirror.example.com
```

Or with plain `docker build`:

```bash
docker build \
  --build-arg BASE_IMAGE=registry.example.com/python:3.12-slim \
  --build-arg PIP_INDEX_URL=https://mirror.example.com/pypi/simple \
  --build-arg PIP_TRUSTED_HOST=mirror.example.com \
  -t gemini-traffic .
```

For non-mock local Docker, mount creds (uncomment the gcloud volume in
`docker-compose.yml`, or set `GOOGLE_APPLICATION_CREDENTIALS`).

## Packet capture (optional)

Tick **capture packets** before **Start** to run `tcpdump` around the experiment,
filtered to the Vertex host on tcp/443. When the run finishes a **⬇ download
.pcap** link appears next to Start (open in Wireshark). The TLS payload is
encrypted — packet **sizes + timing** are the real-traffic proof.

Requires `tcpdump` + raw-socket capability:

- **Local:** run with privileges to open raw sockets (e.g. `sudo`, or
  `setcap cap_net_raw+ep $(which tcpdump)`).
- **Docker:** `docker compose up` already adds `NET_RAW` (see `cap_add` in
  `docker-compose.yml`).
- **Cloud Run:** NOT supported (gVisor sandbox has no raw sockets). The checkbox
  shows as unavailable there and the experiment runs normally without a pcap.
- **Mock mode:** no real traffic leaves the process, so the pcap is empty — the UI
  reports "no packets captured". Use a live Vertex run for a meaningful pcap.

## Endpoint inspector (MCP / A2A / any URL)

A pcap can't show HTTP headers — TLS encrypts them. The **Endpoint inspector**
panel captures them at the app layer instead: enter a method + URL (a local MCP
server, an A2A agent, the Gemini REST endpoint), optional headers/body, and get
back:

- request + **response headers** (plaintext),
- status, real wire bytes, payload sizes, elapsed ms,
- **protocol hints** — detected MCP / A2A / JSON-RPC / SSE markers,
- optional request/response **bodies** (tick *include bodies*),
- a downloadable **transcript** JSON.

This is where protocol-specific headers (`Mcp-Session-Id`, `text/event-stream`,
JSON-RPC, A2A agent-card) show up. Note: the Gemini call itself carries no MCP/A2A
headers — point the inspector at the MCP/A2A hop to see those. Works on Cloud Run
(no NET_RAW needed).

**SSRF guard:** the inspector refuses targets resolving to private / loopback /
reserved IPs by default. Tick *allow local/private targets* to reach a localhost
MCP server. The link-local range (`169.254.0.0/16`, incl. the GCP metadata server
`169.254.169.254`) is **always** refused. Redirects are not followed.

## Deploy to Cloud Run

```bash
PROJECT=your-project
REGION=us-central1
SA=gemini-traffic@$PROJECT.iam.gserviceaccount.com

# Service account needs Vertex + Firestore access:
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:$SA" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:$SA" --role="roles/datastore.user"   # Firestore

# Build + deploy (source build; or push your own image)
gcloud run deploy gemini-traffic \
  --source . \
  --region=$REGION \
  --service-account=$SA \
  --set-env-vars=GOOGLE_CLOUD_PROJECT=$PROJECT,VERTEX_LOCATION=$REGION,FIRESTORE_COLLECTION=gemini_runs \
  --allow-unauthenticated
```

Cloud Run injects `PORT` (8080); the app binds it via gunicorn. Firestore is in
your project already, so history persists there (the local JSON copy is ephemeral
on Cloud Run — that's expected).

## Pre-merge gate

Run before merging — exits non-zero on any failure:

```bash
make preflight        # or: ./preflight.sh
```

Steps: unit tests → docker build → start mock container → smoke `/run` and
`/inspect` (the `/inspect` check asserts `wire_recv > 0`, guarding the response
byte counter). No GCP creds / quota needed. `make test` runs unit tests only.

## Tests

```bash
python -m unittest discover tests      # pure metric math, no network
```

## What's measured per call

- Tokens — from Vertex response `usageMetadata`.
- Wire bytes — HTTP bytes crossing the TLS stream (headers + content-encoded body;
  real transferred size, not raw ciphertext), cross-checked vs decoded JSON payload
  size. For true packet/ciphertext sizes use the optional pcap.
- Everything tagged `mode = stateless | delta`. One service account is enough —
  attribution is per-request in our own code, not from billing.

## Config (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `GOOGLE_CLOUD_PROJECT` | — | GCP project for Vertex + Firestore |
| `VERTEX_LOCATION` | `us-central1` | Vertex region (`global` allowed) |
| `GEMINI_MOCK` | `0` | `1` = synthetic, no GCP/quota |
| `FIRESTORE_COLLECTION` | `gemini_runs` | run-history collection |
| `FIRESTORE_DATABASE` | `(default)` | Firestore database id |
| `FIRESTORE_DISABLE` | `0` | `1` = skip Firestore, JSON only |
| `GEMINI_PRICE_PER_TOKEN` | `0.0000001` | cost-estimate rate (USD/token) |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | local SA key path (ADC) |
| `PORT` | `8080` | server port (Cloud Run sets this) |
| `GEMINI_DATA_DIR` | `data/runs` | local JSON dir |
| `PCAP_DIR` | `data/pcaps` | captured .pcap output dir |
| `PCAP_IFACE` | `any` | tcpdump capture interface |
| `PCAP_DISABLE` | `0` | `1` = hide/disable packet capture |
| `TRANSCRIPT_DIR` | `data/transcripts` | inspector transcript output dir |
| `INSPECT_MAX_BODY` | `1048576` | max response-body bytes captured (1 MiB) |
