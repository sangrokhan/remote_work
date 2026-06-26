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
export GOOGLE_CLOUD_PROJECT=your-project   # or GEMINI_MOCK=1
docker compose up --build                  # http://localhost:8080
```

For non-mock local Docker, mount creds (uncomment the gcloud volume in
`docker-compose.yml`, or set `GOOGLE_APPLICATION_CREDENTIALS`).

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

## Tests

```bash
python -m unittest discover tests      # pure metric math, no network
```

## What's measured per call

- Tokens — from Vertex response `usageMetadata`.
- Wire bytes — raw bytes through the TLS socket (real on-wire), cross-checked vs
  JSON payload size.
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
