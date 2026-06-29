FROM python:3.12-slim

WORKDIR /app

# tcpdump for optional packet capture (needs --cap-add=NET_RAW at run time;
# not available on Cloud Run — capture auto-disables there).
RUN apt-get update && apt-get install -y --no-install-recommends tcpdump \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# Cloud Run: bind to $PORT, 0.0.0.0. Auth (Vertex + Firestore) via the service
# account / metadata server (ADC) — no key in env. Set GEMINI_MOCK=1 to run
# without GCP creds (synthetic data). One worker keeps the single-threaded socket
# byte counter accurate.
CMD exec gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 4 --timeout 300 app:app
