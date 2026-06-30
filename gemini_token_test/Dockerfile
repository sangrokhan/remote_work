# Base image — override for a registry mirror, e.g.
#   --build-arg BASE_IMAGE=registry.example.com/python:3.12-slim
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# pip index mirror — override for an internal/regional PyPI, e.g.
#   --build-arg PIP_INDEX_URL=https://mirror.example.com/pypi/simple
#   --build-arg PIP_TRUSTED_HOST=mirror.example.com
ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

# apt mirror — override the Debian package server (scheme+host, no trailing
# path), e.g. --build-arg APT_MIRROR=http://mirror.example.com
# Replaces deb.debian.org; the /debian and /debian-security paths are kept.
ARG APT_MIRROR=

# tcpdump for optional packet capture (needs --cap-add=NET_RAW at run time;
# not available on Cloud Run — capture auto-disables there).
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i "s|https\?://deb.debian.org|$APT_MIRROR|g" \
        /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list 2>/dev/null || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends tcpdump \
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
