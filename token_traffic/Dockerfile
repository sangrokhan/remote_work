# Base image — override for a registry mirror, e.g.
#   --build-arg BASE_IMAGE=registry.example.com/python:3.12-slim
ARG BASE_IMAGE=python:3.12-slim

# The netlink congestion-window helper is compiled here and copied into the runtime
# image, so a C toolchain never ships to production. gcc alone is not enough on slim --
# it has no libc headers, and the build fails on <arpa/inet.h> -- hence libc6-dev.
FROM ${BASE_IMAGE} AS native
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i "s|https\?://deb.debian.org|$APT_MIRROR|g" \
        /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list 2>/dev/null || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY native/cwnd_monitor.c .
RUN cc -O2 -Wall -o cwnd_monitor cwnd_monitor.c

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
# see docker-compose.yml. Where raw sockets are unavailable, core.capture
# reports it and the run proceeds without a pcap).
#
# ethtool and iproute2, so a capture can turn segmentation offload off for its own
# window. iproute2 is not optional here: TRAFFIC_PCAP_IFACE defaults to tcpdump's
# `any`, which ethtool cannot address, so the real device has to come from
# `ip route get` before anything can be configured on it. Under
# Docker the capture runs in the container's network namespace, so the interface that
# matters is the container's eth0 -- changing the host NIC would not affect anything a
# containerized tcpdump can see. It needs NET_ADMIN, which docker-compose.yml already
# grants. Off unless TRAFFIC_PCAP_NO_OFFLOAD says otherwise, because segmenting in
# software changes the timings this package measures.
#
# The congestion monitor needs no package and no capability at run time: sock_diag
# reports the container's own sockets to its own uid. A sandbox that blocks
# AF_NETLINK (gVisor, a restrictive seccomp profile) makes core.cwnd report itself
# unavailable and the run proceeds without the samples.
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i "s|https\?://deb.debian.org|$APT_MIRROR|g" \
        /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list 2>/dev/null || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends tcpdump ethtool iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Prebuilt, not built on first use: an image that discovers it has no compiler halfway
# through a paid run is an image that reports a missing column as a failure.
COPY --from=native /build/cwnd_monitor native/cwnd_monitor

ENV PORT=8080
EXPOSE 8080

# Two hosts, both key-authenticated over the public internet: the Gemini
# *Developer* API (GEMINI_API_KEY) and the OpenAI API (OPENAI_API_KEY). No ADC,
# no metadata server, no Firestore — run history is JSON under /app/data.
# Set TRAFFIC_MOCK=1 to run with no keys and no quota (synthetic data).
# One worker keeps the single-threaded socket byte counter accurate.
CMD exec gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 4 --timeout 300 core.app:app
