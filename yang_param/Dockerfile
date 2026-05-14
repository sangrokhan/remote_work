FROM python:3.12-slim-trixie

ARG APT_MIRROR=deb.debian.org
RUN grep -rl "deb.debian.org" /etc/apt/ | xargs -r sed -i "s|deb.debian.org|${APT_MIRROR}|g" \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential libyang3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
ARG PIP_MIRROR=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url ${PIP_MIRROR} -e .

ENV YANG_DIR=/app/data/yang
ENV YANG_DB=/app/data/schema.db

EXPOSE 8000
CMD ["uvicorn", "server.rest_server:app", "--host", "0.0.0.0", "--port", "8000"]
