#!/usr/bin/env python3
"""외부 임베딩 서버 연결을 검증한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib import error, request


def _normalize_url(raw_url: str) -> str:
    base = raw_url.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an OpenAI-compatible embedding server."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ENCODER_URL") or os.environ.get("EMBEDDING_URL", ""),
        help="Embedding server base URL or /embeddings URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ENCODER_MODEL")
        or os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5"),
        help="Model name passed in the request payload",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ENCODER_API_KEY")
        or os.environ.get("EMBEDDING_API_KEY", ""),
        help="Optional Bearer token",
    )
    parser.add_argument(
        "--text",
        default="Samsung LTE parameter reference example",
        help="Sample text to embed",
    )
    args = parser.parse_args()

    if not args.url:
        print("Missing embedding URL. Set ENCODER_URL or EMBEDDING_URL.", file=sys.stderr)
        return 2

    url = _normalize_url(args.url)
    payload = json.dumps({"model": args.model, "input": [args.text]}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    req = request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error: {exc.code}", file=sys.stderr)
        print(detail, file=sys.stderr)
        return 1
    except error.URLError as exc:
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        return 1

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print("Invalid JSON response", file=sys.stderr)
        print(body, file=sys.stderr)
        return 1

    embeddings = data.get("data")
    if not isinstance(embeddings, list):
        embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        print("Response does not contain embeddings", file=sys.stderr)
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    first = embeddings[0]
    if isinstance(first, dict):
        vector = first.get("embedding")
    else:
        vector = first
    if not isinstance(vector, list) or not vector:
        print("Embedding vector is missing or empty", file=sys.stderr)
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print("Embedding server OK")
    print(f"URL: {url}")
    print(f"Model: {args.model}")
    print(f"Vector dimension: {len(vector)}")
    print(f"Sample prefix: {vector[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
