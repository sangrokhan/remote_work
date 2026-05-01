#!/usr/bin/env python3
"""SPAR API smoke test.

Usage:
    python scripts/test_api.py              # default http://localhost:9000
    python scripts/test_api.py --url http://host:9000
"""

import argparse
import json
import sys

import httpx

CASES = [
    {
        "name": "basic query",
        "payload": {"query": "What is the default value of maxTxPower parameter?"},
    },
    {
        "name": "product filter",
        "payload": {"query": "How to configure RACH in NR?", "product": "NR", "release": "v7.1"},
    },
    {
        "name": "top_k override",
        "payload": {"query": "List all HO-related alarms", "top_k": 5},
    },
]


def run(base_url: str) -> None:
    with httpx.Client(base_url=base_url, timeout=30) as client:
        # health check
        r = client.get("/health")
        r.raise_for_status()
        print(f"[health] {r.json()}\n")

        # query cases
        for case in CASES:
            print(f"[{case['name']}]")
            r = client.post("/query", json=case["payload"])
            if r.status_code != 200:
                print(f"  ERROR {r.status_code}: {r.text}\n")
                continue
            data = r.json()
            print(f"  request_id : {data['request_id']}")
            print(f"  answer     : {data['answer']}")
            print(f"  sources    : {len(data['sources'])} chunk(s)")
            print(f"  latency    : {data['latency_ms']} ms")
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:9000")
    args = parser.parse_args()

    try:
        run(args.url)
    except httpx.ConnectError:
        print(f"ERROR: cannot connect to {args.url}")
        print("Start server first:  uvicorn spar.api.app:app --reload --port 9000")
        sys.exit(1)


if __name__ == "__main__":
    main()
