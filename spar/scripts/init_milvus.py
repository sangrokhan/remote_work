#!/usr/bin/env python3
"""Initialize Milvus collections for SPAR.

Usage:
    python scripts/init_milvus.py              # create (skip if exists)
    python scripts/init_milvus.py --reset      # drop + recreate all
    python scripts/init_milvus.py --list       # list existing collections
"""

import argparse
import sys
from pathlib import Path

# repo root를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spar.retrieval.milvus_client import DOC_TYPES, SparMilvusClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Init SPAR Milvus collections")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all collections")
    parser.add_argument("--list", action="store_true", help="List existing collections and exit")
    args = parser.parse_args()

    with SparMilvusClient() as client:
        if args.list:
            cols = client.list_collections()
            if cols:
                print("Existing collections:")
                for c in cols:
                    print(f"  {c}")
            else:
                print("No SPAR collections found.")
            return

        if args.reset:
            print("WARNING: dropping all existing SPAR collections...")
            for dt in DOC_TYPES:
                if client.collection_exists(dt):
                    from pymilvus import utility
                    utility.drop_collection(client.collection_name(dt))
                    print(f"  dropped: {client.collection_name(dt)}")

        print("Creating collections...")
        for dt in DOC_TYPES:
            col = client.create_collection(dt, drop_if_exists=False)
            print(f"  OK: {col.name}")

        print("\nDone. Collections:")
        for name in client.list_collections():
            print(f"  {name}")


if __name__ == "__main__":
    main()
