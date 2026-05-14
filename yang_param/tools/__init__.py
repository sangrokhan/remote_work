from __future__ import annotations
from indexer.store import SchemaStore

_store: SchemaStore | None = None


def init_store(yang_dir: str, db_path: str) -> None:
    global _store
    _store = SchemaStore(db_path)
    _store.build(yang_dir)


def init_store_from_instance(store: SchemaStore) -> None:
    global _store
    _store = store


def get_store() -> SchemaStore:
    if _store is None:
        raise RuntimeError("Store not initialized. Call init_store() first.")
    return _store
