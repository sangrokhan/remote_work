import pytest
from pathlib import Path

YANG_DIR = str(Path(__file__).parent.parent / "data" / "yang")

@pytest.fixture
def yang_dir():
    return YANG_DIR


@pytest.fixture(scope="session")
def loaded_store(tmp_path_factory, yang_dir):
    from indexer.store import SchemaStore
    db = str(tmp_path_factory.mktemp("db") / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    return store
