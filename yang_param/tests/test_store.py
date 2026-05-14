from indexer.store import SchemaStore


def test_build_and_load(yang_dir, tmp_path):
    db = str(tmp_path / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    assert store.count() > 0


def test_get_by_id(yang_dir, tmp_path):
    db = str(tmp_path / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    results = store.search_by_name("mtu")
    assert results
    node = store.get_by_id(results[0].node_id)
    assert node.name == "mtu"


def test_search_by_name(yang_dir, tmp_path):
    db = str(tmp_path / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    results = store.search_by_name("interface")
    assert any(r.name == "interface" for r in results)


def test_search_by_keyword(yang_dir, tmp_path):
    db = str(tmp_path / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    results = store.search_by_keyword("MTU")
    assert any(r.name == "mtu" for r in results)


def test_get_by_path(yang_dir, tmp_path):
    db = str(tmp_path / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    node = store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    assert node is not None
    assert node.node_kind == "leaf"


def test_list_modules(yang_dir, tmp_path):
    db = str(tmp_path / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    modules = store.list_modules()
    assert "ietf-interfaces" in modules


def test_search_by_keyword_with_kind_filter(yang_dir, tmp_path):
    db = str(tmp_path / "schema.db")
    store = SchemaStore(db)
    store.build(yang_dir)
    results = store.search_by_keyword("interface", kind="list")
    assert all(r.node_kind == "list" for r in results)
