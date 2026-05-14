from indexer.parser import parse_yang_dir


def test_parse_returns_nodes(yang_dir):
    nodes = list(parse_yang_dir(yang_dir))
    assert len(nodes) > 0


def test_node_has_required_fields(yang_dir):
    nodes = list(parse_yang_dir(yang_dir))
    node = next(n for n in nodes if n["node_kind"] == "leaf" and n["name"] == "mtu")
    assert node["schema_path"] == "/ietf-interfaces:interfaces/interface/mtu"
    assert node["module"] == "ietf-interfaces"
    assert node["node_kind"] == "leaf"
    assert node["config"] is True
    assert "uint16" in node["type_base"]


def test_list_node_has_keys(yang_dir):
    nodes = list(parse_yang_dir(yang_dir))
    lst = next(n for n in nodes if n["node_kind"] == "list")
    assert "name" in lst["keys"]


def test_parent_path_populated(yang_dir):
    nodes = list(parse_yang_dir(yang_dir))
    mtu = next(n for n in nodes if n["name"] == "mtu")
    assert mtu["parent_path"] is not None
    assert "interface" in mtu["parent_path"]


def test_container_has_child_paths(yang_dir):
    nodes = list(parse_yang_dir(yang_dir))
    iface_list = next(
        n for n in nodes if n["name"] == "interface" and n["node_kind"] == "list"
    )
    assert len(iface_list["child_paths"]) > 0


# --- Normalizer tests ---
from indexer.normalizer import normalize, normalize_all, NodeRecord


def test_normalize_produces_node_record(yang_dir):
    from indexer.parser import parse_yang_dir
    raw = next(
        n for n in parse_yang_dir(yang_dir)
        if n["name"] == "mtu"
    )
    record = normalize(raw, path_to_id={})
    assert isinstance(record, NodeRecord)
    assert record.node_id.startswith("ietf-interfaces:")
    assert record.schema_path == "/ietf-interfaces:interfaces/interface/mtu"
    assert record.config is True


def test_node_id_stable(yang_dir):
    from indexer.parser import parse_yang_dir
    raw = next(n for n in parse_yang_dir(yang_dir) if n["name"] == "mtu")
    r1 = normalize(raw, path_to_id={})
    r2 = normalize(raw, path_to_id={})
    assert r1.node_id == r2.node_id


def test_normalize_all_links_parent_child(yang_dir):
    from indexer.parser import parse_yang_dir
    raw_nodes = list(parse_yang_dir(yang_dir))
    records = normalize_all(raw_nodes)
    mtu = next(r for r in records if r.name == "mtu")
    assert mtu.parent_id is not None
    # parent should be the interface list record
    parent = next((r for r in records if r.node_id == mtu.parent_id), None)
    assert parent is not None
    assert parent.name == "interface"
