import tools
from tools.explore import list_modules, search_nodes, find_leaf


def test_list_modules(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = list_modules()
    assert "modules" in result
    assert "ietf-interfaces" in result["modules"]


def test_search_nodes_by_keyword(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = search_nodes("mtu")
    assert "nodes" in result
    assert any(n["name"] == "mtu" for n in result["nodes"])


def test_search_nodes_by_kind(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = search_nodes("interface", kind="list")
    assert all(n["node_kind"] == "list" for n in result["nodes"])


def test_find_leaf(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = find_leaf("mtu")
    assert result["node"] is not None
    assert result["node"]["name"] == "mtu"


def test_find_leaf_with_parent_hint(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = find_leaf("name", parent_hint="interface")
    assert result["node"] is not None
    assert "interface" in result["node"]["schema_path"]


def test_find_leaf_not_found(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = find_leaf("nonexistent_leaf_xyz")
    assert result["node"] is None
    assert "error" in result
