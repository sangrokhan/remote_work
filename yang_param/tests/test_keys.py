import tools
from tools.keys import get_path_to_leaf, get_required_keys, resolve_instance_path


def _mtu_id(loaded_store):
    node = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    assert node is not None
    return node.node_id


def test_get_path_to_leaf_includes_target(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = get_path_to_leaf(_mtu_id(loaded_store))
    paths = [n["schema_path"] for n in result["path"]]
    assert "/ietf-interfaces:interfaces/interface/mtu" in paths


def test_get_path_to_leaf_root_first(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = get_path_to_leaf(_mtu_id(loaded_store))
    paths = [n["schema_path"] for n in result["path"]]
    # interfaces must come before interface, which must come before mtu
    assert paths.index("/ietf-interfaces:interfaces") < paths.index("/ietf-interfaces:interfaces/interface")
    assert paths.index("/ietf-interfaces:interfaces/interface") < paths.index("/ietf-interfaces:interfaces/interface/mtu")


def test_get_required_keys(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = get_required_keys(_mtu_id(loaded_store))
    assert result["target_path"] == "/ietf-interfaces:interfaces/interface/mtu"
    assert len(result["required_keys"]) == 1
    key = result["required_keys"][0]
    assert key["key_name"] == "name"
    assert "list_path" in key
    assert "type" in key


def test_resolve_instance_path_complete(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = resolve_instance_path(_mtu_id(loaded_store), {"name": "eth0"})
    assert result["missing_keys"] == []
    assert result["instance_path"] is not None
    assert "eth0" in result["instance_path"]
    assert "mtu" in result["instance_path"]


def test_resolve_instance_path_missing_key(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = resolve_instance_path(_mtu_id(loaded_store), {})
    assert "name" in result["missing_keys"]
    assert result["instance_path"] is None
