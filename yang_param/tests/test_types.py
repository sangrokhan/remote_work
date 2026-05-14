import tools
from tools.types import get_type_info, validate_value, resolve_identityref


def _get_node_id(loaded_store, path):
    node = loaded_store.get_by_path(path)
    assert node is not None, f"Node not found: {path}"
    return node.node_id


def test_get_type_info_uint16(loaded_store):
    tools.init_store_from_instance(loaded_store)
    nid = _get_node_id(loaded_store, "/ietf-interfaces:interfaces/interface/mtu")
    result = get_type_info(nid)
    assert result["type"]["base"] == "uint16"
    assert "range" in result["type"]


def test_get_type_info_boolean(loaded_store):
    tools.init_store_from_instance(loaded_store)
    nid = _get_node_id(loaded_store, "/ietf-interfaces:interfaces/interface/enabled")
    result = get_type_info(nid)
    assert result["type"]["base"] == "boolean"


def test_validate_value_valid_uint16(loaded_store):
    tools.init_store_from_instance(loaded_store)
    nid = _get_node_id(loaded_store, "/ietf-interfaces:interfaces/interface/mtu")
    result = validate_value(nid, "1500")
    assert result["valid"] is True


def test_validate_value_out_of_range(loaded_store):
    tools.init_store_from_instance(loaded_store)
    nid = _get_node_id(loaded_store, "/ietf-interfaces:interfaces/interface/mtu")
    result = validate_value(nid, "50")  # below min 68
    assert result["valid"] is False
    assert "error" in result


def test_validate_value_boolean_true(loaded_store):
    tools.init_store_from_instance(loaded_store)
    nid = _get_node_id(loaded_store, "/ietf-interfaces:interfaces/interface/enabled")
    assert validate_value(nid, "true")["valid"] is True
    assert validate_value(nid, "maybe")["valid"] is False


def test_validate_value_not_integer(loaded_store):
    tools.init_store_from_instance(loaded_store)
    nid = _get_node_id(loaded_store, "/ietf-interfaces:interfaces/interface/mtu")
    result = validate_value(nid, "notanumber")
    assert result["valid"] is False


def test_resolve_identityref(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = resolve_identityref("some-type", "some-value")
    assert "type" in result
    assert "value" in result
    assert "candidates" in result
