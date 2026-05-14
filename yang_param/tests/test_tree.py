import tools
from tools.tree import get_node, get_children, get_ancestors


def test_get_node_by_path(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = get_node("/ietf-interfaces:interfaces/interface/mtu")
    assert result["node"] is not None
    assert result["node"]["name"] == "mtu"
    assert result["node"]["node_kind"] == "leaf"


def test_get_node_by_id(loaded_store):
    tools.init_store_from_instance(loaded_store)
    # First find the node_id via path
    by_path = get_node("/ietf-interfaces:interfaces/interface/mtu")
    node_id = by_path["node"]["node_id"]
    result = get_node(node_id)
    assert result["node"]["name"] == "mtu"


def test_get_node_not_found(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = get_node("/nonexistent:path")
    assert result["node"] is None
    assert "error" in result


def test_get_children(loaded_store):
    tools.init_store_from_instance(loaded_store)
    # Get the 'interfaces' container node_id
    interfaces = get_node("/ietf-interfaces:interfaces")
    node_id = interfaces["node"]["node_id"]
    result = get_children(node_id)
    assert "children" in result
    assert any(c["name"] == "interface" for c in result["children"])


def test_get_ancestors(loaded_store):
    tools.init_store_from_instance(loaded_store)
    mtu = get_node("/ietf-interfaces:interfaces/interface/mtu")
    node_id = mtu["node"]["node_id"]
    result = get_ancestors(node_id)
    names = [a["name"] for a in result["ancestors"]]
    assert "interfaces" in names
    assert "interface" in names
    # Root-first order: interfaces before interface
    assert names.index("interfaces") < names.index("interface")
