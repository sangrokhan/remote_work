import tools
from tools.builder import build_edit_config, build_get_config, build_delete_config, validate_edit_config
from lxml import etree

NETCONF_NS = "urn:ietf:params:xml:ns:netconf:base:1.0"
NC = f"{{{NETCONF_NS}}}"
IF_NS = "urn:ietf:params:xml:ns:yang:ietf-interfaces"


def _mtu_id(loaded_store):
    n = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    assert n is not None
    return n.node_id


def test_build_edit_config_merge(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = build_edit_config(_mtu_id(loaded_store), {"name": "eth0"}, "1500")
    assert result["xml"] is not None
    root = etree.fromstring(result["xml"].encode())
    assert root.tag == f"{NC}rpc"
    mtu_els = root.findall(f".//{{{IF_NS}}}mtu")
    assert mtu_els and mtu_els[0].text == "1500"


def test_build_edit_config_delete(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = build_edit_config(_mtu_id(loaded_store), {"name": "eth0"}, None, operation="delete")
    assert result["xml"] is not None
    root = etree.fromstring(result["xml"].encode())
    mtu_els = root.findall(f".//{{{IF_NS}}}mtu")
    assert mtu_els
    assert mtu_els[0].get(f"{NC}operation") == "delete"


def test_build_get_config(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = build_get_config(_mtu_id(loaded_store), {"name": "eth0"})
    assert result["xml"] is not None
    root = etree.fromstring(result["xml"].encode())
    assert root.tag == f"{NC}rpc"
    get_cfg = root.find(f"{NC}get-config")
    assert get_cfg is not None
    filt = get_cfg.find(f"{NC}filter")
    assert filt is not None


def test_build_delete_config_startup(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = build_delete_config("startup")
    assert result["xml"] is not None
    root = etree.fromstring(result["xml"].encode())
    assert root.tag == f"{NC}rpc"
    del_cfg = root.find(f"{NC}delete-config")
    assert del_cfg is not None


def test_build_delete_config_running_rejected(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = build_delete_config("running")
    assert result["xml"] is None
    assert "error" in result
    assert "running" in result["error"].lower()


def test_validate_edit_config_valid(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = build_edit_config(_mtu_id(loaded_store), {"name": "eth0"}, "1500")
    validation = validate_edit_config(result["xml"])
    assert validation["valid"] is True


def test_validate_edit_config_invalid_xml(loaded_store):
    tools.init_store_from_instance(loaded_store)
    result = validate_edit_config("<broken xml")
    assert result["valid"] is False
    assert "error" in result
