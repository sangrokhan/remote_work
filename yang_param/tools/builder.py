from __future__ import annotations
from lxml import etree
from tools import get_store
from tools.keys import get_path_to_leaf

NETCONF_NS = "urn:ietf:params:xml:ns:netconf:base:1.0"
NC = f"{{{NETCONF_NS}}}"
_ALLOWED_DELETE = ("startup", "candidate")


def _build_hierarchy(
    path_nodes: list[dict],
    key_values: dict,
    store,
    operation: str | None,
) -> etree._Element | None:
    """Build nested XML elements from root-to-target path.

    Returns the root element of the built subtree.
    If operation is given, it is applied to the last (target) element.
    """
    if not path_nodes:
        return None

    root_el = None
    parent_el = None
    last_idx = len(path_nodes) - 1

    for i, node_dict in enumerate(path_nodes):
        r = store.get_by_id(node_dict["node_id"])
        if not r:
            continue
        ns = r.namespace
        tag = f"{{{ns}}}{r.name}"

        if parent_el is None:
            el = etree.Element(tag)
            root_el = el
        else:
            el = etree.SubElement(parent_el, tag)

        # Insert list key children immediately after the list element
        if r.node_kind == "list":
            for key_name in r.keys:
                key_leaf = store.get_by_path(f"{r.schema_path}/{key_name}")
                key_ns = key_leaf.namespace if key_leaf else ns
                key_el = etree.SubElement(el, f"{{{key_ns}}}{key_name}")
                key_el.text = key_values.get(key_name, "")

        # Apply operation attribute on the target (last) element
        if operation and i == last_idx:
            el.set(f"{NC}operation", operation)

        parent_el = el

    return root_el


def build_edit_config(
    target_node_id: str,
    key_values: dict[str, str],
    value: str | None = None,
    operation: str = "merge",
    datastore: str = "running",
) -> dict:
    store = get_store()
    target = store.get_by_id(target_node_id)
    if not target:
        return {"xml": None, "error": f"Node not found: {target_node_id}"}

    path_result = get_path_to_leaf(target_node_id)
    path_nodes = path_result["path"]

    # Build RPC envelope
    rpc = etree.Element(f"{NC}rpc", nsmap={"nc": NETCONF_NS})
    edit = etree.SubElement(rpc, f"{NC}edit-config")
    tgt = etree.SubElement(edit, f"{NC}target")
    etree.SubElement(tgt, f"{NC}{datastore}")
    config_el = etree.SubElement(edit, f"{NC}config")

    # For delete: pass operation into hierarchy so it lands on the leaf.
    # For merge/others: set value on the leaf after building.
    hier_op = operation if operation == "delete" else None
    content = _build_hierarchy(path_nodes, key_values, store, hier_op)
    if content is not None:
        config_el.append(content)

    if operation != "delete" and value is not None:
        leaf_el = config_el.find(f".//{{{target.namespace}}}{target.name}")
        if leaf_el is not None:
            leaf_el.text = value

    xml_str = etree.tostring(rpc, pretty_print=True, encoding="unicode")
    return {"xml": xml_str, "operation": operation, "datastore": datastore}


def build_get_config(
    target_node_id: str,
    key_values: dict[str, str] | None = None,
    datastore: str = "running",
) -> dict:
    store = get_store()
    target = store.get_by_id(target_node_id)
    if not target:
        return {"xml": None, "error": f"Node not found: {target_node_id}"}

    path_result = get_path_to_leaf(target_node_id)
    path_nodes = path_result["path"]

    rpc = etree.Element(f"{NC}rpc", nsmap={"nc": NETCONF_NS})
    get_cfg = etree.SubElement(rpc, f"{NC}get-config")
    src = etree.SubElement(get_cfg, f"{NC}source")
    etree.SubElement(src, f"{NC}{datastore}")
    filter_el = etree.SubElement(get_cfg, f"{NC}filter", type="subtree")

    content = _build_hierarchy(path_nodes, key_values or {}, store, None)
    if content is not None:
        filter_el.append(content)

    xml_str = etree.tostring(rpc, pretty_print=True, encoding="unicode")
    return {"xml": xml_str, "datastore": datastore}


def build_delete_config(datastore: str) -> dict:
    if datastore == "running":
        return {"xml": None, "error": "Deleting running datastore is not allowed"}
    if datastore not in _ALLOWED_DELETE:
        return {"xml": None, "error": f"datastore must be one of {_ALLOWED_DELETE}"}

    rpc = etree.Element(f"{NC}rpc", nsmap={"nc": NETCONF_NS})
    del_cfg = etree.SubElement(rpc, f"{NC}delete-config")
    tgt = etree.SubElement(del_cfg, f"{NC}target")
    etree.SubElement(tgt, f"{NC}{datastore}")

    xml_str = etree.tostring(rpc, pretty_print=True, encoding="unicode")
    return {"xml": xml_str, "datastore": datastore}


def validate_edit_config(xml: str) -> dict:
    try:
        root = etree.fromstring(xml.encode())
        valid_tags = {
            f"{NC}rpc",
            f"{NC}edit-config",
            f"{NC}get-config",
            f"{NC}delete-config",
        }
        if root.tag not in valid_tags:
            return {"valid": False, "error": f"Unexpected root tag: {root.tag}"}
        return {"valid": True}
    except etree.XMLSyntaxError as e:
        return {"valid": False, "error": str(e)}
