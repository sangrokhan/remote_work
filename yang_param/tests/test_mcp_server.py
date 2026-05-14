import importlib


def test_mcp_server_importable():
    mod = importlib.import_module("server.mcp_server")
    assert hasattr(mod, "app")


def test_tool_list_complete():
    from server.mcp_server import TOOLS
    names = {t["name"] for t in TOOLS}
    required = {
        "list_modules", "search_nodes", "find_leaf",
        "get_node", "get_children", "get_ancestors",
        "get_path_to_leaf", "get_required_keys", "resolve_instance_path",
        "get_type_info", "validate_value", "resolve_identityref",
        "build_edit_config", "build_get_config", "build_delete_config",
        "validate_edit_config",
    }
    assert required <= names, f"Missing tools: {required - names}"


def test_tool_schemas_have_required_fields():
    from server.mcp_server import TOOLS
    for tool in TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        assert isinstance(tool["inputSchema"], dict)
