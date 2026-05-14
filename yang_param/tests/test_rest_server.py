import tools
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(loaded_store):
    tools.init_store_from_instance(loaded_store)
    from server.rest_server import app
    return TestClient(app)


def _mtu_id(loaded_store):
    n = loaded_store.get_by_path("/ietf-interfaces:interfaces/interface/mtu")
    assert n is not None
    return n.node_id


def test_rest_importable():
    import importlib
    mod = importlib.import_module("server.rest_server")
    assert hasattr(mod, "app")


def test_list_modules(client):
    resp = client.get("/tools/list_modules")
    assert resp.status_code == 200
    assert "ietf-interfaces" in resp.json()["modules"]


def test_search_nodes(client):
    resp = client.get("/tools/search_nodes", params={"keyword": "mtu"})
    assert resp.status_code == 200
    assert any(n["name"] == "mtu" for n in resp.json()["nodes"])


def test_find_leaf(client):
    resp = client.get("/tools/find_leaf", params={"name": "mtu"})
    assert resp.status_code == 200
    assert resp.json()["node"]["name"] == "mtu"


def test_get_node_by_path(client):
    resp = client.get("/tools/get_node", params={"node_id_or_path": "/ietf-interfaces:interfaces/interface/mtu"})
    assert resp.status_code == 200
    assert resp.json()["node"]["name"] == "mtu"


def test_get_required_keys(client, loaded_store):
    resp = client.get("/tools/get_required_keys", params={"node_id": _mtu_id(loaded_store)})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["required_keys"]) == 1
    assert data["required_keys"][0]["key_name"] == "name"


def test_build_edit_config_post(client, loaded_store):
    resp = client.post("/tools/build_edit_config", json={
        "target_node_id": _mtu_id(loaded_store),
        "key_values": {"name": "eth0"},
        "value": "1500"
    })
    assert resp.status_code == 200
    assert resp.json()["xml"] is not None
    assert "1500" in resp.json()["xml"]


def test_build_delete_config_running_rejected(client):
    resp = client.post("/tools/build_delete_config", json={"datastore": "running"})
    assert resp.status_code == 200
    assert resp.json()["xml"] is None
