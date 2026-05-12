import json
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader

from causality_graph.extraction.review_queue import ReviewQueue, ReviewStatus
from causality_graph.fixtures import make_sample_graph
from causality_graph.schema import NodeType

QUEUE_PATH = Path("data/review_queue.jsonl")
QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Causality Graph API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

# Initialize Jinja2 environment directly to avoid Starlette Jinja2Templates caching issues
template_dir = Path(__file__).parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

queue = ReviewQueue(QUEUE_PATH)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    pending = queue.list_pending()
    template = jinja_env.get_template("index.html")
    return HTMLResponse(template.render({
        "request": request,
        "items": pending,
        "count": len(pending),
    }))


@app.post("/approve/{item_id}")
async def approve(item_id: str):
    queue.approve(item_id)
    return RedirectResponse("/", status_code=303)


@app.post("/reject/{item_id}")
async def reject(item_id: str, reason: str = Form(default="")):
    queue.reject(item_id, reason=reason)
    return RedirectResponse("/", status_code=303)


@app.post("/auto-approve")
async def auto_approve(threshold: float = Form(default=0.9)):
    queue.auto_approve(threshold=threshold)
    return RedirectResponse("/", status_code=303)


def _serialize_graph():
    g = make_sample_graph()
    color_map = {
        NodeType.KPI.value: "#e8a838",
        NodeType.FEATURE.value: "#4a90d9",
        NodeType.PARAMETER.value: "#7bc67e",
    }
    nodes = []
    for node_id, data in g._g.nodes(data=True):
        ntype = data.get("node_type", "")
        nodes.append({
            "id": node_id,
            "label": data.get("name", node_id),
            "title": f"{ntype} | {data.get('unit', data.get('category', ''))}",
            "color": color_map.get(ntype, "#ccc"),
            "shape": "box" if ntype == NodeType.KPI.value else (
                "ellipse" if ntype == NodeType.FEATURE.value else "diamond"
            ),
        })
    edges = []
    for from_id, to_id, data in g._g.edges(data=True):
        direction = data.get("direction") or ""
        label = data.get("relation", "")
        if direction:
            label += f" {direction}"
        edges.append({
            "from": from_id,
            "to": to_id,
            "label": label,
            "color": "#dc3545" if direction == "-" else "#28a745" if direction == "+" else "#888",
        })
    return {"nodes": nodes, "edges": edges}


@app.get("/api/graph")
async def api_graph():
    return JSONResponse(_serialize_graph())


@app.get("/graph", response_class=HTMLResponse)
async def graph_view():
    data = _serialize_graph()
    template = jinja_env.get_template("graph.html")
    return HTMLResponse(template.render({
        "nodes_json": json.dumps(data["nodes"]),
        "edges_json": json.dumps(data["edges"]),
    }))
