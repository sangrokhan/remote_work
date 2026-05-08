from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader

from causality_graph.extraction.review_queue import ReviewQueue, ReviewStatus

QUEUE_PATH = Path("data/review_queue.jsonl")
QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Causality Graph Review UI")

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
