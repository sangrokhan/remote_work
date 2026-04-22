# langgraph_flow Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move workflow logic from `backend/stategraph_workflow.py` into `langgraph_flow/` using a proper per-file-per-node architecture and real LangGraph `astream_events` for SSE streaming.

**Architecture:** Each workflow node lives in its own file under `langgraph_flow/agents/nodes/`; prompts are strings in `langgraph_flow/prompts/`; `AgenticRAGGraph.invoke()` (already written in `graph.py`) streams `astream_events` from the compiled LangGraph graph. The FastAPI backend (`backend/app/main.py`) calls directly into `langgraph_flow` — no more manual thread+queue loop in `stategraph_workflow.py`. Nodes still sleep (simulated LLM latency) to keep the demo visual.

**Tech Stack:** Python, LangGraph `astream_events`, LangChain `RunnableConfig`, FastAPI SSE, pytest

**Test execution:** Always from repo root:
```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python3 -m pytest backend/tests/ -v
```
`conftest.py` already adds repo root to `sys.path` so `langgraph_flow` imports work.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `langgraph_flow/prompts/planner.py` | Planner prompt string constant |
| Modify | `langgraph_flow/prompts/executor.py` | Executor prompt string constant |
| Modify | `langgraph_flow/prompts/refiner.py` | Refiner prompt string constant |
| Modify | `langgraph_flow/prompts/synthesizer.py` | Synthesizer prompt string constant |
| Modify | `langgraph_flow/prompts/var_constructor.py` | Var constructor prompt constant |
| Modify | `langgraph_flow/prompts/var_binder.py` | Var binder prompt constant |
| Modify | `langgraph_flow/agents/nodes/executor_node.py` | Executor node (was empty stub) |
| Modify | `langgraph_flow/agents/nodes/refiner_node.py` | Refiner node (was empty stub) |
| Modify | `langgraph_flow/agents/nodes/synthesizer_node.py` | Synthesizer node (was empty stub) |
| Modify | `langgraph_flow/agents/nodes/retriever_node.py` | Dummy retriever — no real RAG |
| Modify | `langgraph_flow/agents/nodes/var_constructor_node.py` | Dummy var constructor |
| Modify | `langgraph_flow/agents/nodes/var_binder_node.py` | Dummy var binder |
| Modify | `backend/tests/test_workflow_integration.py` | Switch tests from stategraph_workflow to langgraph_flow |
| Modify | `backend/app/main.py` | Import from langgraph_flow; remove stategraph_workflow |
| Modify | `backend/graph_schema.py` | Use create_agentic_rag_graph for schema |
| Modify | `backend/Dockerfile` | Add COPY langgraph_flow /app/langgraph_flow |
| Delete | `backend/stategraph_workflow.py` | No longer needed |

---

## Task 1: Prompt strings

**Files:**
- Modify: `langgraph_flow/prompts/planner.py`
- Modify: `langgraph_flow/prompts/executor.py`
- Modify: `langgraph_flow/prompts/refiner.py`
- Modify: `langgraph_flow/prompts/synthesizer.py`
- Modify: `langgraph_flow/prompts/var_constructor.py`
- Modify: `langgraph_flow/prompts/var_binder.py`

No tests needed — plain string constants. Implement directly.

- [ ] **Step 1: Write `langgraph_flow/prompts/planner.py`**

```python
PLANNER_PROMPT = "입력을 분석하고 단계별 실행 계획을 수립하세요."
```

- [ ] **Step 2: Write `langgraph_flow/prompts/executor.py`**

```python
EXECUTOR_PROMPT = "계획을 바탕으로 각 단계를 실행하고 결과를 생성하세요."
```

- [ ] **Step 3: Write `langgraph_flow/prompts/refiner.py`**

```python
REFINER_PROMPT = "실행 결과를 검토하고 품질을 개선하세요."
```

- [ ] **Step 4: Write `langgraph_flow/prompts/synthesizer.py`**

```python
SYNTHESIZER_PROMPT = "모든 결과를 종합하여 최종 답변을 작성하세요."
```

- [ ] **Step 5: Write `langgraph_flow/prompts/var_constructor.py`**

```python
VAR_CONSTRUCTOR_PROMPT = "검색에 필요한 쿼리 변수를 구성하세요."
```

- [ ] **Step 6: Write `langgraph_flow/prompts/var_binder.py`**

```python
VAR_BINDER_PROMPT = "검색 결과를 상태 변수에 바인딩하세요."
```

- [ ] **Step 7: Commit**

```bash
git add langgraph_flow/prompts/
git commit -m "feat: add prompt strings for all langgraph_flow nodes"
```

---

## Task 2: Core node implementations (executor, refiner, synthesizer)

**Files:**
- Modify: `langgraph_flow/agents/nodes/executor_node.py`
- Modify: `langgraph_flow/agents/nodes/refiner_node.py`
- Modify: `langgraph_flow/agents/nodes/synthesizer_node.py`
- Modify: `backend/tests/test_workflow_integration.py`

- [ ] **Step 1: Write failing tests — replace `backend/tests/test_workflow_integration.py`**

```python
from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig

from llm.models.gauss_o4 import GaussO4
from langgraph_flow.agents.state import AgentState
from langgraph_flow.agents.nodes.planner_node import planner_node
from langgraph_flow.agents.nodes.executor_node import executor_node
from langgraph_flow.agents.nodes.refiner_node import refiner_node
from langgraph_flow.agents.nodes.synthesizer_node import synthesizer_node


@pytest.fixture
def gauss_config():
    llm = GaussO4(api_url="", api_key="")
    return RunnableConfig(configurable={"llm": llm})


@pytest.fixture
def base_state() -> AgentState:
    return AgentState(
        input="테스트 입력",
        agentic_rag=False,
        planner_output="",
        executor_output="",
        refiner_output="",
        retriever_output="",
        var_bindings="",
        final_output="",
        hop_count=0,
    )


def test_planner_uses_llm(gauss_config, base_state):
    result = planner_node(base_state, gauss_config)
    assert "[GaussO4]" in result["planner_output"]
    assert result["hop_count"] == 1


def test_executor_uses_llm(gauss_config, base_state):
    base_state["planner_output"] = "[GaussO4] plan"
    result = executor_node(base_state, gauss_config)
    assert "[GaussO4]" in result["executor_output"]
    assert result["hop_count"] == 1


def test_refiner_uses_llm(gauss_config, base_state):
    base_state["executor_output"] = "[GaussO4] exec"
    result = refiner_node(base_state, gauss_config)
    assert "[GaussO4]" in result["refiner_output"]
    assert result["hop_count"] == 1


def test_synthesizer_uses_llm(gauss_config, base_state):
    base_state["refiner_output"] = "[GaussO4] refined"
    result = synthesizer_node(base_state, gauss_config)
    assert "[GaussO4]" in result["final_output"]
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python3 -m pytest backend/tests/test_workflow_integration.py -v
```

Expected: `ImportError` — executor/refiner/synthesizer nodes are empty stubs

- [ ] **Step 3: Implement `langgraph_flow/agents/nodes/executor_node.py`**

```python
from __future__ import annotations

import time
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.executor import EXECUTOR_PROMPT


def executor_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    time.sleep(random.uniform(1.0, 5.0))
    result = llm.generate(prompt=EXECUTOR_PROMPT, context=state.get("planner_output", ""))
    return {"executor_output": result, "hop_count": state.get("hop_count", 0) + 1}
```

- [ ] **Step 4: Implement `langgraph_flow/agents/nodes/refiner_node.py`**

```python
from __future__ import annotations

import time
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.refiner import REFINER_PROMPT


def refiner_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    time.sleep(random.uniform(1.0, 5.0))
    result = llm.generate(prompt=REFINER_PROMPT, context=state.get("executor_output", ""))
    return {"refiner_output": result, "hop_count": state.get("hop_count", 0) + 1}
```

- [ ] **Step 5: Implement `langgraph_flow/agents/nodes/synthesizer_node.py`**

```python
from __future__ import annotations

import time
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.synthesizer import SYNTHESIZER_PROMPT


def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    time.sleep(random.uniform(1.0, 5.0))
    context = state.get("refiner_output") or state.get("executor_output", "")
    result = llm.generate(prompt=SYNTHESIZER_PROMPT, context=context)
    return {"final_output": result}
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python3 -m pytest backend/tests/test_workflow_integration.py -v
```

Expected: 4 passed

- [ ] **Step 7: Run all tests**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python3 -m pytest backend/tests/ -v --ignore=backend/tests/test_workflow_integration.py
```

Expected: 11 passed (llm tests unaffected)

- [ ] **Step 8: Commit**

```bash
git add langgraph_flow/agents/nodes/executor_node.py langgraph_flow/agents/nodes/refiner_node.py langgraph_flow/agents/nodes/synthesizer_node.py backend/tests/test_workflow_integration.py
git commit -m "feat: implement executor, refiner, synthesizer nodes in langgraph_flow"
```

---

## Task 3: Dummy RAG nodes

**Files:**
- Modify: `langgraph_flow/agents/nodes/retriever_node.py`
- Modify: `langgraph_flow/agents/nodes/var_constructor_node.py`
- Modify: `langgraph_flow/agents/nodes/var_binder_node.py`

These are dummy — no real retrieval. They echo input and pass through.

- [ ] **Step 1: Implement `langgraph_flow/agents/nodes/retriever_node.py`**

```python
from __future__ import annotations

import time
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


def retriever_node(state: AgentState, config: RunnableConfig) -> dict:
    time.sleep(random.uniform(0.5, 2.0))
    return {"retriever_output": f"[retriever-dummy] 검색 결과: {state.get('input', '')[:50]}"}
```

- [ ] **Step 2: Implement `langgraph_flow/agents/nodes/var_constructor_node.py`**

```python
from __future__ import annotations

import time
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.var_constructor import VAR_CONSTRUCTOR_PROMPT


def var_constructor_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    time.sleep(random.uniform(0.5, 2.0))
    result = llm.generate(prompt=VAR_CONSTRUCTOR_PROMPT, context=state.get("retriever_output", ""))
    return {"var_bindings": result}
```

- [ ] **Step 3: Implement `langgraph_flow/agents/nodes/var_binder_node.py`**

```python
from __future__ import annotations

import time
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.var_binder import VAR_BINDER_PROMPT


def var_binder_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    time.sleep(random.uniform(0.5, 2.0))
    result = llm.generate(prompt=VAR_BINDER_PROMPT, context=state.get("var_bindings", ""))
    return {"retriever_output": result}
```

- [ ] **Step 4: Verify import chain works**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python3 -c "
from langgraph_flow.agents.nodes.retriever_node import retriever_node
from langgraph_flow.agents.nodes.var_constructor_node import var_constructor_node
from langgraph_flow.agents.nodes.var_binder_node import var_binder_node
print('OK')
"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add langgraph_flow/agents/nodes/retriever_node.py langgraph_flow/agents/nodes/var_constructor_node.py langgraph_flow/agents/nodes/var_binder_node.py
git commit -m "feat: add dummy RAG nodes (retriever, var_constructor, var_binder)"
```

---

## Task 4: Backend integration — main.py and graph_schema.py

**Files:**
- Modify: `backend/app/main.py`
- Modify: `backend/graph_schema.py`

Read both files before editing.

- [ ] **Step 1: Read `backend/graph_schema.py` and note how `serialize_stategraph_to_json` is called**

It currently receives `build_workflow_graph()` output (a compiled `StateGraph`). With new code it will receive `create_agentic_rag_graph(False)._graph` which is also a compiled `StateGraph`. No change needed to `graph_schema.py` logic — only its import.

- [ ] **Step 2: Update `backend/graph_schema.py`**

Find the line that imports from `stategraph_workflow`:
```python
# If there is any reference to stategraph_workflow in graph_schema.py, remove it.
# graph_schema.py likely only has serialize_stategraph_to_json — no changes needed.
```

Check: `grep -n stategraph_workflow backend/graph_schema.py` — if empty, no change needed.

- [ ] **Step 3: Update `backend/app/main.py`**

Replace the entire file with:

```python
from __future__ import annotations

import json
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig

from app.models import RunWorkflowRequest
from graph_schema import serialize_stategraph_to_json
from langgraph_flow.agents.graph import create_agentic_rag_graph
from llm.factory import get_llm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("workflow_api")

app = FastAPI(title="LangGraph Vis")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_workflow_graph = create_agentic_rag_graph(False)._graph


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/graph")
def serve_graph_schema() -> dict:
    return serialize_stategraph_to_json(_workflow_graph)


@app.post("/api/run")
async def run_workflow_sse(req: RunWorkflowRequest) -> StreamingResponse:
    logger.debug(
        "POST /api/run: model=%s agentic_rag=%s response_mode=%s max_tokens=%s input_len=%d",
        req.model, req.agentic_rag, req.response_mode, req.max_tokens, len(req.input),
    )

    async def event_gen():
        init = {
            "event": "run_started",
            "run_id": req.run_id,
            "model": req.model,
            "agentic_rag": req.agentic_rag,
            "response_mode": req.response_mode,
            "max_tokens": req.max_tokens,
        }
        yield f"event: run_started\ndata: {json.dumps(init, ensure_ascii=False)}\n\n"

        try:
            llm = get_llm(req.model, req.api_url, req.api_key)
            config = RunnableConfig(configurable={"llm": llm})
            state = {
                "input": req.input,
                "agentic_rag": req.agentic_rag,
                "planner_output": "",
                "executor_output": "",
                "refiner_output": "",
                "retriever_output": "",
                "var_bindings": "",
                "final_output": "",
                "hop_count": 0,
            }
            graph = create_agentic_rag_graph(req.agentic_rag)
            async for event in graph.invoke(state, config):
                event_type = event.get("event", "workflow_event")
                yield f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            err = {"event": "workflow_error", "message": str(exc)}
            yield f"event: workflow_error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: replace stategraph_workflow with langgraph_flow in FastAPI backend"
```

---

## Task 5: Dockerfile and delete stategraph_workflow

**Files:**
- Modify: `backend/Dockerfile`
- Delete: `backend/stategraph_workflow.py`

- [ ] **Step 1: Update `backend/Dockerfile`**

Current content:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Replace with:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app
COPY langgraph_flow /app/langgraph_flow

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Delete `backend/stategraph_workflow.py`**

```bash
rm backend/stategraph_workflow.py
```

- [ ] **Step 3: Verify no remaining imports of stategraph_workflow**

```bash
grep -rn stategraph_workflow backend/
```

Expected: no output

- [ ] **Step 4: Commit**

```bash
git add backend/Dockerfile
git rm backend/stategraph_workflow.py
git commit -m "chore: remove stategraph_workflow, copy langgraph_flow into Docker image"
```

---

## Task 6: Deploy and verify

- [ ] **Step 1: Build and start**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
docker compose up --build -d
```

- [ ] **Step 2: Check services up**

```bash
docker compose ps
```

Expected: both `chat-front` and `workflow-api` show `Up`

- [ ] **Step 3: Health check**

```bash
curl http://localhost:10001/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 4: Graph schema — verify new nodes visible**

```bash
curl -s http://localhost:10001/graph | python3 -m json.tool | grep '"id"'
```

Expected: `planner`, `executor`, `refiner`, `synthesizer` (and `retriever`, `var_constructor`, `var_binder` for agentic_rag=true graph)

- [ ] **Step 5: Test run — GaussO4 model, no RAG**

```bash
curl -sN -X POST http://localhost:10001/api/run \
  -H "Content-Type: application/json" \
  -d '{"run_id":"t1","input":"테스트","model":"GaussO4","api_url":"","api_key":"","agentic_rag":false}' \
  --max-time 30 2>&1 | grep -E "node_started|node_finished|workflow_complete"
```

Expected: sequence of `node_started`/`node_finished` for planner→executor→(refiner|synthesizer), ending with `workflow_complete`

- [ ] **Step 6: Test unknown model**

```bash
curl -sN -X POST http://localhost:10001/api/run \
  -H "Content-Type: application/json" \
  -d '{"run_id":"t2","input":"test","model":"BadModel","api_url":"","api_key":""}' \
  --max-time 5 2>&1
```

Expected: `workflow_error` with `Unknown model: BadModel`

- [ ] **Step 7: Commit deploy verification**

```bash
git commit --allow-empty -m "chore: deploy verified - langgraph_flow migration complete"
```
