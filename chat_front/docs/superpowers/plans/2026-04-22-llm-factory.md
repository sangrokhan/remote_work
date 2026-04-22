# LLM Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM factory (`langgraph/core/`) that creates dummy model objects by name and passes them through LangGraph `RunnableConfig` so each workflow node can call `llm.generate(prompt, context)`.

**Architecture:** A `BaseLLM` abstract class defines the interface. `get_llm(model_name, api_url, api_key)` returns the correct dummy implementation. At workflow start, the factory instantiates the model and injects it into `RunnableConfig(configurable={"llm": llm})`, which each node receives as its second argument. The demo backend (`backend/stategraph_workflow.py`) imports from `langgraph.core.factory`.

**Tech Stack:** Python, LangChain `RunnableConfig`, pytest, FastAPI/Pydantic

**Test execution:** Always run from repo root with `PYTHONPATH=backend`:
```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/ -v
```

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `langgraph/__init__.py` | Package marker |
| Create | `langgraph/core/__init__.py` | Package marker |
| Modify | `langgraph/core/base.py` | `BaseLLM` abstract class (was empty stub) |
| Create | `langgraph/core/models/__init__.py` | Package marker |
| Create | `langgraph/core/models/gauss_o4.py` | `GaussO4` dummy |
| Create | `langgraph/core/models/gauss_o4_think.py` | `GaussO4Think` dummy |
| Create | `langgraph/core/models/gemma4_e4b_it.py` | `Gemma4E4BIt` dummy |
| Modify | `langgraph/core/factory.py` | `get_llm()` + `MODEL_REGISTRY` (was empty stub) |
| Create | `backend/tests/__init__.py` | Package marker |
| Create | `backend/tests/conftest.py` | Add repo root to sys.path for imports |
| Create | `backend/tests/test_llm.py` | Tests for base, models, factory |
| Create | `backend/tests/test_workflow_integration.py` | Tests nodes use RunnableConfig LLM |
| Modify | `backend/app/models.py` | Add `api_url`, `api_key` fields |
| Modify | `backend/stategraph_workflow.py` | Add `RunnableConfig` to 4 nodes + factory call |
| Modify | `backend/Dockerfile` | Add `COPY langgraph /app/langgraph` |

---

## Task 1: BaseLLM abstract class

**Files:**
- Create: `langgraph/__init__.py`
- Create: `langgraph/core/__init__.py`
- Modify: `langgraph/core/base.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_llm.py`

- [ ] **Step 1: Create package markers**

```bash
touch langgraph/__init__.py langgraph/core/__init__.py backend/tests/__init__.py
```

- [ ] **Step 2: Create `backend/tests/conftest.py`**

This makes `langgraph` importable when running tests with `PYTHONPATH=backend` from repo root.

```python
import sys
import os

# Add repo root so `langgraph` package is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
```

- [ ] **Step 3: Write failing test**

Create `backend/tests/test_llm.py`:

```python
import pytest
from langgraph.core.base import BaseLLM


def test_base_llm_is_abstract():
    with pytest.raises(TypeError):
        BaseLLM(api_url="http://example.com", api_key="key")


def test_concrete_subclass_requires_generate():
    class Incomplete(BaseLLM):
        pass

    with pytest.raises(TypeError):
        Incomplete(api_url="", api_key="")


def test_concrete_subclass_works():
    class Concrete(BaseLLM):
        def generate(self, prompt: str, context: str) -> str:
            return f"ok: {context}"

    llm = Concrete(api_url="http://x", api_key="k")
    assert llm.generate("p", "ctx") == "ok: ctx"
    assert llm.api_url == "http://x"
    assert llm.api_key == "k"
```

- [ ] **Step 4: Run test — expect FAIL**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/test_llm.py::test_base_llm_is_abstract -v
```

Expected: `ImportError` or `TypeError` (BaseLLM is empty stub, not abstract yet)

- [ ] **Step 5: Implement `langgraph/core/base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(self, api_url: str, api_key: str) -> None:
        self.api_url = api_url
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str: ...
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/test_llm.py -v
```

Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add langgraph/__init__.py langgraph/core/__init__.py langgraph/core/base.py backend/tests/__init__.py backend/tests/conftest.py backend/tests/test_llm.py
git commit -m "feat: add BaseLLM abstract interface in langgraph/core"
```

---

## Task 2: Dummy model implementations

**Files:**
- Create: `langgraph/core/models/__init__.py`
- Create: `langgraph/core/models/gauss_o4.py`
- Create: `langgraph/core/models/gauss_o4_think.py`
- Create: `langgraph/core/models/gemma4_e4b_it.py`
- Modify: `backend/tests/test_llm.py`

- [ ] **Step 1: Write failing tests — append to `backend/tests/test_llm.py`**

```python
from langgraph.core.models.gauss_o4 import GaussO4
from langgraph.core.models.gauss_o4_think import GaussO4Think
from langgraph.core.models.gemma4_e4b_it import Gemma4E4BIt


def test_gauss_o4_generate():
    llm = GaussO4(api_url="http://example.com", api_key="key")
    result = llm.generate(prompt="분석하세요", context="사용자 질문")
    assert "[GaussO4]" in result
    assert "사용자 질문" in result


def test_gauss_o4_think_generate():
    llm = GaussO4Think(api_url="http://example.com", api_key="key")
    result = llm.generate(prompt="추론하세요", context="복잡한 문제")
    assert "[GaussO4-think]" in result
    assert "복잡한 문제" in result


def test_gemma4_generate():
    llm = Gemma4E4BIt(api_url="http://example.com", api_key="key")
    result = llm.generate(prompt="요약하세요", context="긴 텍스트")
    assert "[Gemma4-E4B-it]" in result
    assert "긴 텍스트" in result


def test_all_models_are_base_llm():
    for cls in [GaussO4, GaussO4Think, Gemma4E4BIt]:
        llm = cls(api_url="", api_key="")
        assert isinstance(llm, BaseLLM)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/test_llm.py -k "gauss or gemma" -v
```

Expected: `ModuleNotFoundError: No module named 'langgraph.core.models'`

- [ ] **Step 3: Create package marker**

```bash
mkdir -p langgraph/core/models && touch langgraph/core/models/__init__.py
```

- [ ] **Step 4: Implement `langgraph/core/models/gauss_o4.py`**

```python
from __future__ import annotations

from langgraph.core.base import BaseLLM


class GaussO4(BaseLLM):
    def generate(self, prompt: str, context: str) -> str:
        return f"[GaussO4] {prompt[:30]} → {context[:80]}"
```

- [ ] **Step 5: Implement `langgraph/core/models/gauss_o4_think.py`**

```python
from __future__ import annotations

from langgraph.core.base import BaseLLM


class GaussO4Think(BaseLLM):
    def generate(self, prompt: str, context: str) -> str:
        return f"[GaussO4-think] <thinking>{prompt[:30]}</thinking> → {context[:80]}"
```

- [ ] **Step 6: Implement `langgraph/core/models/gemma4_e4b_it.py`**

```python
from __future__ import annotations

from langgraph.core.base import BaseLLM


class Gemma4E4BIt(BaseLLM):
    def generate(self, prompt: str, context: str) -> str:
        return f"[Gemma4-E4B-it] {context[:60]}"
```

- [ ] **Step 7: Run tests — expect PASS**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/test_llm.py -v
```

Expected: 7 passed

- [ ] **Step 8: Commit**

```bash
git add langgraph/core/models/ backend/tests/test_llm.py
git commit -m "feat: add GaussO4, GaussO4Think, Gemma4E4BIt dummy models"
```

---

## Task 3: Factory function

**Files:**
- Modify: `langgraph/core/factory.py`
- Modify: `backend/tests/test_llm.py`

- [ ] **Step 1: Write failing tests — append to `backend/tests/test_llm.py`**

```python
from langgraph.core.factory import get_llm


def test_factory_returns_gauss_o4():
    llm = get_llm("GaussO4", "http://example.com", "key")
    assert isinstance(llm, GaussO4)
    assert llm.api_url == "http://example.com"
    assert llm.api_key == "key"


def test_factory_returns_gauss_o4_think():
    llm = get_llm("GaussO4-think", "http://example.com", "key")
    assert isinstance(llm, GaussO4Think)


def test_factory_returns_gemma4():
    llm = get_llm("Gemma4-E4B-it", "http://example.com", "key")
    assert isinstance(llm, Gemma4E4BIt)


def test_factory_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model: bogus"):
        get_llm("bogus", "", "")
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/test_llm.py -k "factory" -v
```

Expected: `ImportError` (factory.py is empty stub)

- [ ] **Step 3: Implement `langgraph/core/factory.py`**

```python
from __future__ import annotations

from langgraph.core.base import BaseLLM
from langgraph.core.models.gauss_o4 import GaussO4
from langgraph.core.models.gauss_o4_think import GaussO4Think
from langgraph.core.models.gemma4_e4b_it import Gemma4E4BIt

MODEL_REGISTRY: dict[str, type[BaseLLM]] = {
    "GaussO4": GaussO4,
    "GaussO4-think": GaussO4Think,
    "Gemma4-E4B-it": Gemma4E4BIt,
}


def get_llm(model_name: str, api_url: str, api_key: str) -> BaseLLM:
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name}")
    return cls(api_url=api_url, api_key=api_key)
```

- [ ] **Step 4: Run all tests — expect PASS**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/test_llm.py -v
```

Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add langgraph/core/factory.py backend/tests/test_llm.py
git commit -m "feat: add LLM factory with MODEL_REGISTRY"
```

---

## Task 4: Update RunWorkflowRequest

**Files:**
- Modify: `backend/app/models.py`

- [ ] **Step 1: Edit `backend/app/models.py`**

Replace the entire file with:

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class RunWorkflowRequest(BaseModel):
    run_id: str
    input: str
    model: str = Field(default="GaussO4")
    response_mode: str = Field(default="normal")
    max_tokens: int = Field(default=1024)
    agentic_rag: bool = Field(default=False)
    api_url: str = Field(default="")
    api_key: str = Field(default="")
```

- [ ] **Step 2: Verify import still works**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -c "from app.models import RunWorkflowRequest; r = RunWorkflowRequest(run_id='x', input='y'); print(r.model, r.api_url)"
```

Expected: `GaussO4 `

- [ ] **Step 3: Commit**

```bash
git add backend/app/models.py
git commit -m "feat: add api_url and api_key to RunWorkflowRequest"
```

---

## Task 5: Workflow integration

**Files:**
- Modify: `backend/stategraph_workflow.py`
- Create: `backend/tests/test_workflow_integration.py`

- [ ] **Step 1: Write failing integration tests**

Create `backend/tests/test_workflow_integration.py`:

```python
from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.core.models.gauss_o4 import GaussO4
from stategraph_workflow import DemoState, _planner, _executor, _refiner, _synthesizer


@pytest.fixture
def gauss_config():
    llm = GaussO4(api_url="", api_key="")
    return RunnableConfig(configurable={"llm": llm})


@pytest.fixture
def base_state() -> DemoState:
    return DemoState(
        llm_input="테스트 입력",
        planner_output="",
        executor_output="",
        refiner_output="",
        final_output="",
        hop_count=0,
        planner_delay=0.0,
        executor_delay=0.0,
        refiner_delay=0.0,
        synthesizer_delay=0.0,
    )


def test_planner_uses_llm(gauss_config, base_state):
    result = _planner(base_state, gauss_config)
    assert "[GaussO4]" in result["planner_output"]
    assert result["hop_count"] == 1


def test_executor_uses_llm(gauss_config, base_state):
    base_state["planner_output"] = "[GaussO4] plan"
    result = _executor(base_state, gauss_config)
    assert "[GaussO4]" in result["executor_output"]


def test_refiner_uses_llm(gauss_config, base_state):
    base_state["executor_output"] = "[GaussO4] exec"
    result = _refiner(base_state, gauss_config)
    assert "[GaussO4]" in result["refiner_output"]


def test_synthesizer_uses_llm(gauss_config, base_state):
    base_state["refiner_output"] = "[GaussO4] refined"
    result = _synthesizer(base_state, gauss_config)
    assert "[GaussO4]" in result["final_output"]
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/test_workflow_integration.py -v
```

Expected: `TypeError: _planner() takes 1 positional argument but 2 were given`

- [ ] **Step 3: Update `backend/stategraph_workflow.py` — add imports**

At the top of the file, after existing imports, add:

```python
from langchain_core.runnables import RunnableConfig
from langgraph.core.factory import get_llm
```

- [ ] **Step 4: Update `_planner`**

Replace:
```python
def _planner(state: DemoState) -> dict:
    delay = _sleep_node()
    return {
        "planner_output": f"[planner] 계획 생성 완료: {state['llm_input']} ({delay:.1f}s)",
        "hop_count": state.get("hop_count", 0) + 1,
        "planner_delay": delay,
    }
```

With:
```python
def _planner(state: DemoState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    delay = _sleep_node()
    result = llm.generate(
        prompt="입력을 분석하고 실행 계획을 수립하세요.",
        context=state["llm_input"],
    )
    return {
        "planner_output": result,
        "hop_count": state.get("hop_count", 0) + 1,
        "planner_delay": delay,
    }
```

- [ ] **Step 5: Update `_executor`**

Replace:
```python
def _executor(state: DemoState) -> dict:
    delay = _sleep_node()
    return {
        "executor_output": f"[executor] 실행 결과: {state.get('planner_output', '')} ({delay:.1f}s)",
        "hop_count": state.get("hop_count", 0) + 1,
        "executor_delay": delay,
    }
```

With:
```python
def _executor(state: DemoState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    delay = _sleep_node()
    result = llm.generate(
        prompt="계획을 실행하고 결과를 생성하세요.",
        context=state.get("planner_output", ""),
    )
    return {
        "executor_output": result,
        "hop_count": state.get("hop_count", 0) + 1,
        "executor_delay": delay,
    }
```

- [ ] **Step 6: Update `_refiner`**

Replace existing `_refiner` with:
```python
def _refiner(state: DemoState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    delay = _sleep_node()
    result = llm.generate(
        prompt="실행 결과를 검토하고 개선점을 제안하세요.",
        context=state.get("executor_output", ""),
    )
    return {
        "refiner_output": result,
        "hop_count": state.get("hop_count", 0) + 1,
        "refiner_delay": delay,
    }
```

- [ ] **Step 7: Update `_synthesizer`**

Replace existing `_synthesizer` with:
```python
def _synthesizer(state: DemoState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    delay = _sleep_node()
    result = llm.generate(
        prompt="모든 결과를 종합하여 최종 답변을 작성하세요.",
        context=state.get("refiner_output", state.get("executor_output", "")),
    )
    return {
        "final_output": result,
        "synthesizer_delay": delay,
    }
```

- [ ] **Step 8: Update `run_demo_workflow_events` — create LLM and config**

Find `run_demo_workflow_events`. After `logger.debug(...)` and state initialization, add:

```python
    llm = get_llm(req.model, req.api_url, req.api_key)
    config = RunnableConfig(configurable={"llm": llm})
```

Then update every direct node call: `_planner(state)` → `_planner(state, config)`, same for `_executor`, `_refiner`, `_synthesizer`.

- [ ] **Step 9: Update `run_demo_workflow` to avoid breaking LangGraph invoke**

Replace:
```python
def run_demo_workflow(llm_input: str) -> dict:
    graph = build_workflow_graph()
    initial_state: DemoState = {
        "llm_input": llm_input,
        "planner_output": "",
        "executor_output": "",
        "refiner_output": "",
        "final_output": "",
        "hop_count": 0,
        "planner_delay": 0.0,
        "executor_delay": 0.0,
        "refiner_delay": 0.0,
        "synthesizer_delay": 0.0,
    }
    return graph.invoke(initial_state)
```

With:
```python
def run_demo_workflow(llm_input: str) -> dict:
    from langgraph.core.models.gauss_o4 import GaussO4
    graph = build_workflow_graph()
    llm = GaussO4(api_url="", api_key="")
    config = RunnableConfig(configurable={"llm": llm})
    initial_state: DemoState = {
        "llm_input": llm_input,
        "planner_output": "",
        "executor_output": "",
        "refiner_output": "",
        "final_output": "",
        "hop_count": 0,
        "planner_delay": 0.0,
        "executor_delay": 0.0,
        "refiner_delay": 0.0,
        "synthesizer_delay": 0.0,
    }
    return graph.invoke(initial_state, config=config)
```

- [ ] **Step 10: Run integration tests — expect PASS**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/test_workflow_integration.py -v
```

Expected: 4 passed

- [ ] **Step 11: Run all tests — expect full PASS**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
PYTHONPATH=backend python -m pytest backend/tests/ -v
```

Expected: 15 passed

- [ ] **Step 12: Commit**

```bash
git add backend/stategraph_workflow.py backend/tests/test_workflow_integration.py
git commit -m "feat: integrate LLM factory into workflow nodes via RunnableConfig"
```

---

## Task 6: Update Dockerfile and deploy

**Files:**
- Modify: `backend/Dockerfile`

- [ ] **Step 1: Edit `backend/Dockerfile`**

Replace:
```dockerfile
COPY backend /app
```

With:
```dockerfile
COPY backend /app
COPY langgraph /app/langgraph
```

- [ ] **Step 2: Build and start**

```bash
cd /home/han/.openclaw/workspace/remote_work/chat_front
docker compose up --build -d
```

- [ ] **Step 3: Check services up**

```bash
docker compose ps
```

Expected: both `chat-front` and `workflow-api` show `Up`

- [ ] **Step 4: Health check**

```bash
curl http://localhost:10001/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 5: Test run with model selection**

```bash
curl -N -X POST http://localhost:10001/api/run \
  -H "Content-Type: application/json" \
  -d '{"run_id":"test-1","input":"hello","model":"GaussO4","api_url":"","api_key":""}'
```

Expected: SSE stream with `workflow_event` messages containing `[GaussO4]` in node outputs

- [ ] **Step 6: Test unknown model error**

```bash
curl -N -X POST http://localhost:10001/api/run \
  -H "Content-Type: application/json" \
  -d '{"run_id":"test-2","input":"hello","model":"UnknownModel","api_url":"","api_key":""}'
```

Expected: SSE `workflow_error` event with `Unknown model: UnknownModel`

- [ ] **Step 7: Commit**

```bash
git add backend/Dockerfile
git commit -m "chore: copy langgraph package into backend Docker image"
```
