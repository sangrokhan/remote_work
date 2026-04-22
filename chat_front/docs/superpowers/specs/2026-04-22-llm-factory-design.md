---
title: LLM Factory Design
date: 2026-04-22
status: approved
---

# LLM Factory — Design Spec

## 목적

`req.model`로 선택된 LLM을 factory에서 생성하고 LangGraph `RunnableConfig`를 통해 각 노드에 전달한다. 현재는 dummy 구현. 실제 인하우스/오픈소스 모델 HTTP 연결 시 구현체만 교체.

## 지원 모델

| model_name | class | 특성 |
|---|---|---|
| `GaussO4` | `GaussO4` | 표준 응답 |
| `GaussO4-think` | `GaussO4Think` | 추론 강조 응답 |
| `Gemma4-E4B-it` | `Gemma4E4BIt` | 경량 응답 |

## 파일 구조

```
backend/
  llm/
    __init__.py
    base.py          — BaseLLM abstract class
    factory.py       — get_llm(model_name, api_url, api_key) -> BaseLLM
    models/
      __init__.py
      gauss_o4.py
      gauss_o4_think.py
      gemma4_e4b_it.py
  app/
    models.py        — RunWorkflowRequest에 api_url, api_key 필드 추가
  stategraph_workflow.py  — 노드에 RunnableConfig 인자, factory 호출 추가
```

## 인터페이스

### `BaseLLM` (`llm/base.py`)

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    def __init__(self, api_url: str, api_key: str) -> None:
        self.api_url = api_url
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str: ...
```

### `factory.py`

```python
MODEL_REGISTRY = {
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

### Dummy 구현체 패턴

```python
class GaussO4(BaseLLM):
    def generate(self, prompt: str, context: str) -> str:
        return f"[GaussO4] {prompt[:20]}… → {context[:50]}"
```

각 모델별로 응답 prefix만 다르게. 실제 연결 시 `generate` 내부에 `POST self.api_url` 추가.

## RunWorkflowRequest 변경

```python
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

## 워크플로우 통합

### 노드 시그니처 변경

```python
from langchain_core.runnables import RunnableConfig

def _planner(state: DemoState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    delay = _sleep_node()
    result = llm.generate(
        prompt="입력을 분석하고 실행 계획을 수립하세요.",
        context=state["llm_input"],
    )
    return {
        "planner_output": result,
        "hop_count": state["hop_count"] + 1,
        "planner_delay": delay,
    }
```

4개 노드(`planner`, `executor`, `refiner`, `synthesizer`) 동일 패턴. 노드별 `prompt` 문자열만 다름. delay(sleep 시뮬레이션)는 유지.

### `run_demo_workflow_events` 진입점

```python
def run_demo_workflow_events(req: RunWorkflowRequest) -> Generator[dict, None, None]:
    llm = get_llm(req.model, req.api_url, req.api_key)
    config = RunnableConfig(configurable={"llm": llm})
    # 이후 노드 호출 시 config 전달
```

## 에러 처리

- 알 수 없는 `model_name` → `ValueError` → `workflow_error` 이벤트로 프론트 전달 (기존 핸들러 재사용)
- `api_url`/`api_key` 미설정 시 dummy는 무시, 실제 구현체에서 검증

## 확장 가이드

새 모델 추가:
1. `llm/models/new_model.py` 작성 (`BaseLLM` 상속)
2. `factory.py` `MODEL_REGISTRY`에 한 줄 추가

실제 HTTP 연결:
- `generate` 내부만 수정 — `POST self.api_url`, `Authorization: Bearer self.api_key`
- 인터페이스 변경 없음
