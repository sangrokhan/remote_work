"""
AgenticRAGGraph — compiles and runs the LangGraph StateGraph.

Graph topology:
  var_constructor → planner → (var_binder | synthesizer)
  var_binder → executor → retriever → (refiner | synthesizer)
  refiner → (var_binder | synthesizer)

Methods:
  invoke()        async generator — streams per-node SSE events (delegates to stepby_invoke)
  ainvoke()       coroutine — returns final AgentState dict
  stepby_invoke() async generator — streams {event, node, stage, message, payload} dicts
                  via astream_events for WorkflowPanel node highlighting
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from langgraph_flow.agents.state import AgentState
from langgraph_flow.agents.nodes.planner_node import planner_node
from langgraph_flow.agents.nodes.executor_node import executor_node
from langgraph_flow.agents.nodes.refiner_node import refiner_node
from langgraph_flow.agents.nodes.synthesizer_node import synthesizer_node
from langgraph_flow.agents.nodes.retriever_node import retriever_node
from langgraph_flow.agents.nodes.var_constructor_node import var_constructor_node
from langgraph_flow.agents.nodes.var_binder_node import var_binder_node
from langgraph_flow.agents.edges.routing_logic import route_after_planner, route_after_executor, route_after_refiner
from tools.registry import ToolRegistry

_GRAPH_NODES = frozenset({
    "planner", "executor", "refiner", "synthesizer",
    "retriever", "var_constructor", "var_binder",
})


class AgenticRAGGraph:
    def __init__(self) -> None:
        self.tool_registry = ToolRegistry()
        self._graph = self._build()

    def _inject_registry(self, config: RunnableConfig) -> RunnableConfig:
        configurable = dict((config or {}).get("configurable", {}))
        configurable["tool_registry"] = self.tool_registry
        return {**(config or {}), "configurable": configurable}

    @property
    def graph(self):
        return self._graph

    def _build(self):
        builder = StateGraph(AgentState)

        builder.add_node("planner", planner_node.invoke)
        builder.add_node("executor", executor_node.invoke)
        builder.add_node("refiner", refiner_node.invoke)
        builder.add_node("synthesizer", synthesizer_node.invoke)
        builder.add_node("retriever", retriever_node.invoke)
        builder.add_node("var_constructor", var_constructor_node.invoke)
        builder.add_node("var_binder", var_binder_node.invoke)

        builder.add_edge(START, "var_constructor")
        builder.add_edge("var_constructor", "planner")
        builder.add_conditional_edges(
            "planner",
            route_after_planner,
            {"var_binder": "var_binder", "synthesizer": "synthesizer"},
        )
        builder.add_edge("var_binder", "executor")
        builder.add_edge("executor", "retriever")
        builder.add_conditional_edges(
            "retriever",
            route_after_executor,
            {"refiner": "refiner", "synthesizer": "synthesizer"},
        )
        builder.add_conditional_edges(
            "refiner",
            route_after_refiner,
            {"var_binder": "var_binder", "synthesizer": "synthesizer"},
        )
        builder.add_edge("synthesizer", END)

        return builder.compile()

    async def invoke(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> AsyncGenerator[dict, None]:
        async for event in self.stepby_invoke(state, config=self._inject_registry(config)):
            yield event
        # else: result = await self._graph.ainvoke(state, config=config); yield {"event": "workflow_complete", "payload": result}

    async def ainvoke(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        return await self._graph.ainvoke(state, config=self._inject_registry(config))

    async def stepby_invoke(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> AsyncGenerator[dict, None]:
        async for event in self._graph.astream_events(state, config=self._inject_registry(config), version="v2"):
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node not in _GRAPH_NODES:
                continue
            kind = event["event"]
            if kind == "on_chain_start":
                yield {
                    "event": "node_started",
                    "node": node,
                    "name": node,
                    "stage": "start",
                    "message": f"{node} 실행됨",
                }
            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if not isinstance(output, dict):
                    output = {}
                msg = str(next(iter(output.values()), f"{node} 완료"))
                yield {
                    "event": "node_finished",
                    "node": node,
                    "name": node,
                    "stage": "end",
                    "message": msg,
                    "payload": output,
                }


def create_agentic_rag_graph() -> AgenticRAGGraph:
    return AgenticRAGGraph()

########################################### remote code
# langgraph_agenticrag/src/agents/graph.py

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, InputState, create_initial_state
from langgraph_flow.agents.nodes.planner_node import PlannerNode
from langgraph_flow.agents.nodes.executor_node import ExecutorNode
from langgraph_flow.agents.nodes.refiner_node import RefinerNode
from langgraph_flow.agents.nodes.synthesizer_node import SynthesizerNode
from langgraph_flow.agents.nodes.var_constructor_node import VarConstructorNode
from langgraph_flow.agents.nodes.var_binder_node import VarBinderNode
from langgraph_flow.agents.nodes.retriever_node import RetrieverNode
from langgraph_flow.agents.edges.routing_logic import (
    route_after_planner,
    route_after_executor,
    route_after_refiner
)
from langgraph_flow.tools.registry import ToolRegistry


class AgenticRAGGraph:
    """AgenticRAG LangGraph 기반 그래프 클래스
    
    원본 app/services/agent_service.py 플로우:
    1. Planner.plan(state) → subtasks 생성 (한 번만 호출)
    2. Loop (의존관계 기반 subtask 실행):
       - _get_next_executable_subtask() → 실행 가능한 subtask 찾기
       - Executor.execute(state, current_subtask) → tool 결정
       - tool_registry.execute(tool_name, tool_args) → 실제 tool 실행
       - Refiner.refine(state, current_subtask) → 결과 검증
       - verdict가 True이면 subtask 완료, False이면 재시도
    3. Synthesizer.generate(state) → 최종 답변
    """

    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """
        그래프 초기화
        
        Args:
            llm: 언어 모델 (선택적)
        """
        self.llm = llm
        self.tool_registry = ToolRegistry()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        그래프 구성
        
        Returns:
            구성된 StateGraph
        """
        # StateGraph 생성
        workflow = StateGraph(AgentState)

        # 노드 추가
        workflow.add_node("var_constructor", VarConstructorNode().invoke)
        workflow.add_node("planner", PlannerNode().invoke)
        workflow.add_node("var_binder", VarBinderNode().invoke)
        workflow.add_node("retriever", RetrieverNode().invoke)
        workflow.add_node("executor", ExecutorNode().invoke)
        workflow.add_node("refiner", RefinerNode().invoke)
        workflow.add_node("synthesizer", SynthesizerNode().invoke)

        # 시작점에서 var_constructor로 연결
        workflow.add_edge("__start__", "var_constructor")

        # var_constructor → planner (필수 경로)
        workflow.add_edge("var_constructor", "planner")

        # planner 후 조건부 라우팅
        # - subtasks가 있으면 var_binder로
        # - subtasks가 없으면 synthesizer로
        workflow.add_conditional_edges(
            "planner",
            route_after_planner,
            {
                "var_binder": "var_binder",
                "synthesizer": "synthesizer"
            }
        )

        # var_binder → executor (바인딩 해결 후 실행)
        workflow.add_edge("var_binder", "executor")

        # executor → retriever (실행 후 retriever로)
        workflow.add_edge("executor", "retriever")

        # retriever 후 조건부 라우팅
        # - 검색 성공 시 refiner로
        # - 검색 실패 시 synthesizer로
        workflow.add_conditional_edges(
            "retriever",
            route_after_executor,
            {
                "refiner": "refiner",
                "synthesizer": "synthesizer"
            }
        )

        # refiner 후 조건부 라우팅
        # - 모든 subtask 완료 시 synthesizer로
        # - 더 실행할 subtask가 있으면 var_binder로 (루프)
        workflow.add_conditional_edges(
            "refiner",
            route_after_refiner,
            {
                "var_binder": "var_binder",
                "synthesizer": "synthesizer"
            }
        )

        # synthesizer에서 종료
        workflow.add_edge("synthesizer", END)

        return workflow

    def compile(self):
        """
        그래프 컴파일
        
        Returns:
            컴파일된 그래프
        """
        memory = MemorySaver()
        return self.graph.compile(checkpointer=memory)

    async def ainvoke(self, user_query: str, socket_id: str = "default") -> Dict[str, Any]:
        """
        그래프 실행
        
        Args:
            user_query: 사용자 질문
            socket_id: 소켓 ID (선택적)
            
        Returns:
            실행 결과
        """
        # 초기 상태 생성 - interaction_id를 미리 저장 
        initial_state = create_initial_state(user_query, socket_id)
        interaction_id = initial_state["interaction_id"]  # 초기 생성 시 저장

        print(f"=== DEBUG: Initial interaction_id: {interaction_id} ===")

        # 설정 구성
        config = {
            "configurable": {"thread_id": socket_id},
            "llm": self.llm,
            "tool_registry": self.tool_registry
        }

        # 그래프 컴파일 및 실행
        compiled_graph = self.compile()

        try:
            # 그래프 실행
            final_state = await compiled_graph.ainvoke(
                initial_state,
                config=config
            )

            print(f"=== DEBUG: Final state type: {type(final_state)} ===")
            print(
                f"=== DEBUG: Final state keys: {final_state.keys() if isinstance(final_state, dict) else 'N/A'} ===")

            # interaction_id 추출 - 초기 생성 값을 우선 사용 
            result_interaction_id = interaction_id
            if isinstance(final_state, dict):
                final_interaction_id = final_state.get("interaction_id", "")
                if final_interaction_id:  # 빈 문자열이 아닌 경우
                    result_interaction_id = final_interaction_id
                    print(
                        f"=== DEBUG: Using final_state interaction_id: {result_interaction_id} ===")
                else:
                    print(f"=== DEBUG: Using initial interaction_id: {result_interaction_id} ===")

            # steps 정보를 final_state에서 추출
            steps = []
            reference_features = []
            if isinstance(final_state, dict):
                steps = final_state.get("retriever_history", [])
                reference_features = final_state.get("reference_features", [])

            # final_response 추출
            final_response = ""
            if isinstance(final_state, dict):
                final_response = final_state.get("final_response", "")

            yield {
                "final_response": final_response,
                "interaction_id": result_interaction_id,
                "success": True,
                "steps": steps,
                "reference_features": reference_features
            }

        except Exception as e:
            print(f"Graph execution error: {e}")
            import traceback

            traceback.print_exc()
            yield {
                "final_response": "죄송합니다. 처리 중 오류가 발생했습니다.",
                "interaction_id": interaction_id,  # 초기 생성한 ID 사용
                "success": False,
                "error": str(e),
                "steps": []
            }


def create_agentic_rag_graph(llm: Optional[BaseLanguageModel] = None) -> AgenticRAGGraph:
    """
    AgenticRAG 그래프 팩토리 함수
    
    Args:
        llm: 언어 모델 (선택적)
        
    Returns:
        AgenticRAGGraph 인스턴스
    """
    return AgenticRAGGraph(llm)


def create_graph():
    """
    LangGraph Studio WebUI용 그래프 팩토리 함수
    
    InputState를 입력 스키마로 사용하여 WebUI에서 query만 입력받음
    내부적으로 AgentState로 변환되어 실행됨
    
    Returns:
        컴파일된 LangGraph 그래프
    """
    import os
    from langchain_openai import ChatOpenAI
    from agents.nodes.planner_node import PlannerNode
    from agents.nodes.executor_node import ExecutorNode
    from agents.nodes.refiner_node import RefinerNode
    from agents.nodes.synthesizer_node import SynthesizerNode
    from agents.nodes.var_constructor_node import VarConstructorNode
    from agents.nodes.var_binder_node import VarBinderNode
    from agents.nodes.retriever_node import RetrieverNode
    from agents.edges.routing_logic import (
        route_after_planner,
        route_after_executor,
        route_after_refiner
    )

    # LLM 인스턴스 생성 (LiteLLM 또는 환경 변수 설정 사용)
    llm = _create_llm_instance()

    # InputState를 입력 스키마로 사용하는 StateGraph 생성
    workflow = StateGraph(AgentState, input=InputState)

    # 노드 추가 - LLM 주입을 위한 래퍼 사용
    workflow.add_node("var_constructor", _wrap_node_with_llm(VarConstructorNode().invoke, llm))
    workflow.add_node("planner", _wrap_node_with_llm(PlannerNode().invoke, llm))
    workflow.add_node("var_binder", _wrap_node_with_llm(VarBinderNode().invoke, llm))
    workflow.add_node("retriever", _wrap_node_with_llm(RetrieverNode().invoke, llm))
    workflow.add_node("executor", _wrap_node_with_llm(ExecutorNode().invoke, llm))
    workflow.add_node("refiner", _wrap_node_with_llm(RefinerNode().invoke, llm))
    workflow.add_node("synthesizer", _wrap_node_with_llm(SynthesizerNode().invoke, llm))

    # 시작점에서 var_constructor로 연결
    workflow.add_edge("__start__", "var_constructor")

    # var_constructor → planner (필수 경로)
    workflow.add_edge("var_constructor", "planner")

    # planner 후 조건부 라우팅
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "var_binder": "var_binder",
            "synthesizer": "synthesizer"
        }
    )

    # var_binder → executor
    workflow.add_edge("var_binder", "executor")

    # executor → retriever
    workflow.add_edge("executor", "retriever")

    # retriever 후 조건부 라우팅
    workflow.add_conditional_edges(
        "retriever",
        route_after_executor,
        {
            "refiner": "refiner",
            "synthesizer": "synthesizer"
        }
    )

    # refiner 후 조건부 라우팅
    workflow.add_conditional_edges(
        "refiner",
        route_after_refiner,
        {
            "var_binder": "var_binder",
            "synthesizer": "synthesizer"
        }
    )

    # synthesizer에서 종료
    workflow.add_edge("synthesizer", END)

    # 컴파일 (checkpointer 없이)
    return workflow.compile()


def _create_llm_instance() -> BaseLanguageModel:
    """
    GaussO4 API를 사용하여 LLM 인스턴스 생성
    
    Returns:
        LLM 인스턴스
    """
    import os
    from langchain_openai import ChatOpenAI

    # GaussO4 API 설정
    gauss_api_key = os.getenv("GAUSS_O4_API_KEY")
    gauss_base_url = os.getenv("GAUSS_O4_BASE_URL",
                               "https://inference-webtrial-api.shuttle.sr-cloud.com/gauss-o4-instruct/v1/")
    gauss_model = os.getenv("GAUSS_O4_MODEL", "gauss-o4-instruct")

    if not gauss_api_key:
        raise ValueError("GAUSS_O4_API_KEY environment variable is required")

    print(f"=== DEBUG: Using GaussO4 API ===")
    print(f"=== DEBUG: Base URL: {gauss_base_url} ===")
    print(f"=== DEBUG: Model: {gauss_model} ===")

    return ChatOpenAI(
        model=gauss_model,
        base_url=gauss_base_url,
        api_key=gauss_api_key,
        temperature=0.7,
        default_headers={"Authorization": f"Basic {gauss_api_key}"}
    )


def _wrap_node_with_llm(node_func, llm: BaseLanguageModel):
    """
    노드 함수에 LLM을 주입하는 래퍼
    
    Args:
        node_func: 원본 노드 함수
        llm: 주입할 LLM 인스턴스
        
    Returns:
        LLM이 주입된 노드 함수
    """

    async def wrapped(state, config=None):
        # config가 없으면 생성
        if config is None:
            config = {}

        # config에 LLM 주입
        if isinstance(config, dict):
            if "configurable" not in config:
                config["configurable"] = {}
            config["configurable"]["llm"] = llm
            config["llm"] = llm
        elif hasattr(config, 'configurable'):
            config.configurable["llm"] = llm

        return await node_func(state, config)

    return wrapped


# 사용 예시
"""
# LLM 없이 테스트
graph = create_agentic_rag_graph()
result = await graph.invoke("삼성전자의 최신 스마트폰은 무엇인가요?")

# LLM과 함께 사용
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
graph = create_agentic_rag_graph(llm)
result = await graph.invoke("삼성전자의 최신 스마트폰은 무엇인가요?")
"""
