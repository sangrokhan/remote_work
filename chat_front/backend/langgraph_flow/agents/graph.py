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

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)
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
        logger.debug("AgenticRAGGraph.invoke: start | query=%s", str(state.get("user_query", ""))[:80])
        async for event in self.stepby_invoke(state, config=self._inject_registry(config)):
            yield event
        logger.debug("AgenticRAGGraph.invoke: done")
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
                logger.debug("[FLOW] → %s start", node)
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
                next_node = output.get("next", "?")
                logger.debug("[FLOW] ← %s end | next=%s | keys=%s", node, next_node, list(output.keys()))
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
