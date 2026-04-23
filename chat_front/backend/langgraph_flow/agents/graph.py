"""
AgenticRAGGraph — compiles and runs the LangGraph StateGraph.

create_agentic_rag_graph(agentic_rag)  builds the graph (with or without RAG path)
AgenticRAGGraph.invoke()               streams node events as an async generator
                                       yielding {event, node, stage, message, payload} dicts
"""
from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator

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
from langgraph_flow.agents.edges.routing_logic import executor_route, refiner_route

_GRAPH_NODES = frozenset({
    "planner", "executor", "refiner", "synthesizer",
    "retriever", "var_constructor", "var_binder",
})


class AgenticRAGGraph:
    def __init__(self) -> None:
        self._graph = self._build()

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

        builder.add_edge(START, "retriever")
        builder.add_edge("retriever", "var_constructor")
        builder.add_edge("var_constructor", "var_binder")
        builder.add_edge("var_binder", "planner")
        builder.add_edge("planner", "executor")
        builder.add_conditional_edges(
            "executor",
            executor_route,
            {"to_refiner": "refiner", "to_synthesizer": "synthesizer"},
        )
        builder.add_conditional_edges(
            "refiner",
            refiner_route,
            {"to_synthesizer": "synthesizer", "to_planner": "planner"},
        )
        builder.add_edge("synthesizer", END)

        return builder.compile()

    async def invoke(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> AsyncGenerator[dict, None]:
        async for event in self.stepby_invoke(state, config=config):
            yield event

    async def stepby_invoke(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> AsyncGenerator[dict, None]:
        async for event in self._graph.astream_events(state, config=config, version="v2"):
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
