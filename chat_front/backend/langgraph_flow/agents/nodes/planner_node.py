"""
Planner node — generates subtasks from user query via LLM.
On first call: produces subtasks list, routes to executor.
On subsequent calls (subtasks exist): routes directly to executor.
Terminates to synthesizer when is_finished or max_steps reached.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.planner import PLANNER_SYSTEM_PROMPT
from langgraph_flow.prompts.var_constructor import CONSTRUCTOR_SYSTEM_PROMPT


class PlannerNode:
    def __init__(self):
        self.name = "planner"
        self.system_prompt = PLANNER_SYSTEM_PROMPT

    async def invoke(self, state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        llm = None
        if config is not None:
            if isinstance(config, dict):
                llm = config.get("llm")
                if llm is None and "configurable" in config:
                    llm = config["configurable"].get("llm")
            elif hasattr(config, "configurable") and config.configurable:
                llm = config.configurable.get("llm")
            elif hasattr(config, "get"):
                llm = config.get("llm")

        if "next" not in state or state["next"] is None:
            state["next"] = "executor"

        return await self.plan_next_step(state, llm)

    async def _extract_binding_context_with_llm(self, state: AgentState, llm: BaseLanguageModel) -> dict:
        messages = [
            SystemMessage(content=CONSTRUCTOR_SYSTEM_PROMPT),
            HumanMessage(content=f"User Query: {state.get('user_query', '')}"),
        ]
        try:
            response = await llm.ainvoke(messages)
            binding_context = json.loads(response.content or "{}")
            binding_context.setdefault("query_entities", {"features": [], "keywords": []})
            binding_context.setdefault("previous_features", [])
            binding_context.setdefault("explicit_dependencies", [])
            return binding_context
        except Exception:
            return {
                "query_entities": {"features": [], "keywords": []},
                "previous_features": [],
                "explicit_dependencies": [],
            }

    async def plan_next_step(self, state: AgentState, llm: Optional[BaseLanguageModel] = None) -> AgentState:
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", 10)
        is_finished = state.get("is_finished", False)

        if is_finished or current_step >= max_steps:
            return {**state, "is_finished": True, "next": "synthesizer"}

        if state.get("subtasks"):
            return {**state, "next": "executor"}

        if llm is None:
            default_subtasks = [{
                "id": 0,
                "goal": f"Query processing: {state.get('user_query', 'No query')}",
                "task_type": "THINK",
                "verdict": False,
                "dependencies": [],
                "bindings": {},
            }]
            return {**state, "subtasks": default_subtasks, "current_step": current_step + 1, "next": "executor"}

        try:
            binding_context = await self._extract_binding_context_with_llm(state, llm)
            enhanced_prompt = self.system_prompt + f"\n\n# Current Binding Context\n{json.dumps(binding_context, ensure_ascii=False, indent=2)}"

            user_context = ""
            if state.get("history"):
                for msg in state["history"][-3:]:
                    user_context += f"{msg['role']}: {msg['content']}\n"
                user_context = "\n이전 대화:\n" + user_context

            user_query = f"사용자 질문: {state.get('user_query', '')}\n{user_context}현재 단계: {current_step + 1}/{max_steps}"

            response = await llm.ainvoke([
                SystemMessage(content=enhanced_prompt),
                HumanMessage(content=user_query),
            ])
            subtasks = self._parse_planner_response(response.content)
            return {**state, "subtasks": subtasks, "current_step": current_step + 1, "next": "executor", "user_query": state.get("user_query", "")}

        except Exception:
            return {**state, "next": "synthesizer"}

    def _parse_planner_response(self, response_content: str) -> List[Dict[str, Any]]:
        def normalize(subtask: Any, idx: int) -> Dict[str, Any]:
            if isinstance(subtask, dict):
                return {
                    "id": subtask.get("subtask_id", subtask.get("id", idx + 1)),
                    "goal": subtask.get("goal", subtask.get("description", str(subtask))),
                    "task_type": subtask.get("task_type", "THINK"),
                    "verdict": subtask.get("verdict", False),
                    "dependencies": subtask.get("dependencies", []),
                    "bindings": subtask.get("bindings", {}),
                }
            return {"id": idx + 1, "goal": str(subtask), "task_type": "THINK", "verdict": False, "dependencies": [], "bindings": {}}

        try:
            parsed = json.loads(response_content)
            if isinstance(parsed, dict) and "subtasks" in parsed:
                return [normalize(s, i) for i, s in enumerate(parsed["subtasks"])]
            if isinstance(parsed, list):
                return [normalize(s, i) for i, s in enumerate(parsed)]
            return [normalize(parsed, 0)]
        except json.JSONDecodeError:
            return [{"id": 1, "goal": response_content, "task_type": "THINK", "verdict": False, "dependencies": [], "bindings": {}}]


planner_node = PlannerNode()
