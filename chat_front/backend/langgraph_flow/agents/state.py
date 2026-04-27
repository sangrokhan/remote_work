"""
AgentState — shared state TypedDict passed between all LangGraph nodes.
Every node reads from and writes to this dict; LangGraph merges partial returns.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict

KST = ZoneInfo("Asia/Seoul")


class InputState(TypedDict):
    user_query: str


def merge_subtasks(old: List[Dict], new: List[Dict]) -> List[Dict]:
    if not new:
        return old or []
    return new


def merge_subtask_results(old: List[Dict], new: List[Dict]) -> List[Dict]:
    if not old:
        return new or []
    if not new:
        return old

    def _key(item: Dict) -> tuple:
        return (item["id"], item.get("attempt", 0))

    merged: dict = {_key(item): item for item in old if "id" in item}
    no_id = [item for item in old if "id" not in item]
    for item in new:
        if "id" in item:
            merged[_key(item)] = item
        else:
            no_id.append(item)
    return list(merged.values()) + no_id


def merge_reference_features(old: List[Dict], new: List[Dict]) -> List[Dict]:
    if not old:
        return new or []
    if not new:
        return old
    existing_ids = {item.get("feature_id") for item in old if item.get("feature_id")}
    result = list(old)
    for item in new:
        feature_id = item.get("feature_id")
        if feature_id and feature_id not in existing_ids:
            result.append(item)
            existing_ids.add(feature_id)
    return result


def merge_retriever_history(old: List[Dict], new: List[Dict]) -> List[Dict]:
    if not old:
        old = []
    if not new:
        return old

    # Replace ops: drop existing rows for given subtask_id before merging
    replace_subtask_ids = {
        item.get("subtask_id")
        for item in new
        if item.get("_op") == "replace_subtask" and item.get("subtask_id") is not None
    }
    if replace_subtask_ids:
        old = [h for h in old if h.get("subtask_id") not in replace_subtask_ids]

    cleaned_new = [
        {k: v for k, v in item.items() if k != "_op"}
        for item in new
    ]

    existing_keys = {(item.get("subtask_id"), (item.get("query", ""))[:100]) for item in old}
    result = list(old)
    for item in cleaned_new:
        key = (item.get("subtask_id"), (item.get("query", ""))[:100])
        if key not in existing_keys:
            result.append(item)
            existing_keys.add(key)
    return result


class AgentState(TypedDict):
    user_query: Annotated[str, lambda old, new: new if new and new.strip() else (old if old else "")]
    history: Annotated[List[Dict[str, Any]], add_messages]
    agent_flow_context: Annotated[List[Dict[str, Any]], lambda old, new: (old or []) + (new or [])]
    retriever_outputs: Annotated[List[Dict[str, Any]], lambda old, new: new if new is not None else (old or [])]
    retriever_history: Annotated[List[Dict[str, Any]], merge_retriever_history]
    reference_features: Annotated[List[Dict[str, str]], merge_reference_features]

    subtasks: Annotated[List[Dict[str, Any]], merge_subtasks]
    subtask_results: Annotated[List[Dict[str, Any]], merge_subtask_results]
    current_executing_subtask_id: Annotated[Optional[int], lambda old, new: new if new is not None else old]
    resolved_bindings: Annotated[Dict[str, Any], lambda old, new: new if new is not None else old]

    execution_history: Annotated[Dict[int, List[Dict[str, Any]]], lambda old, new: {**old, **new} if old and new else (old or new or {})]
    retry_counts: Annotated[Dict[int, int], lambda old, new: {**old, **new} if old and new else (old or new or {})]
    current_step: Annotated[int, lambda old, new: new if new is not None else old]
    max_steps: Annotated[int, lambda old, new: new if new is not None else old]
    is_finished: Annotated[bool, lambda old, new: new if new is not None else old]
    final_response: Annotated[str, lambda old, new: new if new is not None else old]

    binding_context: Annotated[Dict[str, Any], lambda old, new: new if new is not None else old]
    messages: Annotated[List[Dict[str, Any]], add_messages]
    next: Annotated[str, lambda old, new: new if new is not None else old]


def create_initial_state(user_query: str) -> AgentState:
    return AgentState(
        user_query=user_query,
        history=[],
        agent_flow_context=[],
        retriever_outputs=[],
        retriever_history=[],
        reference_features=[],
        subtasks=[],
        subtask_results=[],
        current_executing_subtask_id=None,
        resolved_bindings={},
        execution_history={},
        retry_counts={},
        current_step=0,
        max_steps=10,
        is_finished=False,
        final_response="",
        binding_context={},
        messages=[],
        next="var_constructor",
    )


def update_state(current_state: AgentState, **kwargs) -> AgentState:
    new_state = AgentState(**current_state)
    new_state.update(kwargs)
    return new_state
