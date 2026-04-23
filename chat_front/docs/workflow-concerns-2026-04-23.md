# Workflow Logic Review — Concerns

Date: 2026-04-23
Scope: `backend/langgraph_flow/` (graph, nodes, edges, state, core, tools)
Recent context: commits `d9f7e0d5` (double step increment, THINK verdict loop), `33c495cb` (planner/synthesizer tweaks), `ddb71dbf` (temp=0.7 bind).

## Summary

21 concerns across graph topology, state merging, routing logic, LLM invocation, tool binding, and observability. 5 high-severity items can cause silent termination, infinite loops, or crashed subtasks under normal load.

---

## HIGH

### H1. Double step increment in `route_after_refiner`
- File: `agents/edges/routing_logic.py:45`
- Symptom: routing adds `+1` before `max_step` check while refiner also increments via `update_state(current_step=current_step+1)`. Effective cap = `max_steps / 2`.
- Fix: remove increment from routing function. Step mutation owned by refiner only.

### H2. Missing LLM config injection in `executor_node`
- File: `agents/nodes/executor_node.py` (`invoke()`)
- Symptom: executor calls `_execute_think_subtask(goal, llm)` but never extracts `llm` from `config["configurable"]`. THINK subtasks always receive `None`.
- Fix: `llm = config.get("configurable", {}).get("llm")` at top of `invoke()`, match pattern in planner/refiner.

### H3. Verdict loop infinite retry on `verdict=False`
- File: `agents/nodes/refiner_node.py:refine_results()`
- Symptom: on `verdict=False` code bumps `retry_counts[subtask_id] += 1` and routes back var_binder → executor → refiner. On `retry_counts > max_retries` sets literal string `"exceeded"` (truthy) instead of boolean — loop may re-enter.
- Fix: route to synthesizer once `retry_counts[subtask_id] >= max_retries`. Use `verdict=True` with `forced=True` flag, not string sentinel.

### H4. Unresolved bindings leak into executor
- File: `agents/nodes/var_binder_node.py:_resolve_bindings_fallback()`
- Symptom: fallback sets `resolved[key] = f"unresolved_{task_id}_{field_name}"`. Executor uses this string verbatim as retrieval query → empty/garbage results.
- Fix: detect unresolved bindings, set subtask `verdict=False, reason="unresolved_bindings"`, skip execution, route to refiner with retry.

### H5. `merge_subtasks` discards prior subtask results
- File: `agents/state.py` (`merge_subtasks`)
- Symptom: reducer returns `new` unconditionally. Any re-plan overwrites all completed subtasks; history lost.
- Fix: append-merge by `id` (dedupe, preserve `status` from old if `completed`). Pattern in `merge_subtask_results` correct — copy it.

---

## MEDIUM

### M6. `route_after_executor` ignores `retriever_outputs`
- File: `agents/edges/routing_logic.py:25`
- Symptom: routes to refiner only when `retriever_history[].subtask_id == current_executing_id`. THINK subtasks write to `retriever_outputs` instead, so THINK results never trigger refiner.
- Fix: `any(h.get("subtask_id") == current_executing_id for h in state.get("retriever_outputs", []))` additional predicate.

### M7. LLM temperature inconsistency
- Files: all nodes
- Symptom: planner=0.2, constructor=0.7, executor=0.7, refiner=0.7, var_binder=0.1, synthesizer=0.4. Refiner at 0.7 may flip verdicts non-deterministically; constructor/executor too creative for structured JSON output.
- Fix: planning/refining=0.1, retrieval/binding=0.0, synthesis=0.3. Centralize in `config.py`.

### M8. Null-LLM fallback inconsistent across nodes
- Files: planner has dev-mode fallback; refiner, retriever, var_constructor do not.
- Symptom: `llm=None` → crash in refiner/retriever/var_constructor.
- Fix: add test-mode fallbacks (return canned shape), or fail fast at graph entry if LLM missing.

### M9. Tool registry miss crashes executor
- File: `agents/nodes/executor_node.py:_execute_retrieve_subtask()`
- Symptom: `tool = tool_registry.get_tool(tool_name)` → `None` → `raise ValueError`. Whole run dies.
- Fix: log warn, set verdict=False with `reason="tool_not_found"`, continue.

### M10. `route_after_planner` `is_finished` check never reachable first pass
- File: `agents/edges/routing_logic.py:15`
- Symptom: only synthesizer sets `is_finished=True`; planner cannot set it, so check dead on first invoke.
- Fix: also branch on `len(subtasks) == 0` to short-circuit to synthesizer for trivial queries.

### M11. `var_constructor` fallback omits `query_entities`
- File: `agents/nodes/var_constructor_node.py:construct_binding_context()`
- Symptom: LLM timeout → fallback dict missing `query_entities`; var_binder unpacks `None`.
- Fix: initialize all required keys in fallback.

### M12. Subtask ID collision across re-plans
- File: `agents/nodes/planner_node.py:_parse_planner_response()`
- Symptom: `idx + 1` IDs reset each plan call. Old `task_0` and new `task_0` collide in `execution_history` / binding resolution.
- Fix: `id = str(uuid4())` or monotonic counter in state.

### M13. No schema validation on LLM JSON responses
- Files: all LLM-calling nodes
- Symptom: `json.loads()` may succeed but missing required keys (`subtasks`, `verdict`, `bindings`). KeyError mid-run.
- Fix: pydantic model per node or explicit `required_keys` check with fallback.

### M14. Retriever dedup uses truncated query
- File: `agents/nodes/retriever_node.py:invoke()`
- Symptom: `logged_entries.add((subtask_id, query[:100]))` — two 100-char queries with same prefix treated equal.
- Fix: `hashlib.md5(query.encode()).hexdigest()`.

---

## LOW

### L15. SSE payload schema drift
- File: `agents/graph.py:stepby_invoke()`
- Symptom: `node_started` empty payload; `node_finished` payload varies per node. Frontend cannot build deterministic state machine.
- Fix: uniform `{key: v for k, v in delta.items() if k != "next"}`.

### L16. `print()` debug in production
- Files: all nodes
- Fix: replace with `logging.getLogger(__name__).debug(...)`.

### L17. `execution_history` shallow-merge loses retries
- File: `agents/state.py` (`execution_history` reducer)
- Symptom: `{**old, **new}` overwrites list under same `subtask_id`; first-attempt record lost.
- Fix: deep merge — concatenate lists per key.

### L18. `max_steps` not enforced inside planner
- File: `agents/edges/routing_logic.py:45`
- Symptom: planner can emit new subtasks even when `current_step >= max_steps`; only edge function gates.
- Fix: planner short-circuits to synthesizer when cap reached.

### L19. No END fallback from planner/executor/refiner
- File: `agents/graph.py:_build()`
- Symptom: invalid return from route function → graph hangs.
- Fix: validate route return value ∈ {known nodes} else → synthesizer → END.

### L20. Binding resolver case-sensitivity
- File: `agents/nodes/var_binder_node.py:_resolve_bindings_fallback()`
- Symptom: `if 'feature' in line.lower()` then stores original case — later regex fails variant caps.
- Fix: normalize consistently.

### L21. No LLM timeout
- Files: all nodes calling `llm.ainvoke`
- Symptom: hung provider blocks run forever, SSE stream stalls.
- Fix: `await asyncio.wait_for(llm.ainvoke(...), timeout=30)` + catch → verdict=False.

---

## Cross-cutting recommendations

1. Single source of truth for step counter — refiner only, routing read-only.
2. Pydantic schemas per node IO → kill class M13/M11/H4 crashes at boundary.
3. Timeout + retry wrapper around `llm.ainvoke` (class L21, M8).
4. Centralize temperatures in `config.py` (M7).
5. Structured logger w/ run_id + subtask_id correlation keys replaces `print()`.
6. Reducer audit — all mutable collections use append-merge semantics or explicit replace flag.

---

## Priority order for fixes

H1 → H2 → H4 → H5 → H3 → M6 → M9 → M12 → M13 → rest.
