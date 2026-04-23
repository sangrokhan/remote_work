# System prompt for var_binder_node — instructs LLM to bind retrieval results into state variables

BINDER_SYSTEM_PROMPT = (
    "당신은 이전 subtask 결과에서 바인딩 변수를 해결하는 어시스턴트입니다. "
    "bindings의 각 참조($task_N.field)를 previous results에서 찾아 JSON으로 반환하세요."
)
