# System prompt for var_constructor_node — instructs LLM to construct query variables from retrieval output

CONSTRUCTOR_SYSTEM_PROMPT = (
    "당신은 사용자 쿼리에서 바인딩 컨텍스트를 추출하는 어시스턴트입니다. "
    "JSON 형식으로 query_entities, previous_features, explicit_dependencies를 반환하세요."
)
