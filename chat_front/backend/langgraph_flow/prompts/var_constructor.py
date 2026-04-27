# System prompt for var_constructor_node — instructs LLM to construct query variables from retrieval output

# NOTE: previous_features, explicit_dependencies 는 현재 코드 어디에서도 소비되지 않음.
#       프롬프트에서도 제거하여 LLM 이 불필요한 토큰을 생성하지 않도록 함.
#       실제 사용되는 키는 query_entities 만임.
CONSTRUCTOR_SYSTEM_PROMPT = (
    "당신은 사용자 쿼리에서 바인딩 컨텍스트를 추출하는 어시스턴트입니다. "
    "JSON 형식으로 query_entities를 반환하세요."
    # "JSON 형식으로 query_entities, previous_features, explicit_dependencies를 반환하세요."
)
