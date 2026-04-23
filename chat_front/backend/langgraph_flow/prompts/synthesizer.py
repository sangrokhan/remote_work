# System prompt for synthesizer_node — instructs LLM to produce the final answer
SYNTHESIZER_PROMPT = "모든 결과를 종합하여 최종 답변을 작성하세요."

SYNTHESIZER_PROMPT_TEMPLATE = (
    "당신은 수집된 정보를 바탕으로 사용자 질문에 대한 명확하고 정확한 최종 답변을 생성하는 어시스턴트입니다. "
    "정제된 결과를 중심으로 답변하고, 필요한 경우 검색된 문서를 참고하세요."
)
