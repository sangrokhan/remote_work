# llm/ — LLM 클라이언트 모듈

## 역할

OpenAI-compatible 엔드포인트 기반 LLM 클라이언트 팩토리 + 싱글톤 레지스트리.  
`LLMBackend` 프로토콜을 따르는 어떤 백엔드든 교체 가능하도록 설계.

## 파일 맵

| 파일 | 역할 |
|---|---|
| `client.py` | `LLMClient` — AsyncOpenAI 래퍼 (chat completions) |
| `config.py` | `LLMSettings` — pydantic-settings로 `.env` 로드 (main/router URL, Gemini 토글) |
| `factory.py` | `LLMFactory.create(role, settings)` — role별 클라이언트 생성, Gemini fallback 주입 |
| `fallback.py` | `LLMBackend` Protocol + `FallbackLLMClient` — primary 1회 시도 → last에 지수 백오프 재시도 |
| `gemini_cli.py` | `GeminiCliClient` — headless `gemini --yolo` CLI 어댑터 (dev/외부 전용, 기본 비활성) |
| `registry.py` | `get_client(role)` — 프로세스 전역 캐시, 첫 호출 시 초기화 |
| `__init__.py` | `get_client`, `LLMRole` 재노출 |

## 핵심 인터페이스

```python
class LLMBackend(Protocol):
    @property
    def model(self) -> str: ...
    async def chat(self, messages: list[dict], *, temperature: float, max_tokens: int) -> str: ...
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLM_MAIN_URL` | `http://localhost:8000/v1` | 메인 생성 모델 |
| `LLM_MAIN_MODEL` | `qwen2.5-72b-instruct` | |
| `LLM_ROUTER_URL` | `http://localhost:8001/v1` | 라우터 소형 모델 |
| `LLM_ROUTER_MODEL` | `qwen2.5-7b-instruct` | |
| `GEMINI_CLI_FALLBACK_ENABLED` | `false` | 온프레미스 환경에서는 반드시 `false` |

## 규약

- 새 백엔드 추가 시 `LLMBackend` Protocol 구현 후 `LLMFactory`에만 등록
- `get_client()` 외부에서 `LLMClient` 직접 생성 금지 (테스트 제외)
- Gemini fallback은 개발/CI 환경 전용 — 온프레미스 배포 시 절대 활성화 금지
