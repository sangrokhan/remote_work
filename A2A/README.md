# A2A (Agent-to-Agent) System

## Testing

This project uses `unittest` for testing.

## Deployment

시스템은 App, Frontend, LLM 서비스로 구성됩니다. GPU 가속 방식에 따라 플랫폼별 설정이 다릅니다.

### 🍎 Mac (Apple Silicon) - 주 개발 환경
Mac GPU (MPS)를 활용하기 위해 LLM 서버는 **네이티브**로, 나머지는 **Docker**로 실행합니다.

1. **LLM 서버 실행 (네이티브):**
   ```bash
   ./run_llm_mac.sh
   ```
   *포트 8000번에서 작동하며, Docker 앱은 `host.docker.internal`을 통해 연결됩니다.*

2. **App & Frontend 실행 (Docker):**
   ```bash
   docker compose up -d
   ```

### 🐧 Linux (NVIDIA GPU)
NVIDIA CUDA 가속을 Docker 내부에서 100% 활용합니다.

1. **전체 시스템 실행:**
   ```bash
   docker compose -f docker-compose.linux.yml up -d --build
   ```
   *(호스트에 `nvidia-container-toolkit` 설치가 필요합니다.)*

---

### Prerequisites

- Python 3.10+
- Virtual Environment (recommended)

### Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Tests

### Running Tests

To run all tests (Unit + Integration):

```bash
python -m unittest discover tests
```

### Test Structure

- `tests/test_workflow.py`: Integration test for the full A2A workflow (Summarizer -> Emailer).
- `tests/integration/test_llm_serving.py`: Verifies model downloading and local LLM serving logic.
