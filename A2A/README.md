# A2A (Agent-to-Agent) System

## Testing

This project uses `unittest` for testing.

## Deployment

To run the full system including the local LLM:

```bash
docker compose up -d
```

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
