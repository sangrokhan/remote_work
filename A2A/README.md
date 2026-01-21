# A2A (Agent-to-Agent) System

## Testing

This project uses `unittest` for testing.

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

To run all tests (Unit + Integration):

```bash
python -m unittest discover tests
```

### Test Structure

- `tests/test_base.py`: Unit tests for `A2AMessage` and `BaseAgent`.
- `tests/test_manager.py`: Unit tests for `ManagerAgent` logic.
- `tests/test_workflow.py`: Integration test for the full workflow (requires MCP server).
