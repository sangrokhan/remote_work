# Agent Guidelines for A2A

## Testing Protocol

When modifying the A2A system, you MUST perform the following testing procedures to ensure stability.

### 1. Environment Verification
- Always use the provided virtual environment: `venv`
- Ensure dependencies are installed: `pip install -r requirements.txt`

### 2. Execution
- Run the full test suite after ANY code change:
  ```bash
  source venv/bin/activate
  python -m unittest discover tests
  ```

### 3. CI/CD Checks
- If you add new agents or logic, add corresponding unit tests in `tests/`.
- Ensure `test_workflow.py` passes. It mocks the LLM but runs the real MCP integration.

### 4. Common Pitfalls
- **MCP Path**: Ensure `utils/mcp_client_helper.py` uses `sys.executable` to launch the server, otherwise it may fail in environments where `python` is not in the system PATH or points to the wrong version.
- **Asyncio**: Agents often run in their own event loops. Be careful when testing async code synchronously.
