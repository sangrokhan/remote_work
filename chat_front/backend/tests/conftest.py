"""
pytest configuration. Adds the repo root to sys.path so langgraph_flow
is importable without installation during test runs.
"""
import sys
import os

# Add repo root so `langgraph` package is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
