# Flow service registry — main.py selects by req.agentic_rag
from services.simple_flow import SimpleService
from services.agentic_rag_flow import AgenticService

__all__ = ["SimpleService", "AgenticService"]
