from __future__ import annotations

import json
import logging

from spar.llm import LLMRole, get_client
from spar.prompts import load_prompt
from spar.router.schemas import Route, RouteResult

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("router_system.txt")


class LLMRouter:
    async def route(self, query: str) -> RouteResult:
        try:
            client = await get_client(LLMRole.ROUTER)
            raw = await client.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=128,
            )
            data = json.loads(raw)
            return RouteResult(
                route=Route(data["route"]),
                confidence=float(data.get("confidence", 0.8)),
                layer="llm",
                entities=data.get("entities", {}),
                product=data.get("product"),
                release=data.get("release"),
            )
        except Exception as exc:
            _log.warning("LLMRouter fallback — %s: %s", type(exc).__name__, exc)
            return RouteResult(route=Route.DEFAULT_RAG, confidence=0.0, layer="llm_fallback")
