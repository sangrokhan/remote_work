from __future__ import annotations

import json
import logging

from spar.llm import LLMRole, get_client
from spar.router.schemas import Route, RouteResult

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a query router for a Samsung RAN (Radio Access Network) documentation system.
Classify the user query into exactly one of these routes:
- structured_lookup: exact parameter/counter/alarm lookup
- definition_explain: explain what something is
- procedural: how-to, installation, configuration steps
- diagnostic: troubleshooting, root cause, why something failed
- comparative: comparing versions, features, configurations
- default_rag: anything else

Also extract:
- entities: dict of alarm_code, param_name, mo_name, feature_name if present
- product: "LTE" | "NR" | "both" | null
- release: version string like "v6.0" | null

Respond ONLY with valid JSON:
{"route": "<route>", "confidence": <0.0-1.0>, "entities": {}, "product": null, "release": null}"""


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
