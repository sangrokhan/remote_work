from __future__ import annotations

import json
import os

from openai import AsyncOpenAI

from spar.router.schemas import Route, RouteResult

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
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str = "dummy",
    ) -> None:
        self._base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1")
        self._model = model or os.environ.get("LLM_ROUTER_MODEL", "qwen2.5-7b-instruct")
        self._client = AsyncOpenAI(base_url=self._base_url, api_key=api_key)

    async def route(self, query: str) -> RouteResult:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            data = json.loads(resp.choices[0].message.content)
            return RouteResult(
                route=Route(data["route"]),
                confidence=float(data.get("confidence", 0.8)),
                layer="llm",
                entities=data.get("entities", {}),
                product=data.get("product"),
                release=data.get("release"),
            )
        except Exception:
            return RouteResult(route=Route.DEFAULT_RAG, confidence=0.0, layer="llm_fallback")
