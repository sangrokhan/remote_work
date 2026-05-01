from __future__ import annotations

import json
import logging

from spar.llm import LLMRole, get_client
from spar.prompts import load_prompt

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("decompose_system.txt")
_MAX_SUB_QUESTIONS = 4


class QueryDecomposer:
    async def decompose(self, query: str) -> list[str]:
        try:
            client = await get_client(LLMRole.ROUTER)
            raw = await client.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=256,
            )
            sub_questions: list[str] = json.loads(raw)
            if not isinstance(sub_questions, list) or not sub_questions:
                return [query]
            return [q for q in sub_questions[:_MAX_SUB_QUESTIONS] if isinstance(q, str) and q.strip()]
        except Exception as exc:
            _log.warning("QueryDecomposer fallback — %s: %s", type(exc).__name__, exc)
            return [query]
