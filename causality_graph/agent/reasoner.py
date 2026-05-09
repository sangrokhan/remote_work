import json
from dataclasses import dataclass, field

REASONING_PROMPT = """You are a cellular network optimization expert. Given a subgraph of causal relationships, answer the user's question.

Question: {question}

Relevant graph context:
Nodes:
{nodes}

Edges (causal relationships):
{edges}

Output ONLY valid JSON:
{{
  "answer": "<concise answer with specific parameter recommendations>",
  "reasoning_chain": ["<step 1>", "<step 2>", ...],
  "source_nodes": ["<node_id>", ...]
}}

Rules:
- Only reference nodes and edges present in the context above
- Be specific: include parameter names and recommended values when available
- reasoning_chain should show the inference path from KPI to parameter
"""


@dataclass
class ReasoningResult:
    answer: str
    reasoning_chain: list[str] = field(default_factory=list)
    source_nodes: list[str] = field(default_factory=list)


class Reasoner:
    def __init__(self, client=None, model: str = "claude-sonnet-4-6"):
        self._client = client
        self._model = model

    def answer(self, question: str, subgraph: dict) -> ReasoningResult:
        prompt = REASONING_PROMPT.format(
            question=question,
            nodes=json.dumps(subgraph["nodes"], indent=2),
            edges=json.dumps(subgraph["edges"], indent=2),
        )
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            return ReasoningResult(
                answer=data["answer"],
                reasoning_chain=data.get("reasoning_chain", []),
                source_nodes=data.get("source_nodes", []),
            )
        except (json.JSONDecodeError, KeyError):
            return ReasoningResult(answer=raw, reasoning_chain=[], source_nodes=[])
