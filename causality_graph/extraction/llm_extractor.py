import json
from dataclasses import dataclass, field

from causality_graph.extraction.md_parser import ParsedFeature
from causality_graph.llm import call_llm

EXTRACTION_PROMPT = """You are a cellular network domain expert. Given a parsed vendor feature document, extract all causal relationships as structured triples.

Feature: {feature_id} ({name}, gen={gen})
Description: {description}

KPI Impacts from doc:
{kpi_impacts}

Controlling Parameters from doc:
{params}

Output ONLY valid JSON in this format:
{{
  "triples": [
    {{
      "from": "<node_id>",
      "to": "<node_id>",
      "relation": "AFFECTS|CONTROLLED_BY|DEPENDS_ON|CORRELATES",
      "direction": "+|-|null",
      "magnitude": "low|medium|high|null",
      "condition": "<string or empty>",
      "confidence": <0.0-1.0>
    }}
  ]
}}

Rules:
- Use exact node IDs from the document (feature:X, kpi:X, param:X)
- Only output relationships explicitly supported by the document
- Set confidence < 0.8 for inferred or ambiguous relationships
"""


@dataclass
class ExtractionResult:
    source_feature_id: str
    triples: list[dict] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


class LLMExtractor:
    def __init__(self, client=None, model: str = "claude-sonnet-4-6"):
        self._client = client
        self._model = model

    def _build_prompt(self, feature: ParsedFeature) -> str:
        return EXTRACTION_PROMPT.format(
            feature_id=feature.feature_id,
            name=feature.name,
            gen=feature.gen,
            description=feature.description,
            kpi_impacts=json.dumps(feature.kpi_impacts, indent=2),
            params=json.dumps(feature.controlling_params, indent=2),
        )

    def extract(self, feature: ParsedFeature) -> ExtractionResult:
        prompt = self._build_prompt(feature)
        raw = call_llm(self._client, self._model, prompt, max_tokens=2048)
        try:
            data = json.loads(raw)
            triples = data.get("triples", [])
            return ExtractionResult(source_feature_id=feature.feature_id, triples=triples)
        except (json.JSONDecodeError, KeyError) as e:
            return ExtractionResult(
                source_feature_id=feature.feature_id,
                triples=[],
                parse_errors=[str(e)],
            )
