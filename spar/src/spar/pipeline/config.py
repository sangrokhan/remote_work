from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphConfig:
    name: str
    use_query_expansion: bool = False
    use_prepare_context: bool = False
    use_reranker: bool = False
    use_real_generate: bool = False
    use_verify_loop: bool = False


PRESET_CONFIGS: list[GraphConfig] = [
    GraphConfig(name="baseline"),
    GraphConfig(name="+reranker", use_reranker=True),
    GraphConfig(name="+qexpand", use_query_expansion=True),
    GraphConfig(name="+context", use_prepare_context=True),
    GraphConfig(
        name="full_retrieval",
        use_query_expansion=True,
        use_prepare_context=True,
        use_reranker=True,
    ),
    GraphConfig(
        name="e2e",
        use_query_expansion=True,
        use_prepare_context=True,
        use_reranker=True,
        use_real_generate=True,
    ),
    GraphConfig(
        name="verify_loop",
        use_query_expansion=True,
        use_prepare_context=True,
        use_reranker=True,
        use_real_generate=True,
        use_verify_loop=True,
    ),
]
