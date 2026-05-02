# Call Flow — Backend 쿼리 처리 (Query Time)

> `POST /query` 요청 → LLM 응답까지 전체 흐름.  
> 코드 기준: `src/spar/`

---

## 1. API Layer

```
Client
  │  POST /query
  │  { query, product, release, top_k, history }
  ▼
src/spar/api/app.py — FastAPI
  │
  ├─ [startup: app.lifespan()]
  │     ├─ encoder   = get_encoder()         # SentenceTransformerEncoder (singleton)
  │     ├─ reranker  = get_reranker()        # CrossEncoderClient (HTTP) or Local
  │     ├─ router    = HybridRouter(encoder)
  │     ├─ milvus    = SparMilvusClient(config)
  │     └─ _graph    = build_graph(router, reranker, encoder, milvus)
  │
  └─ POST /query
        ├─ QueryRequest → initial SparState
        └─ await _graph.ainvoke(initial_state)
```

---

## 2. LangGraph Pipeline

```
src/spar/pipeline/graph.py — StateGraph
  │
  ├─ [preprocess]
  │     src/spar/pipeline/nodes.py — Nodes.preprocess()
  │     ├─ expand_query(state["query"], acronyms)
  │     │     [src/spar/preprocessing/abbrev_mapper.py]
  │     │     ├─ "SMF" → "SMF (Session Management Function)"
  │     │     └─ 모호한 약어: LLM conflict resolution
  │     ├─ extract_terms(expanded_query) → matched_terms
  │     └─ state out: { expanded_query, matched_terms }
  │
  ├─ [rewrite_query]
  │     Nodes.rewrite_query()
  │     ├─ 입력: query + history (이전 대화)
  │     ├─ _rewrite_query(query, history, acronyms)
  │     │     └─ LLMClient.chat(rewrite_prompt)
  │     └─ state out: { rewritten_query }
  │
  ├─ [prepare_context]
  │     Nodes.prepare_context()
  │     ├─ build_context(query, history, acronyms)
  │     └─ state out: { history_context }
  │
  ├─ [route]
  │     Nodes.route()
  │     └─ HybridRouter.route(rewritten_query)
  │           [src/spar/router/hybrid_router.py]
  │           ├─ Layer 1 — RegexRouter: spec 번호 패턴 매칭
  │           ├─ Layer 2 — EmbeddingRouter: 벡터 유사도 분류
  │           └─ Layer 3 — LLMRouter: 애매한 케이스 LLM 판단
  │           → RouteResult(route, entities, layer, needs_decomposition)
  │           → state out: { route_result }
  │
  ├─ [conditional] — _route_selector(state)
  │     route_result.needs_decomposition → [decompose]
  │     route_result.route:
  │       RAG         → [rag_retrieve]
  │       STRUCTURED  → [structured_retrieve]
  │       MULTI_HOP   → [multi_hop_retrieve]
  │
  ├─ [*_retrieve]
  │     Nodes.rag_retrieve() | structured_retrieve() | multi_hop_retrieve()
  │     │
  │     ├─ doc_types = doc_types_for_route(route_result)
  │     ├─ expr      = build_expr(route_result, matched_terms)
  │     │               → Milvus filter expr (e.g., spec_number == '38.300')
  │     ├─ vector    = encoder.encode([rewritten_query])  # [1, 1024]
  │     │
  │     └─ SparMilvusClient.hybrid_search(
  │             doc_type, query_text, query_vector, top_k, expr)
  │           [src/spar/retrieval/milvus_client.py]
  │           ├─ Dense ANN search (HNSW)
  │           ├─ Sparse BM25 search
  │           ├─ RRFRanker (reciprocal rank fusion)
  │           └─ asyncio.gather() — doc_type 복수 시 병렬 실행
  │           → List[dict] raw_chunks (scored, sorted desc)
  │     state out: { raw_chunks }
  │
  ├─ [rerank]
  │     Nodes.rerank()
  │     └─ CrossEncoderClient.rerank(query, [c["text"] for c in raw_chunks])
  │           [src/spar/reranker/client.py]
  │           POST {model, query, documents} → List[float] relevance_scores
  │           → sort chunks by score → top_k
  │     state out: { reranked_chunks }
  │
  ├─ [generate]
  │     Nodes.generate()
  │     ├─ prompt  = load_prompt("answer_generation.txt")
  │     ├─ context = reranked_chunks + history_context
  │     └─ LLMClient.chat(messages, temperature=0.0, max_tokens=1024)
  │           [src/spar/llm/client.py]
  │           AsyncOpenAI (OpenAI-compatible endpoint)
  │     state out: { answer }
  │
  └─ [verify]  (선택적)
        Nodes.verify()
        ├─ LLM self-eval: 답변 품질 1-5점 채점
        ├─ score < 3 and retry_count < 3 → [tool_call] fallback
        └─ score ≥ 3 → END
```

---

## 3. API Response

```
SparState → QueryResponse
  {
    request_id : UUID
    query      : str          # 원본 쿼리
    route      : str          # Route enum value
    answer     : str          # LLM 생성 답변
    sources    : List[dict]   # reranked_chunks
    latency_ms : float
  }
```

---

## 4. 컴포넌트 참조

| 역할 | 파일 | 핵심 클래스/함수 |
|------|------|----------------|
| API 진입점 | `src/spar/api/app.py` | FastAPI, `/query` endpoint |
| 파이프라인 조립 | `src/spar/pipeline/graph.py` | `build_graph()`, `StateGraph` |
| 파이프라인 상태 | `src/spar/pipeline/state.py` | `SparState` (TypedDict) |
| 노드 구현체 | `src/spar/pipeline/nodes.py` | `Nodes` dataclass, async 메서드 |
| 전처리 노드 | `src/spar/preprocessing/abbrev_mapper.py` | `expand_query()`, `extract_terms()` |
| 라우터 | `src/spar/router/hybrid_router.py` | `HybridRouter.route()` |
| 라우터 스키마 | `src/spar/router/schemas.py` | `Route`, `RouteResult` |
| 인코더 | `src/spar/encoder/registry.py` | `SentenceTransformerEncoder`, `get_encoder()` |
| 리랭커 | `src/spar/reranker/client.py` | `CrossEncoderClient`, `LocalCrossEncoderClient` |
| 검색 | `src/spar/retrieval/milvus_client.py` | `SparMilvusClient.hybrid_search()` |
| LLM | `src/spar/llm/client.py` | `LLMClient.chat()` |
| 프롬프트 | `src/spar/prompts/` | `answer_generation.txt` 외 |
