# Call Flow — 전처리 파이프라인 (Ingest Time)

> 문서를 Milvus에 적재하기 전 실행되는 흐름.  
> 코드 기준: `src/spar/`

---

```
scripts/run_ingest.py
    │
    ├─ parse_document(file_path)
    │     [src/spar/parsers/docx_parser.py | pdf_parser.py | excel_parser.py]
    │     └─ raw text + metadata 추출
    │
    ├─ chunk_document(raw_text, metadata)
    │     [src/spar/ingest/chunker.py]
    │     └─ text → List[Chunk(text, parent_sections, doc_type, ...)]
    │
    ├─ tag_keywords(chunks)
    │     [src/spar/preprocessing/abbrev_mapper.py]
    │     ├─ load_acronyms() → dictionary/acronyms.json (2503 3GPP 항목)
    │     ├─ extract_terms(chunk.text) → matched_terms (Set[str])
    │     └─ chunk.keywords = matched_terms  (Milvus ARRAY 필드로 저장)
    │
    ├─ encode_chunks(chunks)
    │     [src/spar/encoder/registry.py → SentenceTransformerEncoder]
    │     ├─ get_encoder() → 싱글턴 인스턴스 (BGE-large-en-v1.5)
    │     └─ encoder.encode([chunk.text]) → np.ndarray [n, 1024]
    │
    └─ ingest_to_milvus(chunks, vectors)
          [src/spar/retrieval/milvus_client.py → SparMilvusClient]
          ├─ Collection schema: chunk_id, embedding(1024d), text,
          │    doc_type, keywords[], sparse_vec (BM25), parent_sections[], ...
          ├─ HNSW index (M=16, efConstruction=200)
          └─ BM25 sparse index
```

---

## 컴포넌트 참조

| 역할 | 파일 | 핵심 클래스/함수 |
|------|------|----------------|
| 파서 | `src/spar/parsers/` | `docx_parser`, `pdf_parser`, `excel_parser` |
| 청커 | `src/spar/ingest/chunker.py` | `chunk_document()` |
| 약어 태깅 | `src/spar/preprocessing/abbrev_mapper.py` | `load_acronyms()`, `extract_terms()` |
| 인코더 | `src/spar/encoder/registry.py` | `SentenceTransformerEncoder`, `get_encoder()` |
| Milvus 적재 | `src/spar/retrieval/milvus_client.py` | `SparMilvusClient` |
| 약어 사전 | `dictionary/acronyms.json` | 3GPP 약어 2503건 |
