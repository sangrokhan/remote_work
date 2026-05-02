# Call Flow — 전처리 파이프라인 (Ingest Time)

> 문서를 Milvus에 적재하기 전 실행되는 흐름.  
> 코드 기준: `src/spar/`

---

## 전체 구조: Two-Pass

```
┌──────────────────────────────────────────────────────────────────┐
│ Pass A: Entity Glossary Build (최초 1회, Excel 변경 시 재실행)    │
│                                                                  │
│  scripts/build_entity_glossary.py                               │
│      ├─ scan parameter_ref Excel → param_names, yang_paths      │
│      ├─ scan counter_ref Excel   → counter_names, groups         │
│      ├─ scan alarm_ref Excel     → alarm_ids, alarm_names        │
│      └─ → dictionary/samsung_entities.json                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │ enriched entity set
┌──────────────────────────▼───────────────────────────────────────┐
│ Pass B: Main Ingest                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Pass B: 텍스트 문서 (DOCX/PDF/MD)

```
scripts/run_ingest.py
    │
    ├─ parse_document(file_path)
    │     [src/spar/parsers/docx_parser.py | pdf_parser.py]
    │     └─ raw text + metadata 추출
    │
    ├─ dispatch(raw_text, source_doc, doc_type=...)
    │     [src/spar/ingest/chunkers.py]
    │     └─ text → List[Chunk(text, parent_sections, doc_type, ...)]
    │
    ├─ tag_keywords(chunks)
    │     [src/spar/preprocessing/abbrev_mapper.py]
    │     ├─ load_acronyms() → dictionary/acronyms.json (2503 3GPP 항목)
    │     ├─ load_entity_glossary() → dictionary/samsung_entities.json
    │     ├─ get_all_keywords(acronyms, entities) → unified keyword set
    │     ├─ extract_terms(chunk.text, keywords) → matched_terms (Set[str])
    │     └─ chunk["keywords"] = matched_terms  (Milvus ARRAY 필드로 저장)
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

## Pass B: Reference Excel (Parameter/Counter/Alarm)

```
scripts/run_ingest.py
    │
    ├─ parse_*_ref_excel(file_path)
    │     [src/spar/parsers/parameter_ref_parser.py | counter_ref_parser.py | alarm_ref_parser.py]
    │     └─ Excel → List[ParameterRecord | CounterRecord | AlarmRecord]
    │
    ├─ dispatch_records(records, source_doc, doc_type=...)
    │     [src/spar/ingest/chunkers.py → src/spar/chunkers/reference_chunker.py]
    │     └─ 1 record = 1 Chunk (record.to_chunk_text() 사용)
    │
    ├─ tag_keywords(chunks)   ← 동일 (위 Pass B 참조)
    │
    ├─ encode_chunks(chunks)  ← 동일
    │
    └─ ingest_to_milvus(chunks, vectors)  ← 동일
```

---

## 컴포넌트 참조

| 역할 | 파일 | 핵심 클래스/함수 |
|------|------|----------------|
| Pass A 스크립트 | `scripts/build_entity_glossary.py` | `build_and_write()` |
| 파서 (텍스트) | `src/spar/parsers/` | `docx_parser`, `pdf_parser` |
| 파서 (Excel) | `src/spar/parsers/` | `parse_parameter_ref_excel`, `parse_counter_ref_excel`, `parse_alarm_ref_excel` |
| 청커 (텍스트) | `src/spar/ingest/chunkers.py` | `dispatch()` |
| 청커 (Excel) | `src/spar/ingest/chunkers.py` + `src/spar/chunkers/reference_chunker.py` | `dispatch_records()`, `chunk_parameter_ref()`, `chunk_counter_ref()`, `chunk_alarm_ref()` |
| 약어+엔티티 태깅 | `src/spar/preprocessing/abbrev_mapper.py` | `load_entity_glossary()`, `get_all_keywords()`, `extract_terms()` |
| 인코더 | `src/spar/encoder/registry.py` | `SentenceTransformerEncoder`, `get_encoder()` |
| Milvus 적재 | `src/spar/retrieval/milvus_client.py` | `SparMilvusClient` |
| 3GPP 약어 사전 | `dictionary/acronyms.json` | 3GPP 약어 2503건 |
| Samsung 엔티티 사전 | `dictionary/samsung_entities.json` | Pass A 생성 (파라미터/카운터/알람 엔티티명) |
