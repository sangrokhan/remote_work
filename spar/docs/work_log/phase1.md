# Phase 1 작업 이력

---

## Task 1.1 — 문서 유형 분류 및 파서 개발

### DocxParser (완료)

- `parsers/docx_parser.py` — heading 스타일 기반 섹션, 표→CSV, 이미지 추출
- `parsers/docx_config.py` — DocxParseConfig

**버그픽스**:
- 이미지+텍스트 혼합 단락 텍스트 소실
- 병합셀 비인접 중복
- 한글 slug `re.UNICODE` 누락
- 동일 rel_id 이미지 중복 저장

### PDF Extractor 패키지 (완료)

`parsers/extractor/` — pdfplumber 기반

- `pipeline.py` — 메인 오케스트레이터 `extract_pdf_to_outputs()`: 페이지별 추출, 크로스페이지 테이블 병합, MD+CSV+이미지 출력
- `text.py` — 폰트 메트릭 기반 헤딩/본문/불릿 추출; 워터마크(회전 53-57°, gray 0.88-0.96) 필터
- `tables.py` — 격자 분석 기반 테이블 검출, 멀티라인 셀 보존, 빈 열 제거, 크로스페이지 축 정렬 병합
- `images.py` — 임베디드 이미지 스트림 + 드로잉 오브젝트 클러스터링 → PNG/JPEG 저장 + MD 참조 마커
- `font_profile.py` — `--profile-fonts`로 폰트 통계 JSON/CSV 생성 (헤딩 룰 튜닝용)
- `notes.py` — 각주/주석 영역 검출 (청색 구분선, 색상 패턴), 테이블 추출 충돌 방지
- `raw.py` — PDF 페이지 subset → base64 JSON 직렬화/역직렬화
- `shared.py` — bbox 연산, `--pages` 파싱(예: "1,3-5"), 텍스트 정규화
- `__main__.py` — CLI: `python -m spar.parsers.extractor <pdf> [--pages 1-3,5] [--profile-fonts] [--from-raw]`

### Counter Reference Excel 파서 (완료, 2026-05-02)

브랜치: `feat/counter-ref-parser` → main merged (`2864f87f`)

- `parsers/counter_ref_parser.py` — `CounterRecord`, `parse_counter_ref_excel()`
  - `_expand_merged_cells()` 병합셀 전파
  - value_range 분리
  - 한/영 헤더 별칭 지원
- `data/samples/counter_ref_sample.xlsx` — 병합 셀 포함 13개 카운터 샘플 (RRC/MAC/PHY 3개 대그룹, 5개 중그룹)
- `tests/parsers/test_counter_ref_parser.py` — 29개 테스트
- `scripts/ingest_counter_ref.py` — Excel → Milvus ingest CLI (`--dry-run`/`--force`/`--sheet`/`--product`/`--release`)
- `router/regex_router.py` — G-\d{3,5} (conf 1.0) / CELL.X.Y (conf 0.95) 패턴 → STRUCTURED_LOOKUP 추가

### Alarm Reference Excel 파서 (완료, 2026-05-02)

브랜치: `feat/alarm-ref-sample-and-parser`

- `parsers/alarm_ref_parser.py` — `AlarmRecord` (alarm_id/alarm_name/severity/category/module/pdf_ref) + `parse_alarm_ref_excel()`
  - `to_chunk_text()` / `to_keywords()` / `to_dict()`
  - alarm_id 대문자 정규화
- `retrieval/alarm_index.py` — `AlarmIndex` 싱글톤 + `get_alarm_index()`
  - env `SPAR_ALARM_REF_PATH` 우선, 기본 샘플 fallback
  - `lookup()` 대소문자 무관, `search_by_name()` 부분일치
- `retrieval/routing.py::resolve_alarm_entity()` — alarm_code → AlarmIndex 직접 lookup → STRUCTURED_LOOKUP 단축 경로
- `data/samples/alarm_excel_ref_sample.xlsx` — 12행 샘플 (ALM-1001..ALM-1012)
- `scripts/gen_alarm_sample.py` — 재현 가능한 샘플 생성 스크립트
- 테스트: `tests/parsers/test_alarm_ref_parser.py` (5), `tests/retrieval/test_alarm_index.py` (7), `tests/retrieval/test_routing_alarm.py` (3)

**후속 과제**: Alarm Reference PDF 파서 (alarm_id를 join 키로 설명/조치 본문 결합)

---

## Task 1.3 — 청킹 전략 (부분 완료, 2026-05-01)

설계 문서: `docs/superpowers/plans/2026-05-01-md-ingest-pipeline.md`

- `src/spar/ingest/chunkers.py` — md-aware + fixed chunker
- `scripts/run_ingest.py` — md 전용 ingest 파이프라인 (PDF 입력 거부)

---

## Task 1.4 — Hybrid Search (완료, 2026-05-01)

merge commit: `916ed9f7`

- 임베딩 모델 결정: `BAAI/bge-large-en-v1.5` (대안: `intfloat/e5-large-v2`, `nomic-embed-text-v1.5`)
- Milvus 채택 결정: 2026-04-30 (온프레미스 운영성 + 대규모 인덱스)
- `src/spar/ingest/embedder.py` — verbose param: remote path `[done/N]\r` 배치 진행, local path `show_progress_bar` 활성화
- `src/spar/encoder/base.py`, `registry.py` — 싱글톤 (client.py/factory.py/config.py 제거)
- 설계 문서: `docs/superpowers/plans/2026-05-01-encoder-singleton.md`

---

## Task 1.5 — Reranker (구현 중, 2026-05-01)

- 모델 결정: `BAAI/bge-reranker-v2-m3` (대안: `jinaai/jina-reranker-v2-base-multilingual`)
- `src/spar/reranker/client.py` — Remote: vLLM HTTP 서빙; Local: `asyncio.to_thread` 래핑
- `src/spar/reranker/config.py` — `RERANKER_BACKEND`(local|remote) / `RERANKER_DEVICE`
- `src/spar/reranker/factory.py` — local/remote 분기
- 테스트: `tests/reranker/` (18개)

---

## Task 1.6 — 약어/동의어 사전 (완료, 2026-05-01)

브랜치: `feat/abbrev-mapping` → main merged (`89d8bfe5`)
브랜치: `feat/excel-term-dict` → main merged (`e22bbfe5`)

설계 문서:
- `docs/superpowers/specs/2026-04-30-abbrev-mapping-design.md`
- `docs/superpowers/specs/2026-05-01-excel-term-dict-design.md`

**구현 세부**:

- `dictionary/acronyms.json` — Rel-18 920개 파일에서 추출, 2503 entries; `keywords` 섹션 추가
- `src/spar/preprocessing/abbrev_mapper.py`
  - 파싱 직후 병기 확장 (`HO→HO(Handover)`)
  - conflict는 LLM closed-set 분류
  - 역방향 인덱스 `expand_query()`
  - `load_keywords()` + `extract_terms()`
- `scripts/run_ingest.py`
  - Phase 1 pre-pass(전체 파일 약어 수집) → Phase 2 ingest(map → chunk → keywords 주입)
  - `_find_chunk_keywords` — global + keywords 섹션 통합 탐색
  - Phase 2 루프 `[idx/N]` 파일 카운터 + embed 배치 verbose
  - `--llm-url`/`--llm-model` 플래그로 약어 충돌 해소에 LLM 사용 가능
- `scripts/extract_acronyms.py` — 3GPP md → acronyms.json 자동 추출기 (`_clean_expansion`, `_is_stylized_word`)
- `scripts/ingest_excel.py` — Excel column → acronyms.json `keywords` 섹션 병합 CLI
- `src/spar/ingest/excel_loader.py` — `load_excel_terms()` (openpyxl) + `merge_into_acronyms()`
- `src/spar/ingest/term_tagger.py` — `tag_chunk()` — 청크 텍스트에서 keyword 탐지 후 ARRAY 주입
- `src/spar/retrieval/milvus_client.py` — `keywords` ARRAY(VARCHAR, max_capacity=50, max_length=128) 필드 (max_length 32→128 확장)
- `src/spar/retrieval/routing.py` — `build_expr()` — `matched_terms` 기반 `array_contains` OR 필터
- `src/spar/pipeline/state.py` — `matched_terms: list[str]` 필드
- `src/spar/pipeline/nodes.py` — `_keywords: set[str]` + `preprocess()` → `matched_terms` 추출
- 테스트: excel_loader(9) + term_tagger(7) + abbrev_mapper(+8) + routing(+5) + nodes(+2) = 50개+

---

## Task 1.7.2 — 평가 자동화 (구현 중, 2026-05-01)

- `src/spar/eval/run_eval.py` — graph.ainvoke() 기반 eval (직접 Milvus 호출 제거)
- `src/spar/eval/eval_suite.py` — GraphConfig ablation, Recall@K/MRR/faithfulness 비교표
