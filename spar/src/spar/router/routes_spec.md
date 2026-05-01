# SPAR 라우트 명세 (routes_spec.md)

> Task 2.1 산출물. 라우트 정의 + 판단 기준 + 검색 전략 요약.

---

## 라우트 목록

| Route | 설명 | 주요 인덱스 | 비고 |
|---|---|---|---|
| `structured_lookup` | 파라미터/카운터/알람 정확 조회 | parameter_ref, counter_ref, alarm_ref | BM25 가중치 높게 |
| `definition_explain` | 서술형 정의/개념 설명 | spec, feature_desc | Dense 우선 |
| `procedural` | 절차·설치·설정 방법 | mop, install_guide | 순서 보존 청크 |
| `diagnostic` | 트러블슈팅·장애 분석 | 전체 (alarm_ref 포함) | multi-hop 가능 |
| `comparative` | 버전·기능 비교 | release_notes, feature_desc | 메타 필터 (release) |
| `default_rag` | 위 분류 불가 시 fallback | 전체 hybrid | RRF 기본값 |

---

## 라우트별 정의 및 판단 기준

### `structured_lookup`

**정의**: 특정 파라미터·카운터·알람의 값, 범위, 설명 등 구조화된 사실을 조회하는 질의.

**판단 기준**:
- 알람 코드 포함 (`ALM-\d+`, `alarm \d+`)
- 파라미터명 패턴 포함 (camelCase + 기술 접미사: `maxTxPower`, `tReordering`, `hysteresisA3`)
- MO 이름 포함 (`NRCellDU`, `EUtranCellFDD` 등)
- "What is the default value of", "What is the range of", "Which parameter controls" 형태

**판단 제외**:
- "What does X mean" → `definition_explain`
- "How do I configure X" → `procedural`

---

### `definition_explain`

**정의**: 용어·개념·표준 문서의 의미를 서술형으로 설명하는 질의.

**판단 기준**:
- 3GPP TS 문서번호 포함 (`TS 29.502`, `3GPP TS 38.300`, `TS29502`)
- "What is X", "Explain X", "Define X" 형태
- 약어 풀이 요청 ("What does AMF stand for")
- 기능·아키텍처 개념 질문

**판단 제외**:
- 파라미터 값·범위 조회 → `structured_lookup`
- 설정 방법 → `procedural`

---

### `procedural`

**정의**: 설정·설치·운용 절차를 단계별로 묻는 질의.

**판단 기준**:
- "How to", "How do I", "Steps to", "Procedure for" 형태
- 설치(install), 설정(configure/setup), 활성화(enable/activate) 동사 포함
- MOP 문서 관련 질의

**판단 제외**:
- 절차 결과에 대한 개념 설명 → `definition_explain`

---

### `diagnostic`

**정의**: 장애·이상 증상의 원인 분석 및 해결책을 묻는 질의.

**판단 기준**:
- "Why is X failing", "Root cause of", "Troubleshoot" 형태
- 알람 발생 원인·해결 질문 (알람 코드 없이 증상으로 묻는 경우)
- "X is not working", "X keeps dropping" 형태

**판단 제외**:
- 특정 알람 코드 정확 조회 → `structured_lookup`

---

### `comparative`

**정의**: 버전 간 변경사항, 기능 간 차이점을 비교하는 질의.

**판단 기준**:
- "Difference between X and Y", "Compared to", "Changed in", "New in vX.Y" 형태
- Release Notes 관련 질의
- 두 가지 이상 항목·버전의 대조

**판단 제외**:
- 단일 항목 정의 → `definition_explain`

---

### `default_rag`

**정의**: 위 5개 라우트에 해당하지 않는 일반 질의. 전체 인덱스 hybrid 검색.

**판단 기준**:
- Layer 1 regex 미매칭
- Layer 2 embedding similarity < threshold (0.65)
- Layer 3 LLM이 명시적 라우트 판단 불가 시

---

## 라우터 3-layer 판단 흐름

```
질의 입력
  │
  ├─ Layer 1 (Regex, confidence ≥ 0.9)
  │    └─ 알람/파라미터/MO/TS번호 패턴 매칭 → 즉시 반환
  │
  ├─ Layer 2 (Embedding, threshold 0.65)
  │    └─ 라우트별 예시 centroid와 cosine similarity
  │
  └─ Layer 3 (LLM, optional)
       └─ 소형 LLM (Qwen 2.5 7B / Llama 3.1 8B) structured output
            → route + entities (param_name, alarm_code, mo_name, product, release)
```

---

## 구현 참조

| 컴포넌트 | 파일 |
|---|---|
| Route enum, RouteResult | `src/spar/router/schemas.py` |
| Layer 1 Regex | `src/spar/router/regex_router.py` |
| Layer 2 Embedding | `src/spar/router/embedding_router.py` |
| Layer 3 LLM | `src/spar/router/llm_router.py` |
| 3-layer 통합 | `src/spar/router/hybrid_router.py` |
| 테스트 | `tests/router/`, `tests/test_routing_3gpp.py` |
