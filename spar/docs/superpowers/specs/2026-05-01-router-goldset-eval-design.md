# 라우터 골드셋 구축 + 평가 파이프라인 설계

**날짜:** 2026-05-01  
**범위:** Task 2.3 — 라우터 골드셋 및 평가  
**관련 PRD:** `docs/prd.md` §Task 2.3

---

## 1. 목표

`HybridRouter`의 각 레이어(regex / embedding / llm / hybrid)를 정량 평가할 수 있는 골드셋과 평가 스크립트를 구축한다.  
기존 `retrieval_goldset.jsonl`의 `type` 필드를 라우트 레이블로 재활용해 LLM 호출 없이 즉시 ~300개 레이블을 획득한다.

---

## 2. 산출물

| 파일 | 설명 |
|---|---|
| `scripts/gen_router_goldset.py` | QA goldset → router goldset 변환 스크립트 |
| `scripts/run_router_eval.py` | 라우터 평가 스크립트 |
| `data/goldsets/router_goldset.jsonl` | 라우터 평가용 골드셋 |
| `data/eval_results/router_eval_{date}.md` | 평가 리포트 (Markdown) |

---

## 3. 골드셋 스키마

`data/goldsets/router_goldset.jsonl` — JSONL, 한 줄 = 한 항목:

```json
{
  "query_id": "RQ0001",
  "query": "What is Carrier Aggregation?",
  "expected_route": "definition_explain",
  "source_doc": "29502-i40.md",
  "spec_number": "29.502",
  "release": "Rel-18",
  "qa_query_id": "Q0001"
}
```

### 타입 매핑 (QA `type` → `expected_route`)

| QA `type` | `expected_route` |
|---|---|
| `definition` | `definition_explain` |
| `procedural` | `procedural` |
| `diagnostic` | `diagnostic` |
| `comparative` | `comparative` |
| `lookup` | `structured_lookup` |

- `default_rag`: QA goldset에 해당 타입 없음. 변환 시 생성하지 않으며 평가에서 제외.
- 미매핑 타입은 경고 출력 후 스킵.
- `qa_query_id`: 원본 QA 항목 역추적용.

---

## 4. `scripts/gen_router_goldset.py`

### 역할

기존 QA goldset을 읽어 라우터 골드셋 형식으로 변환 저장.

### CLI

```
python scripts/gen_router_goldset.py \
  [--input data/goldsets/retrieval_goldset.jsonl] \
  [--output data/goldsets/router_goldset.jsonl] \
  [--append] \
  [--dry-run]
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input` | `data/goldsets/retrieval_goldset.jsonl` | QA goldset 경로 |
| `--output` | `data/goldsets/router_goldset.jsonl` | 출력 경로 |
| `--append` | False | 이어쓰기 (기본: 덮어쓰기) |
| `--dry-run` | False | 파일 미생성, 통계만 출력 |

### 동작 흐름

1. QA goldset JSONL 전체 로드
2. 각 항목: `type` → `expected_route` 매핑
3. `query_id` → `RQ{N:04d}` 재번호, `qa_query_id`에 원본 보존
4. 미매핑 타입은 stderr 경고 후 스킵
5. `--append` 시 기존 항목 수부터 번호 이어받기
6. 결과 JSONL 저장, 통계 출력

### 출력 예시

```
QA goldset 312개 로드
매핑: definition=98, procedural=62, diagnostic=71, comparative=39, lookup=42
미매핑(스킵): 0개
router_goldset.jsonl → 312개 저장
```

---

## 5. `scripts/run_router_eval.py`

### 역할

라우터 골드셋으로 지정 레이어를 평가하고 Markdown 리포트 저장.

### CLI

```
python scripts/run_router_eval.py \
  [--goldset data/goldsets/router_goldset.jsonl] \
  [--layer hybrid] \
  [--threshold 0.65] \
  [--output data/eval_results/router_eval_{date}.md]
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--goldset` | `data/goldsets/router_goldset.jsonl` | 평가 데이터 경로 |
| `--layer` | `hybrid` | `regex` \| `embedding` \| `llm` \| `hybrid` |
| `--threshold` | `0.65` | embedding layer 임계값 override |
| `--output` | `data/eval_results/router_eval_{date}.md` | 리포트 저장 경로 |

### 동작 흐름

1. goldset 로드
2. `--layer` 에 따라 라우터 초기화:
   - `regex`: `RegexRouter` 직접 호출
   - `embedding`: `EmbeddingRouter` 직접 호출 (encoder는 `get_encoder()` 싱글톤)
   - `llm`: `LLMRouter` 직접 호출
   - `hybrid`: `HybridRouter` 전체 실행
3. 각 query → `predicted_route` 획득
   - `None` 반환 시 (embedding layer threshold miss) → `default_rag`로 기록
4. `expected_route` vs `predicted_route` 비교
5. 메트릭 계산 + 리포트 저장

### 메트릭

- **Overall accuracy**: correct / total
- **Per-route**: precision / recall / F1 (sklearn `classification_report` 패턴)
- **Confusion matrix**: 6×6 표 (Markdown 테이블)
- **Layer coverage** (embedding/hybrid 한정): threshold 이상 판정 비율 vs fallback 비율

### 리포트 예시

```markdown
# Router Eval — embedding layer (2026-05-01)

## Overall
accuracy: 74.3% (232/312)

## Per-route
| route              | precision | recall | F1   | support |
|--------------------|-----------|--------|------|---------|
| structured_lookup  | 0.91      | 0.88   | 0.89 | 42      |
| definition_explain | 0.78      | 0.82   | 0.80 | 98      |
| procedural         | 0.71      | 0.69   | 0.70 | 62      |
| diagnostic         | 0.68      | 0.72   | 0.70 | 71      |
| comparative        | 0.65      | 0.59   | 0.62 | 39      |

## Coverage
matched (≥0.65): 287/312 (91.9%)
fallback: 25/312 (8.1%)

## Confusion Matrix
...
```

---

## 6. 의존성 및 제약

- `sklearn` (`scikit-learn`) — precision/recall/F1 계산. 이미 dev 의존성에 포함 여부 확인 필요.
- `embedding` / `hybrid` 레이어 평가 시 encoder 서빙 필요 (`ENCODER_URL` env var).
- `llm` / `hybrid` 레이어 평가 시 LLM 서빙 필요.
- `regex` 레이어 평가는 외부 의존성 없음.

---

## 7. 미포함 범위

- `default_rag` 라우트 goldset 항목 생성 (추후 수동 추가 또는 별도 LLM 생성 스크립트)
- CI 자동화 (평가 스크립트는 수동 실행)
- 임계값 자동 튜닝
