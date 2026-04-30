# Abbreviation Mapping — Design Spec

**Date:** 2026-04-30
**Status:** Approved

---

## 1. 목표

통신 도메인 약어를 파싱 직후 확장하여 청크/임베딩/BM25 품질을 높이고,
쿼리 처리 시 역방향 매핑으로 약어·전체어 양방향 검색을 보장한다.

---

## 2. 파이프라인 위치

```
파싱 → [abbrev_map] → 청킹 → 임베딩 → Milvus 삽입
쿼리 → [abbrev_map] → 검색
```

동일한 `acronyms.json` 사전을 인제스트/쿼리 양쪽에서 재사용한다.

---

## 3. 사전 구조 (`dictionary/acronyms.json`)

```json
{
  "global": {
    "HO": {
      "expansion": "Handover",
      "variants": ["Hand-Over"]
    },
    "TTT": {
      "expansion": "Time-To-Trigger",
      "variants": []
    },
    "RACH": {
      "expansion": "Random Access Channel",
      "variants": []
    },
    "BWP": {
      "expansion": "Bandwidth Part",
      "variants": []
    }
  },
  "conflicts": {
    "CA": {
      "candidates": ["Carrier Aggregation", "Cell Activation"],
      "variants": []
    },
    "CR": {
      "candidates": ["Change Request", "Cell Range"],
      "variants": []
    }
  }
}
```

**규칙:**
- `variants`: 하이픈 포함 변형은 실제로 확인된 경우에만 수동 추가. 자동 생성 금지.
- `conflicts.candidates`: 해당 약어가 가질 수 있는 모든 후보 전체어 목록.

---

## 4. 인제스트 시 처리 (`abbrev_mapper.py`)

### 4-1. global 약어

단어 경계(`\b`) 기반 regex 치환 — 병기 방식:

```
HO  →  HO(Handover)
Hand-Over  →  Hand-Over(Handover)   ← variants도 동일 처리
```

### 4-2. conflict 약어

1. 텍스트 내 conflict 약어 전체 출현 위치 수집
2. **문서당 단일 LLM 호출**로 배치 판단 (closed-set 분류)
3. 판단 결과에 따라 3단계 처리:

| LLM 확신 | 처리 |
|----------|------|
| 높음 | 단일 expansion 병기: `CA(Carrier Aggregation)` |
| 낮음 | 후보 전체 병기: `CA(Carrier Aggregation\|Cell Activation)` + chunk metadata에 `candidates` 저장 |
| 판단 불가 | 원문 유지 + metadata에 `ambiguous: true` 마킹 |

LLM 프롬프트 형식 (closed-set):
```
다음 문맥에서 "CA"의 의미를 아래 후보 중 하나로만 답해줘.
후보: ["Carrier Aggregation", "Cell Activation"]
확신이 없으면 "uncertain"으로 답해줘.

문맥: "...{±200자 컨텍스트}..."
```

---

## 5. 쿼리 시 처리

### 5-1. 정방향 (약어 → 전체어)

쿼리에 약어 발견 시 expansion 추가:
```
"HO threshold" → "HO(Handover) threshold"
```

conflict 약어 + 컨텍스트 불충분 시 → 사용자 재질의:
```
"CA를 어떤 의미로 쓰셨나요? (Carrier Aggregation / Cell Activation)"
```

### 5-2. 역방향 (전체어·variants → 약어)

런타임에 역방향 인덱스 빌드:
```python
reverse = {
  "Handover": "HO",
  "Hand-Over": "HO",   # variant
  "handover": "HO",    # lowercase
  ...
}
```

쿼리 토큰 역방향 조회 → 매칭 시 OR 확장:
```
"Hand-Over threshold"
  → reverse["Hand-Over"] = "HO"
  → 검색: "Hand-Over(Handover) threshold OR HO(Handover) threshold"
```

하이픈 정규화도 병행 체크:
```python
token.replace("-", "")  # "Hand-Over" → "HandOver" → reverse 조회
```

---

## 6. 모듈 구성

| 파일 | 역할 |
|------|------|
| `dictionary/acronyms.json` | 약어 사전 |
| `src/spar/preprocessing/abbrev_mapper.py` | 인제스트·쿼리 공용 매핑 로직 |
| `scripts/run_ingest.py` | `map_abbreviations()` 호출 추가 |

### 인터페이스

```python
def load_acronyms(path: Path) -> dict: ...

def map_abbreviations(text: str, acronyms: dict) -> str:
    """global 치환 + conflict LLM 분류 적용. 인제스트·쿼리 양쪽에서 호출."""
    ...

def build_reverse_index(acronyms: dict) -> dict[str, str]:
    """전체어·variants → 약어 역방향 인덱스 빌드."""
    ...
```

---

## 7. 처리 안 하는 것

- variants 자동 생성 (e.g., 모든 약어에 자동으로 하이픈 변형 추가) — 수동 확인 후 추가만 허용
- doc_type별 사전 분리 — conflicts 후보 목록으로 충분
- LLM을 global 약어에 사용 — 결정론적 regex로 처리
