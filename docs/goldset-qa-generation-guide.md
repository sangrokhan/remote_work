# 골드셋 QA 생성 가이드

Task 1.7 — 3GPP `.md` 문서에서 Codex CLI로 질문-답변 세트를 자동 생성한다.

---

## 개요

```
3GPP .md 파일
     ↓
gen_goldset_qa.py (프롬프트 구성 + codex exec 호출)
     ↓
data/goldsets/retrieval_goldset.jsonl
```

Codex에게 각 파일의 앞 N줄을 전달하고, 5가지 유형의 QA를 JSON으로 생성하도록 지시한다.
출력은 `.jsonl` 형식으로 누적된다. `chunk_id`는 이후 Milvus 검색 결과와 수동 매핑.

---

## 출력 스키마

```jsonl
{"query_id": "Q0001", "query": "SMF가 UE에 PDU 세션을 설정할 때 호출하는 서비스는?", "answer": "Nsmf_PDUSession_CreateSMContext", "type": "lookup", "section": "5.2.2.1", "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"}
```

| 필드 | 설명 |
|------|------|
| `query_id` | 고유 ID (Q0001~) |
| `query` | 질문 |
| `answer` | 문서 기반 답변 |
| `type` | `definition` / `procedural` / `diagnostic` / `comparative` / `lookup` |
| `section` | 관련 절 번호/제목 |
| `source_doc` | 원본 파일명 |
| `spec_number` | 파싱된 스펙 번호 (예: `29.502`) |
| `release` | 릴리즈 (예: `Rel-18`) |

> **chunk_id 매핑은 별도 단계**: 생성 후 `data/goldsets/retrieval_goldset.jsonl`을 검토하면서 Milvus에서 관련 청크를 검색하여 `relevant_chunk_ids` 추가.

---

## 사전 조건

```bash
# Codex 로그인 확인
codex whoami

# 미로그인 시
codex login
```

---

## 사용법

### 단일 파일

```bash
python scripts/gen_goldset_qa.py \
  --input-file data/tspec-llm/3GPP-clean/Rel-18/29_series/29502-i40.md
```

### 폴더 전체 (시리즈 단위)

```bash
python scripts/gen_goldset_qa.py \
  --input-dir data/tspec-llm/3GPP-clean/Rel-18/29_series
```

### 릴리즈 전체

```bash
python scripts/gen_goldset_qa.py \
  --input-dir data/tspec-llm/3GPP-clean/Rel-18 \
  --output data/goldsets/retrieval_goldset.jsonl \
  --append
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input-file FILE` | — | 단일 `.md` 파일 |
| `--input-dir DIR` | — | 폴더 내 `.md` 재귀 처리 |
| `--output FILE` | `data/goldsets/retrieval_goldset.jsonl` | 출력 경로 |
| `--max-lines N` | `1500` | 파일당 읽을 최대 줄 수 |
| `--append` | off | 기존 파일에 이어쓰기 |
| `--dry-run` | off | Codex 미호출, 구조 확인만 |

---

## 질의 유형 설명

| 유형 | 목적 | 예시 |
|------|------|------|
| `definition` | 용어/프로토콜/엔티티 정의 | "SMF란?" |
| `procedural` | 절차, 호출 흐름, 시퀀스 | "PDU 세션 수립 절차는?" |
| `diagnostic` | 장애 조건, 에러 처리, 예외 | "N1 인터페이스 실패 시 동작은?" |
| `comparative` | 두 개념/방식 비교 | "UPF와 SMF의 차이는?" |
| `lookup` | 특정 값, 파라미터, 타이머 조회 | "T3591 타이머 기본값은?" |

기본 생성 비율: definition 3 / procedural 2 / diagnostic 2 / comparative 1 / lookup 2  
(스크립트 내 `QA_COUNTS` 수정으로 조정 가능)

---

## 권장 워크플로우

### Step 1: 파일 슬라이스 (선택)

대용량 파일은 앞부분만 먼저 처리:

```bash
python scripts/slice_3gpp_intros.py   # /tmp/3gpp_intros/ 에 1000줄 슬라이스 생성

python scripts/gen_goldset_qa.py \
  --input-dir /tmp/3gpp_intros/29_series \
  --max-lines 1000
```

### Step 2: 소규모 테스트

```bash
# 단일 파일 dry-run
python scripts/gen_goldset_qa.py \
  --input-file data/tspec-llm/3GPP-clean/Rel-18/29_series/29502-i40.md \
  --dry-run

# 실제 생성 (1개 파일)
python scripts/gen_goldset_qa.py \
  --input-file data/tspec-llm/3GPP-clean/Rel-18/29_series/29502-i40.md \
  --output /tmp/qa_test.jsonl
```

### Step 3: 출력 검토

```bash
# 생성 확인
cat /tmp/qa_test.jsonl | python -m json.tool --indent 2 | head -60

# 유형별 분포 확인
cat /tmp/qa_test.jsonl | python -c "
import sys, json, collections
types = [json.loads(l)['type'] for l in sys.stdin]
for t, c in sorted(collections.Counter(types).items()):
    print(f'{t:15s} {c}')
"
```

### Step 4: 전체 생성

```bash
# 시리즈별로 나눠서 실행 (안정성)
for series in 29_series 38_series 36_series; do
  python scripts/gen_goldset_qa.py \
    --input-dir data/tspec-llm/3GPP-clean/Rel-18/$series \
    --append
done
```

---

## 이후 단계 (chunk_id 매핑)

생성된 `.jsonl`을 최종 골드셋으로 만들기 위한 수동/반자동 매핑:

```bash
# TODO: 각 query로 Milvus 검색 → 상위 청크 ID → relevant_chunk_ids 추가
# src/spar/eval/map_chunk_ids.py (Task 1.7.2에서 구현)
```

최종 스키마에 `relevant_chunk_ids` 필드 추가:
```jsonl
{"query_id": "Q0001", ..., "relevant_chunk_ids": ["c_abc123", "c_def456"]}
```
