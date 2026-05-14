## PRD: YANG 기반 NETCONF 명령어 생성 시스템 (PoC)

### 1. 개요

#### 1.1 배경
YANG 모델로 정의된 네트워크 장비 설정을 변경할 때, 사용자는 NETCONF `<edit-config>` XML을 직접 작성해야 한다. 이는 모듈 네임스페이스, list 키 계층, 타입 제약 등 도메인 지식이 요구되어 진입 장벽이 높다.

#### 1.2 목적
자연어 요청을 입력받아 NETCONF `<edit-config>` 명령어를 생성하는 LLM 기반 시스템의 PoC를 구축한다. LLM은 YANG 스키마 전체를 컨텍스트로 들고 있는 대신, **스키마 조회 도구를 호출하며 명령어를 조립**한다.

#### 1.3 범위

**포함**
- YANG 스키마 인덱싱 및 조회 도구 구현
- LLM 에이전트의 도구 사용 흐름 정의
- `<edit-config>` XML 생성 및 자체 검증
- `<get-config>` XML 생성 (설정 조회)
- `<delete-config>` XML 생성 (데이터스토어 전체 삭제)
- 멀티턴 키 수집(positive case는 단일 턴 종료)
- 샘플 NETCONF 서버 기반 end-to-end 검증

**제외**
- 운영 환경 적용, 권한/감사/롤백
- `<get>` (운영 상태 포함 조회) RPC 생성
- 다중 leaf 동시 변경, 트랜잭션 묶음
- 실시간 스키마 변경 반영

### 2. 성공 기준

- 골든 케이스 10\~20건에 대해 생성된 XML이 libyang 검증을 통과한다.
- 샘플 NETCONF 서버(netopeer2 등)에서 `<edit-config>` 적용이 성공한다.
- `<get-config>` 요청에 대해 올바른 filter XML이 생성되고 서버 응답을 반환한다.
- `<delete-config>` 요청에 대해 대상 데이터스토어 삭제 XML이 생성된다.
- 사람이 도구 호출 trace를 보고 실패 원인을 명확히 분류할 수 있다.
- positive case에서 단일 턴 안에 명령어 생성이 완료된다.

### 3. 사용자 시나리오

#### 3.1 Positive Case (PoC 1차 목표)
사용자가 leaf 식별 정보, 모든 list 키 값, 설정 값을 한 번에 제공한다.

> "ietf-interfaces의 eth0 인터페이스 MTU를 1500으로 설정해줘"

시스템은 도구를 호출해 경로/키/타입을 확인하고, `<edit-config>` XML을 반환한다.

#### 3.2 Get-config Case
사용자가 특정 경로의 설정값 조회를 요청한다.

> "ietf-interfaces의 eth0 인터페이스 현재 MTU 값을 조회해줘"

시스템은 대상 노드 경로와 키를 확인하고, subtree filter 기반 `<get-config>` XML을 반환한다.

#### 3.3 Delete-config Case
사용자가 특정 데이터스토어 삭제를 요청한다.

> "startup 데이터스토어를 삭제해줘"

시스템은 `<delete-config>` XML을 생성하고 대상 데이터스토어를 명시한다. 삭제 전 사람 확인 단계를 거친다.

#### 3.4 Multi-turn Case (구조만 지원)
키 누락 시 시스템이 부족한 키를 사용자에게 되묻는다. PoC에서는 분기 동작만 확인한다.

### 4. 시스템 아키텍처

```
[자연어 요청]
     ↓
[LLM Agent] ⇄ [Schema Tool Server] ⇄ [Indexed Schema Store]
     ↓
[<edit-config> XML]
     ↓
[사람 검토] → [샘플 NETCONF 서버 적용] → [결과 수집]
```

### 5. 컴포넌트 요구사항

#### 5.1 Schema Indexer (전처리, 1회 실행)

**입력**: YANG 모듈 파일 집합
**출력**: 정규화된 노드 저장소(SQLite 또는 JSON + in-memory 인덱스)

**요구사항**
- libyang 기반 effective schema tree 생성 (import/include/augment/deviation/uses 전개)
- typedef 재귀 해석 후 base type 저장
- 노드 단위 레코드는 최소 다음 필드를 포함

| 필드 | 설명 |
|---|---|
| node_id | 안정적 ID (모듈명 + schema path 해시) |
| schema_path | `/ietf-interfaces:interfaces/interface/mtu` |
| module / namespace / prefix | 직렬화용 |
| node_kind | container / list / leaf / leaf-list / choice / case |
| config | true/false |
| parent_id / children_ids | 트리 탐색용 |
| keys[] | list인 경우, 순서 보존 |
| type | base type + range/length/pattern/enum/leafref 등 제약 |
| default / mandatory | |
| description | 자연어 매칭용 |
| when / must | 원본 표현식 |

- 인덱스: leaf 이름 → node_id, description 토큰 → node_id, schema path prefix 트리

#### 5.2 Schema Tool Server

LLM이 호출하는 도구를 함수 또는 HTTP/MCP 엔드포인트로 노출. 모든 도구는 결정론적·멱등.

**(a) 탐색 도구**
- `list_modules()`
- `search_nodes(keyword, kind=None, top_k=10)`
- `find_leaf(name, parent_hint=None)`

**(b) 트리 탐색 도구**
- `get_node(node_id_or_path)`
- `get_children(node_id)`
- `get_ancestors(node_id)` — 각 단계의 kind, keys, module 포함

**(c) Key / 계층 추출 도구 (핵심)**
- `get_path_to_leaf(node_id)` — 루트→target 경로 전체
- `get_required_keys(node_id)` — 경로상 모든 list의 키를 평탄화해 반환
- `resolve_instance_path(node_id, key_values)` — 완성된 instance path와 누락 키 리스트 반환 (멀티턴 트리거)

**(d) 타입 / 값 검증 도구**
- `get_type_info(node_id)`
- `validate_value(node_id, value)`
- `resolve_identityref(type, value)`

**(e) 명령어 조립 도구**
- `build_edit_config(target_node_id, key_values, value, operation="merge", datastore="running")`
  - 네임스페이스 prefix 선언, 부모 container 자동 생성, list 키 leaf 삽입, operation 속성 부여 일괄 처리
  - `operation="delete"` 시 값 없이 삭제 XML 생성
- `build_get_config(target_node_id, key_values=None, datastore="running")`
  - subtree filter XML 생성; key_values 있으면 instance filter, 없으면 전체 subtree 반환
- `build_delete_config(datastore)` — 데이터스토어 전체 삭제 XML 생성 (startup/candidate만 허용, running 거부)
- `validate_edit_config(xml)` — libyang round-trip 검증

#### 5.3 LLM Agent

**모델 요구사항**
- 도구 호출(function calling) 지원
- YANG 문법 지식 불필요. 시스템 프롬프트로 다음 최소 개념만 주입
  - container / list / leaf / leaf-list / choice·case의 의미
  - list key의 역할과 NETCONF XML 내 필수성
  - config true/false 구분
  - 모듈마다 namespace/prefix가 존재한다는 사실
  - NETCONF `<edit-config>`, operation, datastore 어휘

**가드레일**
- YANG 파일 직접 읽기 또는 문법 추론 금지
- 모든 사실 확인은 도구 호출로 수행
- 자체 지식으로 답하지 말 것

**표준 작업 흐름 — edit-config**
1. 자연어에서 의도/값 추출
2. `search_nodes` 또는 `find_leaf`로 후보 노드 탐색, 부모 hint로 디스앰비기에이션
3. `get_path_to_leaf`로 경로 확인
4. `get_required_keys`로 필요 키 확인 → 사용자 입력과 매칭, 누락 시 멀티턴 질문
5. `resolve_instance_path`로 instance path 확정
6. `validate_value`로 값 검증
7. `build_edit_config`로 XML 생성
8. `validate_edit_config`로 round-trip 검증 후 반환

**표준 작업 흐름 — get-config**
1. 자연어에서 조회 대상 노드/경로 추출
2. `search_nodes` / `find_leaf`로 노드 특정
3. `get_required_keys`로 필요 키 확인 → 사용자 입력 매칭
4. `build_get_config`로 subtree filter XML 생성 후 반환

**표준 작업 흐름 — delete-config**
1. 자연어에서 삭제 대상 데이터스토어(startup/candidate) 확인
2. running 요청 시 거부 메시지 반환
3. `build_delete_config`로 XML 생성 후 반환 (실제 적용은 사람 확인 후)

#### 5.4 샘플 NETCONF 서버
- netopeer2 또는 ConfD free 버전
- Indexer와 동일한 YANG 모듈 로드
- 생성된 XML을 `<edit-config>`로 전송 → 응답(success/`rpc-error`) 수집 자동화

#### 5.5 검토 인터페이스
한 화면에 다음을 표시한다.
- 원본 자연어 요청
- LLM 도구 호출 trace (호출 순서, 인자, 응답)
- 최종 `<edit-config>` XML
- 샘플 서버 응답

도구 호출 trace 보존은 실패 원인(노드 매칭 실수 vs 키 누락 미감지 vs 타입 검증 실패)을 가르는 데 필수적이다.

### 6. 데이터 모델: 도구 응답 예시

```json
// get_required_keys 응답 예
{
  "target_path": "/ietf-interfaces:interfaces/interface/mtu",
  "required_keys": [
    {
      "list_path": "/ietf-interfaces:interfaces/interface",
      "key_name": "name",
      "type": "string",
      "constraints": {}
    }
  ]
}
```

```json
// resolve_instance_path 응답 예
{
  "instance_path": "/ietf-interfaces:interfaces/interface[name='eth0']/mtu",
  "missing_keys": []
}
```

### 7. 작업 계획

| 단계 | 산출물 |
|---|---|
| 1. Schema Indexer 구현 | libyang 기반 인덱서, 정규화된 노드 DB |
| 2. Schema Tool Server 구현 | 5.2의 도구 함수/엔드포인트 (`build_get_config`, `build_delete_config` 포함) |
| 3. LLM Agent 프롬프트·도구 통합 | 시스템 프롬프트, 도구 스키마, few-shot trace (edit/get/delete 3종) |
| 4. 샘플 NETCONF 서버 셋업 | netopeer2 + 동일 모듈 + 적용 스크립트 |
| 5. 골든 케이스 수집 | 10\~20건 (단순 leaf / 중첩 list / 복합 key / augment 노드 / union / identityref / get-config / delete-config) |
| 6. End-to-end 검증 | trace + XML + 서버 응답 기록, 실패 분류, 도구·프롬프트 보강 |

### 8. 리스크 및 대응

| 리스크 | 대응 |
|---|---|
| LLM이 도구를 건너뛰고 자체 지식으로 추론 | 프롬프트에 도구 우선 원칙 명시, few-shot trace 제공 |
| 동명 leaf로 인한 노드 매칭 실패 | `find_leaf`에 parent_hint, `search_nodes` 후보 다중 반환 |
| 복합 키/중첩 list 키 누락 | `get_required_keys` + `resolve_instance_path`로 결정론적 차단 |
| augment된 노드의 네임스페이스 혼동 | Indexer 단계에서 augment 펼치고 module/namespace를 노드에 보존 |
| 생성 XML이 서버 스키마와 미세하게 불일치 | `validate_edit_config` round-trip + 샘플 서버 적용으로 이중 검증 |

### 9. 향후 확장 (PoC 이후)
- `<get>` RPC (운영 상태 포함 조회)
- 다중 leaf 일괄 변경, 트랜잭션
- candidate datastore + commit 흐름
- 자연어 매칭 품질 개선(임베딩 기반 검색, 별칭 사전)

