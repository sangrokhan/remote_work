# 프로젝트 문서 인덱스

Deep Learning Platform의 전체 문서 구조와 참조 가이드입니다.

## 📚 문서 구조

### 핵심 문서
| 문서 | 목적 | 대상 독자 |
|------|------|----------|
| [README.md](../README.md) | 프로젝트 개요 및 빠른 시작 | 모든 사용자 |
| [CLAUDE.md](../CLAUDE.md) | AI 어시스턴트 가이드라인 | Claude Code |

### 개발 문서
| 문서 | 목적 | 대상 독자 |
|------|------|----------|
| [DEVELOPMENT.md](./DEVELOPMENT.md) | 개발 환경 설정 및 가이드 | 개발자 |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 시스템 아키텍처 및 설계 | 개발자, 아키텍트 |
| [API.md](./API.md) | REST API 명세 및 사용법 | 개발자, API 사용자 |

### GPU 클러스터 연동 문서
| 문서 | 목적 | 대상 독자 |
|------|------|----------|
| [GPU_CLUSTER_INTEGRATION.md](./GPU_CLUSTER_INTEGRATION.md) | GPU 클러스터 연동 시스템 설계 | 개발자, 시스템 관리자 |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | 단계별 구현 계획 | 개발팀, PM |
| [DESIGN_DECISIONS.md](./DESIGN_DECISIONS.md) | 설계 결정사항 및 근거 | 개발자, 아키텍트 |

## 🎯 독자별 가이드

### 신규 개발자
1. [README.md](../README.md) - 프로젝트 이해
2. [DEVELOPMENT.md](./DEVELOPMENT.md) - 환경 설정
3. [ARCHITECTURE.md](./ARCHITECTURE.md) - 시스템 구조 파악
4. [GPU_CLUSTER_INTEGRATION.md](./GPU_CLUSTER_INTEGRATION.md) - 핵심 기능 이해

### API 사용자
1. [README.md](../README.md) - 기본 정보
2. [API.md](./API.md) - API 명세
3. [GPU_CLUSTER_INTEGRATION.md](./GPU_CLUSTER_INTEGRATION.md) - GPU 작업 API

### 시스템 관리자
1. [ARCHITECTURE.md](./ARCHITECTURE.md) - 시스템 구조
2. [DESIGN_DECISIONS.md](./DESIGN_DECISIONS.md) - 기술적 결정사항
3. [DEVELOPMENT.md](./DEVELOPMENT.md) - 배포 가이드

### 프로젝트 매니저
1. [README.md](../README.md) - 프로젝트 로드맵
2. [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) - 구현 계획
3. [GPU_CLUSTER_INTEGRATION.md](./GPU_CLUSTER_INTEGRATION.md) - 기능 명세

## 📋 주요 기능별 문서 매핑

### 인증 및 사용자 관리
- **API 명세**: [API.md - Authentication Endpoints](./API.md#authentication-endpoints)
- **구현 가이드**: [DEVELOPMENT.md - Authentication & Security](./DEVELOPMENT.md)
- **아키텍처**: [ARCHITECTURE.md - Security Architecture](./ARCHITECTURE.md#security-architecture)

### 데이터셋 관리
- **API 명세**: [API.md - Dataset Management](./API.md#dataset-management)
- **아키텍처**: [ARCHITECTURE.md - Data Layer](./ARCHITECTURE.md#data-layer)

### 모델 훈련 (GPU 클러스터)
- **시스템 설계**: [GPU_CLUSTER_INTEGRATION.md](./GPU_CLUSTER_INTEGRATION.md)
- **구현 계획**: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
- **기술적 결정**: [DESIGN_DECISIONS.md](./DESIGN_DECISIONS.md)
- **API 명세**: [API.md - Model Training](./API.md#model-training)

### 모델 관리 및 추론
- **API 명세**: [API.md - Model Management](./API.md#model-management)
- **아키텍처**: [ARCHITECTURE.md - ML Layer](./ARCHITECTURE.md#machine-learning-layer)

### 배포 및 운영
- **배포 가이드**: [README.md - Deployment](../README.md#deployment)
- **개발 환경**: [DEVELOPMENT.md - Local Development](./DEVELOPMENT.md#local-development-setup)
- **Docker 설정**: [DEVELOPMENT.md - Docker Operations](./DEVELOPMENT.md)

## 🔍 빠른 참조

### 개발 명령어
```bash
# 환경 설정
source venv/bin/activate
pip install -r requirements.txt

# 서버 실행
uvicorn src.api.main:app --reload --port 8000
redis-server
celery -A src.worker worker --loglevel=info

# 테스트
pytest tests/
pytest --cov=src tests/

# 빌드
docker build -t dl-platform .
docker run -p 8000:8000 dl-platform
```

### API 엔드포인트
```bash
# 기본 API
GET    /                              # 헬스 체크
GET    /docs                          # API 문서

# GPU 작업 관리
POST   /api/gpu-jobs/submit           # 작업 제출
GET    /api/gpu-jobs/{job_id}/status  # 상태 확인
GET    /api/gpu-jobs/{job_id}/logs    # 로그 조회
POST   /api/gpu-jobs/{job_id}/collect-results  # 결과 수집
```

### 클러스터 명령어 (예시)
```bash
# 프로세스 실행
gpu-run --gpus 2 --memory 32G /path/to/script.sh

# 상태 확인
ps -p {process_id} -o stat --no-headers

# 파일 전송
scp local_file.py cluster:/remote/path/
```

## 📝 문서 업데이트 가이드

### 문서 생성 규칙
1. **Markdown 사용**: 모든 문서는 Markdown 형식
2. **구조화**: 명확한 섹션 구조 및 목차
3. **코드 예시**: 실제 사용 가능한 코드 포함
4. **다이어그램**: Mermaid를 활용한 플로우 차트

### 업데이트 주기
- **README.md**: 주요 기능 추가 시
- **API.md**: API 변경 시마다
- **ARCHITECTURE.md**: 아키텍처 변경 시
- **DEVELOPMENT.md**: 개발 프로세스 변경 시

### 버전 관리
- 문서 변경사항은 Git으로 추적
- 주요 변경 시 CHANGELOG 업데이트
- API 문서는 코드와 동기화 유지

## 🚀 다음 단계

### 즉시 구현 (Week 1-2)
1. **데이터 모델 구현**: `src/ml/cluster/models.py`
2. **클러스터 연결**: `src/ml/cluster/connection.py`
3. **기본 API**: `src/api/endpoints/gpu_jobs.py`

### 우선순위 기능 (Week 3-4)
1. **작업 제출 및 실행**
2. **상태 모니터링**
3. **기본 로그 수집**

### 고급 기능 (Month 2)
1. **실시간 로그 스트리밍**
2. **자동 결과 수집**
3. **에러 처리 및 복구**

### 프로덕션 준비 (Month 3)
1. **성능 최적화**
2. **보안 강화**
3. **모니터링 및 알림**
4. **문서 완성**

---

**문서 작성 완료**: 2024년 8월 25일  
**마지막 업데이트**: 설계 단계 완료, 구현 준비  
**다음 단계**: 핵심 컴포넌트 구현 시작