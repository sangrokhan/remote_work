# 설계 결정사항 및 근거

## 아키텍처 결정사항

### 1. 통신 프로토콜: SSH/SCP 선택

#### 결정
SSH를 통한 명령 실행과 SCP를 통한 파일 전송을 사용

#### 근거
- **기존 인프라 활용**: 대부분의 클러스터 환경에서 SSH는 기본 제공
- **보안성**: 암호화된 통신 및 키 기반 인증
- **단순성**: 별도의 API 서버나 에이전트 설치 불필요
- **표준화**: 업계 표준 프로토콜로 호환성 보장

#### 고려된 대안
- **REST API**: 클러스터에 별도 API 서버 필요
- **gRPC**: 성능상 우수하나 인프라 복잡도 증가
- **Message Queue**: 비동기 처리 우수하나 추가 인프라 필요

### 2. 파일 전송: SFTP over SCP

#### 결정
paramiko의 SFTP 클라이언트를 사용하여 SCP 기능 구현

#### 근거
- **Python 네이티브**: paramiko는 순수 Python SSH 구현체
- **비동기 지원**: asyncio와 호환 가능
- **에러 처리**: 상세한 예외 정보 제공
- **진행률 추적**: 파일 전송 진행률 모니터링 가능

### 3. 작업 상태 관리: Pull 모델

#### 결정
API 서버에서 주기적으로 클러스터에 상태를 조회하는 Pull 모델 채택

#### 근거
- **단순성**: 클러스터 측 변경 없이 구현 가능
- **신뢰성**: 네트워크 단절 시에도 상태 복구 가능
- **확장성**: 다수의 클러스터와 연동 시 관리 용이

#### 고려된 대안
- **Push 모델**: 클러스터에서 콜백 전송 (방화벽/네트워크 제약)
- **Event-driven**: 실시간성 우수하나 복잡도 증가

### 4. 작업 디렉토리 구조

#### 결정
작업별 독립 디렉토리 구조 사용
```
/cluster/ml-jobs/{job_id}/
├── code/      # 실행 코드
├── data/      # 입력 데이터
├── models/    # 입력 모델
├── outputs/   # 출력 결과
├── logs/      # 실행 로그
└── run.sh     # 실행 스크립트
```

#### 근거
- **격리성**: 작업 간 파일 충돌 방지
- **정리 용이성**: 작업 완료 후 쉬운 정리
- **추적성**: 작업별 명확한 파일 구조
- **디버깅**: 문제 발생 시 독립적 분석 가능

### 5. 코드 소스 처리: 다중 소스 지원

#### 결정
Git, 로컬 파일, 기존 클러스터 코드 3가지 소스 지원

#### 근거
- **유연성**: 다양한 개발 워크플로우 지원
- **편의성**: 사용자 선호에 따른 코드 배포 방식 선택
- **효율성**: 이미 클러스터에 있는 코드 재사용 가능

```python
# 코드 소스별 처리 로직
def prepare_code(source_type: str, source_path: str, target_path: str):
    if source_type == "git":
        return f"git clone {source_path} {target_path}"
    elif source_type == "local":
        # 로컬 -> tar -> 전송 -> 압축해제
        return upload_and_extract(source_path, target_path)
    elif source_type == "existing":
        return f"cp -r {source_path}/* {target_path}/"
```

## 기술 스택 결정사항

### 1. SSH 클라이언트: paramiko vs asyncssh

#### 선택: paramiko
- **안정성**: 오랜 기간 검증된 라이브러리
- **문서화**: 풍부한 문서 및 예제
- **커뮤니티**: 활발한 커뮤니티 지원

#### asyncssh (보류)
- **성능**: 비동기 처리로 더 나은 성능
- **복잡도**: 상대적으로 복잡한 설정
- **검증**: 상대적으로 적은 프로덕션 사례

### 2. 데이터베이스 설계: 정규화 vs 비정규화

#### 결정: 하이브리드 접근
```sql
-- 정규화된 기본 구조
CREATE TABLE gpu_jobs (
    job_id UUID PRIMARY KEY,
    status VARCHAR(20),
    -- 기본 필드들...
);

-- JSON 필드로 유연성 확보
CREATE TABLE gpu_jobs (
    -- ...
    request_data JSONB,    -- 요청 정보
    metadata JSONB,        -- 메타데이터
    progress_data JSONB    -- 진행률 정보
);
```

#### 근거
- **쿼리 성능**: 자주 조회되는 필드는 정규화
- **유연성**: 변경 가능한 데이터는 JSON 저장
- **확장성**: 스키마 변경 없이 새로운 필드 추가 가능

### 3. 로그 수집: 실시간 vs 배치

#### 결정: 하이브리드 방식
- **실시간**: WebSocket을 통한 스트리밍 (선택적)
- **배치**: 주기적인 로그 파일 수집 (기본)

#### 근거
- **리소스 효율성**: 모든 작업에 실시간 스트리밍은 과부하
- **사용자 경험**: 중요한 작업은 실시간 모니터링 제공
- **안정성**: 네트워크 문제 시에도 배치 수집으로 복구

## 비기능적 요구사항

### 1. 성능 목표

| 메트릭 | 목표 | 측정 방법 |
|--------|------|----------|
| 작업 제출 응답 시간 | < 5초 | API 응답 시간 |
| 파일 전송 속도 | > 100MB/min | 전송 완료 시간 |
| 상태 조회 응답 시간 | < 1초 | API 응답 시간 |
| 동시 작업 처리 | 20개 | 부하 테스트 |

### 2. 가용성 목표

| 항목 | 목표 | 구현 방법 |
|------|------|----------|
| API 가용성 | 99.9% | 로드밸런싱, 헬스체크 |
| 클러스터 연결 | 95% | 재시도, 에러 복구 |
| 데이터 무결성 | 99.99% | 체크섬, 트랜잭션 |

### 3. 보안 요구사항

| 요구사항 | 구현 방안 |
|----------|----------|
| 인증 | JWT 토큰 기반 API 인증 |
| 인가 | 사용자별 작업 접근 제어 |
| 암호화 | SSH 키 기반 클러스터 통신 |
| 감사 | 모든 작업 활동 로깅 |

## 인터페이스 명세

### 1. 클러스터 명령어 인터페이스

#### 프로세스 실행 명령어 (예상)
```bash
# GPU 작업 실행
gpu-run --gpus 2 --memory 32G /path/to/script.sh

# 반환: 프로세스 ID
12345
```

#### 상태 확인 명령어
```bash
# 프로세스 상태 확인
ps -p 12345 -o stat --no-headers

# 반환: 프로세스 상태
R    # Running
S    # Sleeping
Z    # Zombie (종료됨)
```

#### 작업 완료 감지
```bash
# 성공 마커 파일 확인
test -f /cluster/ml-jobs/{job_id}/outputs/SUCCESS && echo "SUCCESS" || echo "FAILED"
```

### 2. 파일 시스템 규칙

#### 입력 파일 구조
```
/cluster/ml-jobs/{job_id}/
├── code/
│   ├── train.py           # 메인 실행 스크립트
│   ├── model.py           # 모델 정의
│   └── utils.py           # 유틸리티
├── data/
│   ├── train_data.tar.gz  # 학습 데이터
│   └── val_data.tar.gz    # 검증 데이터
├── models/
│   └── pretrained.pth     # 사전 학습 모델
└── run.sh                 # 생성된 실행 스크립트
```

#### 출력 파일 구조
```
/cluster/ml-jobs/{job_id}/
├── outputs/
│   ├── best_model.pth     # 최종 모델
│   ├── checkpoints/       # 체크포인트들
│   ├── metrics.json       # 메트릭 데이터
│   └── SUCCESS            # 성공 마커
└── logs/
    ├── training.log       # 학습 로그
    ├── error.log          # 에러 로그
    └── tensorboard/       # TensorBoard 로그
```

### 3. 실행 스크립트 템플릿

#### 기본 스크립트 구조
```bash
#!/bin/bash
#
# Auto-generated execution script
# Job ID: {job_id}
# Job Name: {job_name}
# GPU Count: {gpu_count}
#

# 작업 시작 시간 기록
echo "Job started at $(date)" > /cluster/ml-jobs/{job_id}/logs/start_time.log

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1
export JOB_ID={job_id}
export OUTPUT_DIR=/cluster/ml-jobs/{job_id}/outputs
export LOG_DIR=/cluster/ml-jobs/{job_id}/logs

# 사용자 정의 환경 변수
{environment_variables}

# 작업 디렉토리로 이동
cd /cluster/ml-jobs/{job_id}/code

# 가상환경 활성화 (필요시)
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# 실행 명령
python {entry_script} \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    {script_arguments} \
    2>&1 | tee $LOG_DIR/training.log

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo "Job completed successfully at $(date)" > $OUTPUT_DIR/SUCCESS
    echo "success" > $OUTPUT_DIR/status.txt
else
    echo "Job failed at $(date)" > $OUTPUT_DIR/FAILED
    echo "failed" > $OUTPUT_DIR/status.txt
    exit 1
fi
```

## 에러 시나리오 및 대응

### 1. 네트워크 연결 실패

#### 시나리오
- SSH 연결 타임아웃
- 네트워크 불안정으로 인한 연결 끊김

#### 대응 방안
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def execute_with_retry(self, command: str):
    """재시도 로직이 포함된 명령 실행"""
    try:
        return await self.connection.execute_command(command)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Connection failed, retrying: {e}")
        # 연결 재설정
        await self.connection.reconnect()
        raise
```

### 2. 파일 전송 실패

#### 시나리오
- 대용량 파일 전송 중 중단
- 디스크 공간 부족
- 권한 문제

#### 대응 방안
```python
async def upload_file_with_verification(self, local_path: str, remote_path: str):
    """체크섬 검증이 포함된 파일 업로드"""
    # 1. 로컬 파일 체크섬 계산
    local_checksum = calculate_checksum(local_path)
    
    # 2. 파일 전송
    await self.connection.upload_file(local_path, remote_path)
    
    # 3. 원격 파일 체크섬 확인
    remote_checksum = await self.get_remote_checksum(remote_path)
    
    # 4. 무결성 검증
    if local_checksum != remote_checksum:
        raise FileTransferError("File integrity check failed")
```

### 3. 작업 실행 실패

#### 시나리오
- GPU 리소스 부족
- 스크립트 실행 에러
- 런타임 에러

#### 대응 방안
```python
class JobFailureHandler:
    async def handle_failure(self, job: GPUJob, error: Exception):
        """작업 실패 처리"""
        # 1. 에러 분류
        error_type = self.classify_error(error)
        
        # 2. 복구 가능성 판단
        if error_type in ["resource_unavailable", "temporary_failure"]:
            # 재시도 큐에 추가
            await self.schedule_retry(job)
        else:
            # 영구 실패로 마킹
            await self.mark_as_failed(job, error)
        
        # 3. 사용자 알림
        await self.notify_user(job, error_type)
```

## 모니터링 전략

### 1. 메트릭 수집

#### 시스템 메트릭
```python
# Prometheus 메트릭 정의
from prometheus_client import Counter, Histogram, Gauge

# 작업 관련 메트릭
jobs_submitted = Counter('gpu_jobs_submitted_total', 'Total submitted jobs')
jobs_completed = Counter('gpu_jobs_completed_total', 'Total completed jobs')
jobs_failed = Counter('gpu_jobs_failed_total', 'Total failed jobs')

# 성능 메트릭
job_duration = Histogram('gpu_job_duration_seconds', 'Job duration in seconds')
file_transfer_duration = Histogram('file_transfer_duration_seconds', 'File transfer time')

# 리소스 메트릭
active_jobs = Gauge('gpu_jobs_active', 'Currently active jobs')
cluster_connections = Gauge('cluster_connections_active', 'Active SSH connections')
```

#### 비즈니스 메트릭
```python
# 사용자 활동
user_job_count = Counter('user_jobs_total', 'Jobs by user', ['user_id'])
gpu_hours_used = Counter('gpu_hours_total', 'Total GPU hours used')

# 모델 학습
model_accuracy = Histogram('model_final_accuracy', 'Final model accuracy')
training_loss = Histogram('training_final_loss', 'Final training loss')
```

### 2. 로그 구조

#### 구조화된 로깅
```python
import structlog

logger = structlog.get_logger()

# 작업 생명주기 로깅
logger.info("job_submitted",
    job_id=job_id,
    user_id=user_id,
    gpu_count=gpu_count,
    estimated_duration=estimated_duration
)

logger.info("job_started",
    job_id=job_id,
    process_id=process_id,
    cluster_path=cluster_path
)

logger.info("job_completed",
    job_id=job_id,
    duration=duration,
    final_metrics=metrics
)
```

#### 로그 레벨 정의
- **DEBUG**: 상세한 실행 정보, 개발/디버깅용
- **INFO**: 일반적인 작업 진행 상황
- **WARNING**: 주의가 필요한 상황 (재시도, 지연 등)
- **ERROR**: 에러 발생, 즉시 조치 필요
- **CRITICAL**: 시스템 전체에 영향을 미치는 심각한 문제

### 3. 알림 시스템

#### 알림 유형
```python
class NotificationTypes:
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_TIMEOUT = "job_timeout"
    SYSTEM_ERROR = "system_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
```

#### 알림 채널
- **이메일**: 작업 완료/실패 시
- **Slack/Teams**: 시스템 이상 시
- **WebSocket**: 실시간 UI 업데이트
- **Webhook**: 외부 시스템 연동

## 확장성 고려사항

### 1. 멀티 클러스터 지원

#### 설계 고려사항
```python
class MultiClusterOrchestrator:
    """여러 클러스터 관리"""
    
    def __init__(self, clusters: List[ClusterConfig]):
        self.clusters = {
            cluster.name: ClusterConnection(cluster)
            for cluster in clusters
        }
    
    async def select_optimal_cluster(self, requirements: JobRequirements):
        """최적 클러스터 선택"""
        # 리소스 가용성, 큐 길이, 성능 등을 고려한 선택
        pass
```

### 2. 작업 스케줄링

#### 우선순위 기반 스케줄링
```python
class JobScheduler:
    """작업 스케줄러"""
    
    priority_queues = {
        "high": PriorityQueue(),
        "normal": PriorityQueue(),
        "low": PriorityQueue()
    }
    
    async def schedule_job(self, job: GPUJob, priority: str = "normal"):
        """작업을 적절한 큐에 추가"""
        queue = self.priority_queues[priority]
        await queue.put((job.created_at, job))
```

### 3. 리소스 최적화

#### 동적 리소스 할당
```python
class ResourceManager:
    """리소스 관리자"""
    
    async def allocate_resources(self, requirements: ResourceRequirements):
        """요구사항에 따른 최적 리소스 할당"""
        available_gpus = await self.get_available_gpus()
        optimal_allocation = self.calculate_optimal_allocation(
            requirements, available_gpus
        )
        return optimal_allocation
```

## 테스트 전략

### 1. 테스트 피라미드

```
       E2E Tests (5%)
      ─────────────────
     Integration Tests (15%)
    ───────────────────────────
   Unit Tests (80%)
  ─────────────────────────────────
```

### 2. 모의 환경

#### Mock Cluster
```python
class MockCluster:
    """테스트용 가짜 클러스터"""
    
    def __init__(self):
        self.processes = {}
        self.files = {}
    
    def execute_command(self, cmd: str):
        """명령 실행 시뮬레이션"""
        if "gpu-run" in cmd:
            process_id = str(random.randint(10000, 99999))
            self.processes[process_id] = "running"
            return process_id, "", 0
        elif "ps -p" in cmd:
            pid = extract_pid(cmd)
            status = self.processes.get(pid, "not_found")
            return status, "", 0 if status != "not_found" else 1
```

### 3. 통합 테스트

#### 엔드투엔드 테스트 시나리오
```python
@pytest.mark.integration
async def test_complete_job_lifecycle():
    """전체 작업 생명주기 테스트"""
    # 1. 작업 제출
    job_request = GPUJobRequest(
        job_name="test_job",
        code_source="local",
        entry_script="train.py"
    )
    job = await orchestrator.submit_job(job_request)
    
    # 2. 상태 변화 확인
    assert job.status == JobStatus.PENDING
    
    # 대기 (실제 실행 시뮬레이션)
    await asyncio.sleep(5)
    
    # 3. 실행 중 상태 확인
    status = await monitor.check_job_status(job)
    assert status == JobStatus.RUNNING
    
    # 4. 완료 시뮬레이션
    await simulate_job_completion(job)
    
    # 5. 결과 수집
    results = await collector.collect_results(job)
    assert len(results['models']) > 0
```

이제 이 상세한 설계 문서를 바탕으로 실제 코드 구현을 시작하겠습니다. 어떤 컴포넌트부터 구현하시겠습니까?

1. **데이터 모델** (기본 구조)
2. **클러스터 연결 모듈** (SSH/SCP)
3. **작업 제출 API** (기본 기능)
4. **모니터링 서비스** (상태 추적)

어떤 순서로 진행하시겠습니까?