"""
GPU Cluster Data Models

Defines the data models and schemas for GPU cluster job management.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class JobStatus(str, Enum):
    """작업 상태 열거형"""
    PENDING = "pending"        # 제출됨, 대기 중
    QUEUED = "queued"         # 큐에 들어감
    PREPARING = "preparing"    # 파일 전송 중
    RUNNING = "running"        # 실행 중
    COMPLETED = "completed"    # 정상 완료
    FAILED = "failed"         # 실패
    CANCELLED = "cancelled"    # 취소됨
    TIMEOUT = "timeout"       # 시간 초과


class CodeSource(str, Enum):
    """코드 소스 타입"""
    GIT = "git"               # Git 저장소
    LOCAL = "local"           # 로컬 파일
    MANUAL = "manual"         # 수동 업로드
    EXISTING = "existing"     # 클러스터 기존 코드


class GPUJobRequest(BaseModel):
    """GPU 작업 요청 모델"""
    
    # 기본 정보
    job_name: str = Field(..., description="작업 이름", max_length=255)
    gpu_count: int = Field(default=1, ge=1, le=8, description="필요한 GPU 수")
    
    # 파일 경로
    model_file_path: Optional[str] = Field(None, description="로컬 모델 파일 경로")
    dataset_path: Optional[str] = Field(None, description="로컬 데이터셋 경로")
    
    # 코드 소스
    code_source: CodeSource = Field(..., description="코드 소스 타입")
    code_path: Optional[str] = Field(None, description="코드 위치 (URL 또는 경로)")
    
    # 실행 설정
    entry_script: str = Field(..., description="실행할 메인 스크립트")
    script_args: Dict[str, str] = Field(default_factory=dict, description="스크립트 인자")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="환경 변수")
    
    # 리소스 설정
    memory_gb: Optional[int] = Field(None, ge=1, le=256, description="메모리 요구사항 (GB)")
    timeout_hours: Optional[int] = Field(24, ge=1, le=168, description="최대 실행 시간 (시간)")
    
    # 알림 설정
    notification_email: Optional[str] = Field(None, description="완료 알림 이메일")
    webhook_url: Optional[str] = Field(None, description="완료 알림 웹훅 URL")

    @validator('job_name')
    def validate_job_name(cls, v):
        """작업 이름 검증"""
        if not v.strip():
            raise ValueError("Job name cannot be empty")
        # 파일시스템 안전한 이름 확인
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Job name contains invalid characters: {invalid_chars}")
        return v.strip()

    @validator('entry_script')
    def validate_entry_script(cls, v):
        """실행 스크립트 검증"""
        if not v.endswith('.py'):
            raise ValueError("Entry script must be a Python file (.py)")
        return v

    @validator('code_path')
    def validate_code_path(cls, v, values):
        """코드 경로 검증"""
        code_source = values.get('code_source')
        
        # MANUAL 업로드의 경우 code_path는 선택적
        if code_source == CodeSource.MANUAL:
            return v
            
        # GIT과 LOCAL은 code_path 필수
        if code_source == CodeSource.GIT:
            if not v:
                raise ValueError("Git code path is required")
            if not (v.startswith('http://') or v.startswith('https://') or v.startswith('git@')):
                raise ValueError("Git code path must be a valid repository URL")
        elif code_source == CodeSource.LOCAL:
            if not v:
                raise ValueError("Local code path is required")
        elif code_source == CodeSource.EXISTING:
            if not v:
                raise ValueError("Existing code path is required")
        return v


class GPUJob(BaseModel):
    """GPU 작업 상태 모델"""
    
    # 식별 정보
    job_id: str = Field(default_factory=lambda: str(uuid4()), description="고유 작업 ID")
    job_name: str = Field(..., description="작업 이름")
    user_id: str = Field(..., description="제출한 사용자 ID")
    
    # 프로세스 정보
    process_id: Optional[str] = Field(None, description="클러스터 프로세스 ID")
    status: JobStatus = Field(default=JobStatus.PENDING, description="현재 상태")
    
    # 경로 정보
    cluster_path: str = Field(..., description="클러스터 내 작업 디렉토리")
    local_results_path: Optional[str] = Field(None, description="로컬 결과 저장 경로")
    
    # 시간 정보
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    queued_at: Optional[datetime] = Field(None, description="큐 진입 시간")
    started_at: Optional[datetime] = Field(None, description="실행 시작 시간")
    completed_at: Optional[datetime] = Field(None, description="완료 시간")
    
    # 실행 정보
    gpu_count: int = Field(..., ge=1, description="할당된 GPU 수")
    exit_code: Optional[int] = Field(None, description="프로세스 종료 코드")
    error_message: Optional[str] = Field(None, description="에러 메시지")
    
    # 요청 정보 (원본 보존)
    original_request: GPUJobRequest = Field(..., description="원본 요청 정보")
    
    # 메타데이터
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")

    @property
    def duration_seconds(self) -> Optional[int]:
        """작업 실행 시간 (초)"""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None

    @property
    def is_active(self) -> bool:
        """작업이 활성 상태인지 확인"""
        return self.status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.PREPARING, JobStatus.RUNNING]

    @property
    def is_finished(self) -> bool:
        """작업이 완료 상태인지 확인"""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT]

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobProgress(BaseModel):
    """작업 진행 상황 모델"""
    
    job_id: str = Field(..., description="작업 ID")
    
    # 진행률 정보
    current_epoch: Optional[int] = Field(None, ge=0, description="현재 에포크")
    total_epochs: Optional[int] = Field(None, ge=1, description="전체 에포크")
    current_step: Optional[int] = Field(None, ge=0, description="현재 스텝")
    total_steps: Optional[int] = Field(None, ge=1, description="전체 스텝")
    
    # 학습 메트릭
    current_loss: Optional[float] = Field(None, description="현재 손실값")
    current_accuracy: Optional[float] = Field(None, ge=0, le=1, description="현재 정확도")
    learning_rate: Optional[float] = Field(None, gt=0, description="현재 학습률")
    
    # 시스템 메트릭
    gpu_utilization: Optional[float] = Field(None, ge=0, le=100, description="GPU 사용률 (%)")
    memory_usage_gb: Optional[float] = Field(None, ge=0, description="메모리 사용량 (GB)")
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU 사용률 (%)")
    
    # 시간 추정
    estimated_time_remaining: Optional[int] = Field(None, ge=0, description="예상 남은 시간 (초)")
    
    # 추가 메트릭
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="사용자 정의 메트릭")
    
    # 메타데이터
    timestamp: datetime = Field(default_factory=datetime.now, description="진행 정보 업데이트 시간")

    @property
    def epoch_progress_percent(self) -> Optional[float]:
        """에포크 진행률 (%)"""
        if self.current_epoch is not None and self.total_epochs is not None:
            return (self.current_epoch / self.total_epochs) * 100
        return None

    @property
    def step_progress_percent(self) -> Optional[float]:
        """스텝 진행률 (%)"""
        if self.current_step is not None and self.total_steps is not None:
            return (self.current_step / self.total_steps) * 100
        return None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobResults(BaseModel):
    """작업 결과 모델"""
    
    job_id: str = Field(..., description="작업 ID")
    
    # 결과 파일들
    model_files: List[str] = Field(default_factory=list, description="모델 파일 경로 목록")
    log_files: List[str] = Field(default_factory=list, description="로그 파일 경로 목록")
    output_files: List[str] = Field(default_factory=list, description="기타 출력 파일 경로 목록")
    
    # 메트릭
    final_metrics: Dict[str, float] = Field(default_factory=dict, description="최종 메트릭")
    training_history: Dict[str, List[float]] = Field(default_factory=dict, description="학습 히스토리")
    
    # 메타데이터
    collected_at: datetime = Field(default_factory=datetime.now, description="수집 시간")
    total_size_mb: Optional[float] = Field(None, description="총 결과 파일 크기 (MB)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ClusterConfig(BaseModel):
    """클러스터 설정 모델"""
    
    # 연결 정보
    host: str = Field(..., description="클러스터 호스트")
    username: str = Field(..., description="SSH 사용자명")
    key_path: str = Field(..., description="SSH 키 파일 경로")
    port: int = Field(default=22, ge=1, le=65535, description="SSH 포트")
    
    # 경로 설정
    base_path: str = Field(default="/cluster/ml-jobs", description="클러스터 내 기본 작업 경로")
    
    # 연결 설정
    timeout: int = Field(default=30, ge=5, description="SSH 연결 타임아웃 (초)")
    max_retries: int = Field(default=3, ge=1, description="최대 재시도 횟수")
    
    # 파일 전송 설정
    chunk_size: int = Field(default=8192, ge=1024, description="파일 전송 청크 크기")
    max_file_size_gb: int = Field(default=100, ge=1, description="최대 파일 크기 (GB)")
    
    class Config:
        schema_extra = {
            "example": {
                "host": "gpu-cluster.example.com",
                "username": "ml-user",
                "key_path": "/home/user/.ssh/id_rsa",
                "port": 22,
                "base_path": "/cluster/ml-jobs",
                "timeout": 30,
                "max_retries": 3
            }
        }


class ResourceRequirements(BaseModel):
    """리소스 요구사항 모델"""
    
    gpu_count: int = Field(..., ge=1, le=8, description="필요한 GPU 수")
    memory_gb: Optional[int] = Field(None, ge=1, le=512, description="메모리 요구사항 (GB)")
    cpu_cores: Optional[int] = Field(None, ge=1, le=64, description="CPU 코어 수")
    disk_gb: Optional[int] = Field(None, ge=1, le=1000, description="디스크 공간 (GB)")
    
    # GPU 타입 지정 (선택사항)
    gpu_type: Optional[str] = Field(None, description="GPU 타입 (예: V100, A100)")
    
    # 우선순위
    priority: int = Field(default=1, ge=1, le=5, description="작업 우선순위 (1=높음, 5=낮음)")


class JobLogEntry(BaseModel):
    """작업 로그 엔트리"""
    
    job_id: str = Field(..., description="작업 ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="로그 시간")
    level: str = Field(..., description="로그 레벨 (DEBUG, INFO, WARNING, ERROR)")
    message: str = Field(..., description="로그 메시지")
    source: str = Field(default="cluster", description="로그 소스")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobTemplate(BaseModel):
    """작업 템플릿 모델"""
    
    template_id: str = Field(default_factory=lambda: str(uuid4()), description="템플릿 ID")
    name: str = Field(..., description="템플릿 이름")
    description: Optional[str] = Field(None, description="템플릿 설명")
    
    # 기본 설정
    default_gpu_count: int = Field(default=1, description="기본 GPU 수")
    default_script_args: Dict[str, str] = Field(default_factory=dict, description="기본 스크립트 인자")
    default_environment_vars: Dict[str, str] = Field(default_factory=dict, description="기본 환경 변수")
    
    # 제약사항
    required_files: List[str] = Field(default_factory=list, description="필수 파일 목록")
    max_gpu_count: int = Field(default=8, description="최대 GPU 수")
    
    # 메타데이터
    created_by: str = Field(..., description="생성자 ID")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    is_public: bool = Field(default=False, description="공개 템플릿 여부")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobStatistics(BaseModel):
    """작업 통계 모델"""
    
    # 기본 통계
    total_jobs: int = Field(default=0, description="총 작업 수")
    completed_jobs: int = Field(default=0, description="완료된 작업 수")
    failed_jobs: int = Field(default=0, description="실패한 작업 수")
    running_jobs: int = Field(default=0, description="실행 중인 작업 수")
    
    # 리소스 사용량
    total_gpu_hours: float = Field(default=0.0, description="총 GPU 사용 시간")
    average_job_duration: Optional[float] = Field(None, description="평균 작업 시간 (초)")
    
    # 성능 메트릭
    success_rate: float = Field(default=0.0, ge=0, le=1, description="성공률")
    average_queue_time: Optional[float] = Field(None, description="평균 대기 시간 (초)")
    
    # 시간 범위
    start_date: datetime = Field(..., description="통계 시작 일자")
    end_date: datetime = Field(..., description="통계 종료 일자")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobNotification(BaseModel):
    """작업 알림 모델"""
    
    job_id: str = Field(..., description="작업 ID")
    notification_type: str = Field(..., description="알림 타입")
    recipient: str = Field(..., description="수신자 (이메일 또는 웹훅 URL)")
    message: str = Field(..., description="알림 메시지")
    
    # 메타데이터
    sent_at: datetime = Field(default_factory=datetime.now, description="전송 시간")
    delivery_status: str = Field(default="pending", description="전송 상태")
    retry_count: int = Field(default=0, description="재시도 횟수")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# API Response Models
class JobSubmissionResponse(BaseModel):
    """작업 제출 응답 모델"""
    
    job_id: str = Field(..., description="생성된 작업 ID")
    status: JobStatus = Field(..., description="초기 상태")
    cluster_path: str = Field(..., description="클러스터 작업 경로")
    estimated_start_time: Optional[datetime] = Field(None, description="예상 시작 시간")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobStatusResponse(BaseModel):
    """작업 상태 응답 모델"""
    
    job_id: str = Field(..., description="작업 ID")
    status: JobStatus = Field(..., description="현재 상태")
    process_id: Optional[str] = Field(None, description="프로세스 ID")
    
    # 시간 정보
    created_at: datetime = Field(..., description="생성 시간")
    started_at: Optional[datetime] = Field(None, description="시작 시간")
    completed_at: Optional[datetime] = Field(None, description="완료 시간")
    
    # 진행 정보
    progress: Optional[JobProgress] = Field(None, description="진행 상황")
    
    # 에러 정보
    error_message: Optional[str] = Field(None, description="에러 메시지")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobListResponse(BaseModel):
    """작업 목록 응답 모델"""
    
    total: int = Field(..., description="전체 작업 수")
    page: int = Field(..., description="현재 페이지")
    limit: int = Field(..., description="페이지당 항목 수")
    jobs: List[JobStatusResponse] = Field(..., description="작업 목록")


class JobLogsResponse(BaseModel):
    """작업 로그 응답 모델"""
    
    job_id: str = Field(..., description="작업 ID")
    logs: List[str] = Field(..., description="로그 라인 목록")
    total_lines: int = Field(..., description="전체 로그 라인 수")
    has_more: bool = Field(..., description="더 많은 로그가 있는지 여부")
    last_update: datetime = Field(..., description="마지막 업데이트 시간")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobResultsResponse(BaseModel):
    """작업 결과 응답 모델"""
    
    job_id: str = Field(..., description="작업 ID")
    status: JobStatus = Field(..., description="작업 상태")
    results: JobResults = Field(..., description="수집된 결과")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }