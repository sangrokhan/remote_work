"""
Application Configuration

Centralized configuration management using Pydantic settings with
environment variable support and GPU cluster integration.
"""

import os
from typing import Dict, List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """데이터베이스 설정"""
    
    # PostgreSQL connection
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/dl_platform",
        description="PostgreSQL database URL"
    )
    
    # Connection pool settings
    pool_size: int = Field(default=10, ge=1, le=50)
    max_overflow: int = Field(default=20, ge=0, le=100)
    pool_recycle: int = Field(default=3600, ge=300)
    pool_pre_ping: bool = Field(default=True)
    
    # Query settings
    echo_sql: bool = Field(default=False)
    
    class Config:
        env_prefix = "DB_"


class ClusterSettings(BaseSettings):
    """GPU 클러스터 설정"""
    
    # Connection settings
    host: str = Field(..., description="클러스터 호스트")
    port: int = Field(default=22, ge=1, le=65535)
    username: str = Field(..., description="SSH 사용자명")
    
    # Authentication
    private_key_path: Optional[str] = Field(default=None, description="SSH 개인키 파일 경로")
    password: Optional[str] = Field(default=None, description="SSH 비밀번호")
    
    # Paths
    base_path: str = Field(default="/tmp/gpu_jobs", description="클러스터 기본 작업 경로")
    python_path: str = Field(default="/usr/bin/python3", description="Python 실행 파일 경로")
    
    # Resource limits
    max_concurrent_jobs: int = Field(default=10, ge=1, le=100)
    max_gpu_per_job: int = Field(default=8, ge=1, le=16)
    default_timeout_hours: int = Field(default=24, ge=1, le=168)
    
    # Connection management
    connection_timeout: int = Field(default=30, ge=5, le=300)
    command_timeout: int = Field(default=3600, ge=60)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    
    # File transfer
    transfer_chunk_size: int = Field(default=32768, ge=1024)
    verify_transfers: bool = Field(default=True)
    
    # Monitoring
    status_check_interval: int = Field(default=30, ge=5, le=300)
    log_fetch_lines: int = Field(default=100, ge=10, le=10000)
    
    @validator('private_key_path')
    def validate_private_key_path(cls, v):
        if v and not os.path.exists(v):
            raise ValueError(f"Private key file not found: {v}")
        return v
    
    @validator('base_path')
    def validate_base_path(cls, v):
        if not v.startswith('/'):
            raise ValueError("Base path must be absolute")
        return v
    
    class Config:
        env_prefix = "CLUSTER_"


class RedisSettings(BaseSettings):
    """Redis 설정"""
    
    # Connection
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: Optional[str] = Field(default=None)
    db: int = Field(default=0, ge=0, le=15)
    
    # Connection pool
    max_connections: int = Field(default=10, ge=1, le=100)
    
    # URL format
    @property
    def redis_url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"


class CelerySettings(BaseSettings):
    """Celery 워커 설정"""
    
    # Broker and backend
    broker_url: Optional[str] = Field(default=None)
    result_backend: Optional[str] = Field(default=None)
    
    # Task settings
    task_serializer: str = Field(default="json")
    result_serializer: str = Field(default="json")
    accept_content: List[str] = Field(default=["json"])
    timezone: str = Field(default="UTC")
    
    # Worker settings
    worker_concurrency: int = Field(default=4, ge=1, le=32)
    worker_prefetch_multiplier: int = Field(default=1, ge=1, le=10)
    task_acks_late: bool = Field(default=True)
    worker_disable_rate_limits: bool = Field(default=False)
    
    # Task routing
    task_routes: Dict[str, Dict] = Field(default_factory=lambda: {
        'src.worker.gpu_tasks.*': {'queue': 'gpu_queue'},
        'src.worker.file_tasks.*': {'queue': 'file_queue'},
        'src.worker.monitor_tasks.*': {'queue': 'monitor_queue'}
    })
    
    # Result settings
    result_expires: int = Field(default=3600, ge=300)
    
    class Config:
        env_prefix = "CELERY_"


class StorageSettings(BaseSettings):
    """스토리지 설정"""
    
    # Local storage paths
    results_path: str = Field(default="/app/results", description="결과 저장 경로")
    uploads_path: str = Field(default="/app/uploads", description="업로드 파일 경로")
    temp_path: str = Field(default="/tmp/dl_platform", description="임시 파일 경로")
    
    # Storage limits
    max_upload_size_mb: int = Field(default=1024, ge=1)  # 1GB
    max_total_storage_gb: int = Field(default=100, ge=1)
    
    # Cleanup settings
    result_retention_days: int = Field(default=30, ge=1, le=365)
    temp_file_retention_hours: int = Field(default=24, ge=1, le=168)
    
    # Archive settings
    auto_archive_after_days: int = Field(default=7, ge=1, le=30)
    compression_level: int = Field(default=6, ge=1, le=9)
    
    @validator('results_path', 'uploads_path', 'temp_path')
    def validate_paths(cls, v):
        path = os.path.abspath(v)
        os.makedirs(path, exist_ok=True)
        return path
    
    class Config:
        env_prefix = "STORAGE_"


class SecuritySettings(BaseSettings):
    """보안 설정"""
    
    # JWT settings
    secret_key: str = Field(..., description="JWT 서명용 시크릿 키")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=1440, ge=30)  # 24 hours
    
    # API security
    cors_origins: List[str] = Field(default=["*"])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: List[str] = Field(default=["*"])
    cors_allow_headers: List[str] = Field(default=["*"])
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    
    # File upload security
    allowed_file_types: List[str] = Field(default=[
        ".py", ".txt", ".json", ".yaml", ".yml", ".csv",
        ".pth", ".pt", ".h5", ".pkl", ".joblib", ".onnx"
    ])
    
    # Command validation
    allowed_python_packages: List[str] = Field(default=[
        "torch", "torchvision", "torchaudio", "transformers",
        "scikit-learn", "numpy", "pandas", "matplotlib",
        "seaborn", "jupyter", "tensorboard"
    ])
    
    blocked_commands: List[str] = Field(default=[
        "rm -rf /", "dd", "mkfs", "fdisk", "parted",
        "shutdown", "reboot", "halt", "init"
    ])
    
    class Config:
        env_prefix = "SECURITY_"


class LoggingSettings(BaseSettings):
    """로깅 설정"""
    
    # Log levels
    log_level: str = Field(default="INFO")
    uvicorn_log_level: str = Field(default="INFO")
    
    # Log format
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # File logging
    log_file_path: Optional[str] = Field(default=None)
    log_rotation_size: str = Field(default="100 MB")
    log_backup_count: int = Field(default=5, ge=1, le=20)
    
    # Structured logging
    structured_logging: bool = Field(default=True)
    log_json_format: bool = Field(default=False)
    
    class Config:
        env_prefix = "LOG_"


class MonitoringSettings(BaseSettings):
    """모니터링 설정"""
    
    # Health checks
    health_check_interval: int = Field(default=60, ge=10, le=3600)
    cluster_health_timeout: int = Field(default=30, ge=5, le=300)
    
    # Metrics collection
    metrics_enabled: bool = Field(default=True)
    metrics_retention_days: int = Field(default=90, ge=1, le=365)
    
    # Alerting
    alert_on_job_failure: bool = Field(default=True)
    alert_on_cluster_down: bool = Field(default=True)
    max_failed_jobs_threshold: int = Field(default=5, ge=1, le=100)
    
    # Performance monitoring
    slow_job_threshold_minutes: int = Field(default=60, ge=1)
    high_memory_threshold_mb: int = Field(default=8192, ge=1024)
    
    class Config:
        env_prefix = "MONITOR_"


class AppSettings(BaseSettings):
    """애플리케이션 메인 설정"""
    
    # Application info
    app_name: str = Field(default="Deep Learning Platform")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False)
    
    # Environment
    environment: str = Field(default="development")
    
    # Settings composition
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    cluster: ClusterSettings = Field(default_factory=ClusterSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Auto-configure Celery URLs from Redis settings
        if not self.celery.broker_url:
            self.celery.broker_url = self.redis.redis_url
        if not self.celery.result_backend:
            self.celery.result_backend = self.redis.redis_url
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed = ["development", "testing", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = AppSettings()


# Configuration utilities

def get_database_url() -> str:
    """데이터베이스 URL 반환"""
    return settings.database.database_url


def get_cluster_config() -> Dict:
    """클러스터 설정 딕셔너리 반환"""
    return {
        "host": settings.cluster.host,
        "port": settings.cluster.port,
        "username": settings.cluster.username,
        "private_key_path": settings.cluster.private_key_path,
        "password": settings.cluster.password,
        "base_path": settings.cluster.base_path,
        "timeout": settings.cluster.connection_timeout,
        "max_retries": settings.cluster.max_retries,
        "retry_delay": settings.cluster.retry_delay
    }


def get_redis_url() -> str:
    """Redis URL 반환"""
    return settings.redis.redis_url


def get_celery_config() -> Dict:
    """Celery 설정 딕셔너리 반환"""
    return {
        "broker_url": settings.celery.broker_url,
        "result_backend": settings.celery.result_backend,
        "task_serializer": settings.celery.task_serializer,
        "result_serializer": settings.celery.result_serializer,
        "accept_content": settings.celery.accept_content,
        "timezone": settings.celery.timezone,
        "task_routes": settings.celery.task_routes,
        "result_expires": settings.celery.result_expires,
        "worker_concurrency": settings.celery.worker_concurrency,
        "worker_prefetch_multiplier": settings.celery.worker_prefetch_multiplier,
        "task_acks_late": settings.celery.task_acks_late,
        "worker_disable_rate_limits": settings.celery.worker_disable_rate_limits
    }


def is_production() -> bool:
    """프로덕션 환경 확인"""
    return settings.environment == "production"


def is_development() -> bool:
    """개발 환경 확인"""
    return settings.environment == "development"


# Validation functions

def validate_cluster_connection() -> bool:
    """클러스터 연결 설정 검증"""
    try:
        if not settings.cluster.host:
            return False
        if not settings.cluster.username:
            return False
        if not (settings.cluster.private_key_path or settings.cluster.password):
            return False
        return True
    except Exception:
        return False


def validate_storage_paths() -> bool:
    """스토리지 경로 검증"""
    try:
        paths = [
            settings.storage.results_path,
            settings.storage.uploads_path,
            settings.storage.temp_path
        ]
        
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            
            # 쓰기 권한 확인
            test_file = os.path.join(path, ".write_test")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except (OSError, IOError):
                return False
        
        return True
    except Exception:
        return False


# Environment file template

ENV_TEMPLATE = """
# Deep Learning Platform Configuration

# Database Settings
DB_DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/dl_platform
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_ECHO_SQL=false

# GPU Cluster Settings
CLUSTER_HOST=your-cluster-host.com
CLUSTER_PORT=22
CLUSTER_USERNAME=your-username
CLUSTER_PRIVATE_KEY_PATH=/path/to/your/private/key
# CLUSTER_PASSWORD=your-password  # Alternative to private key
CLUSTER_BASE_PATH=/tmp/gpu_jobs
CLUSTER_PYTHON_PATH=/usr/bin/python3
CLUSTER_MAX_CONCURRENT_JOBS=10
CLUSTER_MAX_GPU_PER_JOB=8
CLUSTER_DEFAULT_TIMEOUT_HOURS=24
CLUSTER_CONNECTION_TIMEOUT=30
CLUSTER_COMMAND_TIMEOUT=3600
CLUSTER_MAX_RETRIES=3
CLUSTER_RETRY_DELAY=1.0

# Redis Settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
# REDIS_PASSWORD=your-redis-password

# Celery Settings
CELERY_WORKER_CONCURRENCY=4
CELERY_RESULT_EXPIRES=3600

# Storage Settings
STORAGE_RESULTS_PATH=/app/results
STORAGE_UPLOADS_PATH=/app/uploads
STORAGE_TEMP_PATH=/tmp/dl_platform
STORAGE_MAX_UPLOAD_SIZE_MB=1024
STORAGE_RESULT_RETENTION_DAYS=30

# Security Settings
SECURITY_SECRET_KEY=your-secret-key-here-change-in-production
SECURITY_ACCESS_TOKEN_EXPIRE_MINUTES=1440
SECURITY_RATE_LIMIT_PER_MINUTE=60

# Logging Settings
LOG_LEVEL=INFO
LOG_FILE_PATH=/var/log/dl_platform.log

# Monitoring Settings
MONITOR_HEALTH_CHECK_INTERVAL=60
MONITOR_METRICS_ENABLED=true
MONITOR_ALERT_ON_JOB_FAILURE=true

# Application Settings
APP_NAME=Deep Learning Platform
APP_VERSION=0.1.0
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false
APP_ENVIRONMENT=development
"""


def create_env_file(path: str = ".env") -> None:
    """환경 설정 파일 생성"""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(ENV_TEMPLATE.strip())
        print(f"Created environment file: {path}")
    else:
        print(f"Environment file already exists: {path}")


# Configuration validation

def validate_all_settings() -> Dict[str, bool]:
    """모든 설정 검증"""
    return {
        "cluster_connection": validate_cluster_connection(),
        "storage_paths": validate_storage_paths(),
        "database_url": bool(settings.database.database_url),
        "redis_connection": bool(settings.redis.host),
        "security_key": bool(settings.security.secret_key),
    }


def get_config_summary() -> Dict:
    """설정 요약 정보"""
    return {
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug
        },
        "cluster": {
            "host": settings.cluster.host,
            "port": settings.cluster.port,
            "username": settings.cluster.username,
            "max_gpus": settings.cluster.max_gpu_per_job,
            "max_jobs": settings.cluster.max_concurrent_jobs
        },
        "storage": {
            "results_path": settings.storage.results_path,
            "max_upload_mb": settings.storage.max_upload_size_mb,
            "retention_days": settings.storage.result_retention_days
        },
        "validation": validate_all_settings()
    }