"""
Database Models for GPU Cluster Integration

SQLAlchemy models for persisting GPU job data, user information,
and system configuration.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, JSON,
    ForeignKey, Index, CheckConstraint
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .database import Base
from ..ml.cluster.models import JobStatus, CodeSource


class User(Base):
    """사용자 모델"""
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    gpu_jobs: Mapped[List["GPUJobDB"]] = relationship("GPUJobDB", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index("ix_users_username", "username"),
        Index("ix_users_email", "email"),
        Index("ix_users_active", "is_active"),
    )


class GPUJobDB(Base):
    """GPU 작업 데이터베이스 모델"""
    __tablename__ = "gpu_jobs"
    
    # Primary identifiers
    job_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    job_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Job configuration
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False)
    code_source: Mapped[str] = mapped_column(String(50), nullable=False)  # CodeSource enum
    entry_script: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Code source details (JSON field)
    git_url: Mapped[Optional[str]] = mapped_column(String(500))
    git_branch: Mapped[Optional[str]] = mapped_column(String(255))
    git_commit: Mapped[Optional[str]] = mapped_column(String(255))
    manual_files: Mapped[Optional[Dict]] = mapped_column(JSON)  # List of uploaded files
    
    # Model and data files
    model_files: Mapped[Optional[Dict]] = mapped_column(JSON)  # Dict of model file paths
    data_files: Mapped[Optional[Dict]] = mapped_column(JSON)   # Dict of data file paths
    
    # Execution environment
    environment_vars: Mapped[Optional[Dict]] = mapped_column(JSON)
    python_packages: Mapped[Optional[List[str]]] = mapped_column(JSON)
    docker_image: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Job status and timing
    status: Mapped[str] = mapped_column(String(50), nullable=False, default=JobStatus.PENDING.value)
    cluster_job_id: Mapped[Optional[str]] = mapped_column(String(255))
    cluster_path: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Progress tracking
    current_epoch: Mapped[Optional[int]] = mapped_column(Integer)
    total_epochs: Mapped[Optional[int]] = mapped_column(Integer)
    current_loss: Mapped[Optional[float]] = mapped_column(Float)
    current_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    
    # Resource usage
    memory_usage_mb: Mapped[Optional[float]] = mapped_column(Float)
    gpu_usage_percent: Mapped[Optional[float]] = mapped_column(Float)
    
    # Error information
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    exit_code: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Results summary
    final_metrics: Mapped[Optional[Dict]] = mapped_column(JSON)
    output_size_mb: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="gpu_jobs")
    results: Mapped[Optional["JobResultDB"]] = relationship("JobResultDB", back_populates="job", uselist=False)
    logs: Mapped[List["JobLogDB"]] = relationship("JobLogDB", back_populates="job")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("gpu_count >= 1 AND gpu_count <= 8", name="check_gpu_count"),
        CheckConstraint("current_epoch >= 0", name="check_current_epoch"),
        CheckConstraint("total_epochs >= 1", name="check_total_epochs"),
        CheckConstraint("memory_usage_mb >= 0", name="check_memory_usage"),
        CheckConstraint("gpu_usage_percent >= 0 AND gpu_usage_percent <= 100", name="check_gpu_usage"),
        Index("ix_gpu_jobs_user", "user_id"),
        Index("ix_gpu_jobs_status", "status"),
        Index("ix_gpu_jobs_created", "created_at"),
        Index("ix_gpu_jobs_cluster_id", "cluster_job_id"),
    )


class JobResultDB(Base):
    """작업 결과 데이터베이스 모델"""
    __tablename__ = "job_results"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(255), ForeignKey("gpu_jobs.job_id"), unique=True, nullable=False)
    
    # File information
    model_files: Mapped[List[str]] = mapped_column(JSON, default=list)
    log_files: Mapped[List[str]] = mapped_column(JSON, default=list)
    output_files: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Metrics and history
    final_metrics: Mapped[Dict] = mapped_column(JSON, default=dict)
    training_history: Mapped[Dict] = mapped_column(JSON, default=dict)
    
    # Storage information
    total_size_mb: Mapped[float] = mapped_column(Float, default=0.0)
    archive_path: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Collection metadata
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    collection_duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    job: Mapped["GPUJobDB"] = relationship("GPUJobDB", back_populates="results")
    
    # Indexes
    __table_args__ = (
        Index("ix_job_results_job_id", "job_id"),
        Index("ix_job_results_collected", "collected_at"),
        CheckConstraint("total_size_mb >= 0", name="check_total_size"),
    )


class JobLogDB(Base):
    """작업 로그 데이터베이스 모델"""
    __tablename__ = "job_logs"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(255), ForeignKey("gpu_jobs.job_id"), nullable=False)
    
    # Log metadata
    log_type: Mapped[str] = mapped_column(String(50), nullable=False)  # training, error, system
    log_level: Mapped[str] = mapped_column(String(20), default="INFO")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Log content
    message: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(255))  # 로그 출처 (파일명 등)
    
    # Structured data (optional)
    log_metadata: Mapped[Optional[Dict]] = mapped_column(JSON)
    
    # Relationships
    job: Mapped["GPUJobDB"] = relationship("GPUJobDB", back_populates="logs")
    
    # Indexes
    __table_args__ = (
        Index("ix_job_logs_job_id", "job_id"),
        Index("ix_job_logs_timestamp", "timestamp"),
        Index("ix_job_logs_type", "log_type"),
        Index("ix_job_logs_level", "log_level"),
    )


class ClusterConfigDB(Base):
    """클러스터 설정 데이터베이스 모델"""
    __tablename__ = "cluster_configs"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Cluster identification
    cluster_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    cluster_host: Mapped[str] = mapped_column(String(255), nullable=False)
    cluster_port: Mapped[int] = mapped_column(Integer, default=22)
    
    # Authentication
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    private_key_path: Mapped[Optional[str]] = mapped_column(String(500))
    password_encrypted: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Configuration
    base_path: Mapped[str] = mapped_column(String(500), nullable=False)
    max_concurrent_jobs: Mapped[int] = mapped_column(Integer, default=10)
    max_gpu_per_job: Mapped[int] = mapped_column(Integer, default=8)
    default_timeout_hours: Mapped[int] = mapped_column(Integer, default=24)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_health_check: Mapped[Optional[datetime]] = mapped_column(DateTime)
    health_status: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("ix_cluster_configs_name", "cluster_name"),
        Index("ix_cluster_configs_active", "is_active"),
        CheckConstraint("cluster_port > 0 AND cluster_port <= 65535", name="check_port_range"),
        CheckConstraint("max_concurrent_jobs > 0", name="check_max_jobs"),
        CheckConstraint("max_gpu_per_job > 0", name="check_max_gpu"),
    )


class JobTemplateDB(Base):
    """작업 템플릿 데이터베이스 모델"""
    __tablename__ = "job_templates"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Template identification
    template_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100), nullable=False)  # training, inference, preprocessing
    
    # Template configuration
    default_gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    default_code_source: Mapped[str] = mapped_column(String(50), nullable=False)
    default_entry_script: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Environment settings
    default_python_packages: Mapped[Optional[List[str]]] = mapped_column(JSON)
    default_environment_vars: Mapped[Optional[Dict]] = mapped_column(JSON)
    default_docker_image: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Template metadata
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    creator: Mapped["User"] = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index("ix_job_templates_name", "template_name"),
        Index("ix_job_templates_category", "category"),
        Index("ix_job_templates_public", "is_public"),
        Index("ix_job_templates_creator", "created_by"),
        CheckConstraint("default_gpu_count >= 1 AND default_gpu_count <= 8", name="check_template_gpu_count"),
    )


class JobMetricDB(Base):
    """작업 메트릭 히스토리 모델"""
    __tablename__ = "job_metrics"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(255), ForeignKey("gpu_jobs.job_id"), nullable=False)
    
    # Metric information
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    epoch: Mapped[Optional[int]] = mapped_column(Integer)
    step: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Metadata
    metric_type: Mapped[str] = mapped_column(String(50), default="training")  # training, validation, test
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Additional context
    metric_metadata: Mapped[Optional[Dict]] = mapped_column(JSON)
    
    # Relationships
    job: Mapped["GPUJobDB"] = relationship("GPUJobDB")
    
    # Indexes
    __table_args__ = (
        Index("ix_job_metrics_job", "job_id"),
        Index("ix_job_metrics_name", "metric_name"),
        Index("ix_job_metrics_epoch", "epoch"),
        Index("ix_job_metrics_recorded", "recorded_at"),
        Index("ix_job_metrics_type", "metric_type"),
        # Composite indexes for common queries
        Index("ix_job_metrics_job_name", "job_id", "metric_name"),
        Index("ix_job_metrics_job_epoch", "job_id", "epoch"),
    )


class SystemEventDB(Base):
    """시스템 이벤트 로그 모델"""
    __tablename__ = "system_events"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Event information
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)  # job_submitted, job_completed, error
    event_level: Mapped[str] = mapped_column(String(20), default="INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Context
    job_id: Mapped[Optional[str]] = mapped_column(String(255))
    user_id: Mapped[Optional[str]] = mapped_column(String)
    cluster_name: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Additional data
    event_data: Mapped[Optional[Dict]] = mapped_column(JSON)
    
    # Timing
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("ix_system_events_type", "event_type"),
        Index("ix_system_events_level", "event_level"),
        Index("ix_system_events_timestamp", "timestamp"),
        Index("ix_system_events_job", "job_id"),
        Index("ix_system_events_user", "user_id"),
    )


class JobQueueDB(Base):
    """작업 큐 관리 모델"""
    __tablename__ = "job_queue"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(255), ForeignKey("gpu_jobs.job_id"), nullable=False)
    
    # Queue information
    priority: Mapped[int] = mapped_column(Integer, default=5)  # 1-10, 10 = highest priority
    queue_position: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_start_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    estimated_duration_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Queue status
    queue_status: Mapped[str] = mapped_column(String(50), default="QUEUED")  # QUEUED, SCHEDULED, RUNNING, COMPLETED
    
    # Dependencies
    depends_on_jobs: Mapped[Optional[List[str]]] = mapped_column(JSON)  # List of job IDs
    
    # Timestamps
    queued_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    job: Mapped["GPUJobDB"] = relationship("GPUJobDB")
    
    # Indexes
    __table_args__ = (
        Index("ix_job_queue_job", "job_id"),
        Index("ix_job_queue_priority", "priority"),
        Index("ix_job_queue_status", "queue_status"),
        Index("ix_job_queue_queued", "queued_at"),
        CheckConstraint("priority >= 1 AND priority <= 10", name="check_priority_range"),
    )


# Model conversion utilities

def gpu_job_to_db(gpu_job, user_id: str) -> GPUJobDB:
    """GPU 작업 모델을 데이터베이스 모델로 변환"""
    return GPUJobDB(
        job_id=gpu_job.job_id,
        user_id=user_id,
        job_name=gpu_job.job_name,
        gpu_count=gpu_job.gpu_count,
        code_source=gpu_job.code_source.value,
        entry_script=gpu_job.entry_script,
        git_url=gpu_job.git_url,
        git_branch=gpu_job.git_branch,
        git_commit=gpu_job.git_commit,
        manual_files=gpu_job.manual_files,
        model_files=gpu_job.model_files,
        data_files=gpu_job.data_files,
        environment_vars=gpu_job.environment_vars,
        python_packages=gpu_job.python_packages,
        docker_image=gpu_job.docker_image,
        status=gpu_job.status.value,
        cluster_job_id=gpu_job.cluster_job_id,
        cluster_path=gpu_job.cluster_path,
        submitted_at=gpu_job.submitted_at,
        started_at=gpu_job.started_at,
        completed_at=gpu_job.completed_at,
        current_epoch=gpu_job.current_epoch,
        total_epochs=gpu_job.total_epochs,
        current_loss=gpu_job.current_loss,
        current_accuracy=gpu_job.current_accuracy,
        memory_usage_mb=gpu_job.memory_usage_mb,
        gpu_usage_percent=gpu_job.gpu_usage_percent,
        error_message=gpu_job.error_message,
        exit_code=gpu_job.exit_code,
        final_metrics=gpu_job.final_metrics
    )


def db_to_gpu_job(db_job: GPUJobDB):
    """데이터베이스 모델을 GPU 작업 모델로 변환"""
    from ..ml.cluster.models import GPUJob
    
    return GPUJob(
        job_id=db_job.job_id,
        job_name=db_job.job_name,
        gpu_count=db_job.gpu_count,
        code_source=CodeSource(db_job.code_source),
        entry_script=db_job.entry_script,
        git_url=db_job.git_url,
        git_branch=db_job.git_branch,
        git_commit=db_job.git_commit,
        manual_files=db_job.manual_files or {},
        model_files=db_job.model_files or {},
        data_files=db_job.data_files or {},
        environment_vars=db_job.environment_vars or {},
        python_packages=db_job.python_packages or [],
        docker_image=db_job.docker_image,
        status=JobStatus(db_job.status),
        cluster_job_id=db_job.cluster_job_id,
        cluster_path=db_job.cluster_path,
        submitted_at=db_job.submitted_at,
        started_at=db_job.started_at,
        completed_at=db_job.completed_at,
        current_epoch=db_job.current_epoch,
        total_epochs=db_job.total_epochs,
        current_loss=db_job.current_loss,
        current_accuracy=db_job.current_accuracy,
        memory_usage_mb=db_job.memory_usage_mb,
        gpu_usage_percent=db_job.gpu_usage_percent,
        error_message=db_job.error_message,
        exit_code=db_job.exit_code,
        final_metrics=db_job.final_metrics or {}
    )