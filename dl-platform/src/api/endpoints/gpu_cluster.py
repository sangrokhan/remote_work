"""
GPU Cluster API Endpoints

REST API endpoints for managing GPU cluster jobs including submission,
monitoring, result collection, and job management operations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path

from ...ml.cluster.orchestrator import GPUJobOrchestrator as JobOrchestrator
from ...ml.cluster.monitor import JobMonitor
from ...ml.cluster.collector import ResultCollector
from ...ml.cluster.connection import ClusterConnection
from ...ml.cluster.models import (
    GPUJobRequest,
    GPUJob,
    JobStatus,
    JobProgress,
    JobResults
)
from ...ml.cluster.exceptions import (
    ClusterError,
    JobNotFoundError,
    ValidationError,
    ResourceUnavailableError
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/gpu", tags=["GPU Cluster"])


# Dependency injection for services
async def get_cluster_connection():
    """클러스터 연결 의존성"""
    from ...core.config import get_cluster_config
    
    config = get_cluster_config()
    connection = ClusterConnection(**config)
    try:
        yield connection
    finally:
        await connection.close()


async def get_orchestrator(connection: ClusterConnection = Depends(get_cluster_connection)):
    """오케스트레이터 의존성"""
    return JobOrchestrator(connection)


async def get_monitor(connection: ClusterConnection = Depends(get_cluster_connection)):
    """모니터 의존성"""
    return JobMonitor(connection)


async def get_collector(connection: ClusterConnection = Depends(get_cluster_connection)):
    """수집기 의존성"""
    from ...core.config import settings
    return ResultCollector(connection, settings.storage.results_path)


# Job Management Endpoints

@router.post("/jobs", response_model=GPUJob)
async def submit_job(
    request: GPUJobRequest,
    orchestrator: JobOrchestrator = Depends(get_orchestrator),
    user_id: str = "default_user"  # TODO: 인증에서 사용자 ID 추출
):
    """GPU 클러스터에 새 작업 제출"""
    try:
        job = await orchestrator.submit_job(request, user_id)
        logger.info(f"Job submitted successfully: {job.job_id}")
        return job
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ResourceUnavailableError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=GPUJob)
async def get_job(
    job_id: str,
    orchestrator: JobOrchestrator = Depends(get_orchestrator)
):
    """작업 정보 조회"""
    try:
        job = await orchestrator.get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        return job
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=List[GPUJob])
async def list_jobs(
    status: Optional[JobStatus] = None,
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    orchestrator: JobOrchestrator = Depends(get_orchestrator)
):
    """작업 목록 조회"""
    try:
        jobs = await orchestrator.list_jobs(
            status=status,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        return jobs
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    orchestrator: JobOrchestrator = Depends(get_orchestrator)
):
    """작업 취소"""
    try:
        success = await orchestrator.cancel_job(job_id)
        if not success:
            raise JobNotFoundError(job_id)
        return {"message": f"Job {job_id} cancelled successfully"}
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Job Monitoring Endpoints

@router.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    monitor: JobMonitor = Depends(get_monitor),
    orchestrator: JobOrchestrator = Depends(get_orchestrator)
):
    """작업 상태 조회"""
    try:
        job = await orchestrator.get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        
        status = await monitor.check_job_status(job)
        return status
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/progress", response_model=Optional[JobProgress])
async def get_job_progress(
    job_id: str,
    monitor: JobMonitor = Depends(get_monitor),
    orchestrator: JobOrchestrator = Depends(get_orchestrator)
):
    """작업 진행 상황 조회"""
    try:
        job = await orchestrator.get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        
        progress = await monitor.get_job_progress(job)
        return progress
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    log_type: str = "training",
    lines: int = 100,
    monitor: JobMonitor = Depends(get_monitor),
    orchestrator: JobOrchestrator = Depends(get_orchestrator)
):
    """작업 로그 조회"""
    try:
        job = await orchestrator.get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        
        logs = await monitor.get_job_logs(job, log_type, lines)
        return {"job_id": job_id, "log_type": log_type, "logs": logs}
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Result Collection Endpoints

@router.post("/jobs/{job_id}/collect")
async def collect_job_results(
    job_id: str,
    background_tasks: BackgroundTasks,
    cleanup_remote: bool = False,
    collector: ResultCollector = Depends(get_collector),
    orchestrator: JobOrchestrator = Depends(get_orchestrator)
):
    """작업 결과 수집"""
    try:
        job = await orchestrator.get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        
        # 백그라운드에서 결과 수집 실행
        background_tasks.add_task(
            _collect_results_background,
            collector,
            job,
            cleanup_remote
        )
        
        return {"message": f"Result collection started for job {job_id}"}
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/results", response_model=Optional[JobResults])
async def get_job_results(
    job_id: str,
    collector: ResultCollector = Depends(get_collector)
):
    """수집된 작업 결과 조회"""
    try:
        summary = await collector.get_result_summary(job_id)
        if not summary:
            raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")
        
        # JobResults 객체로 변환
        results = JobResults(
            job_id=job_id,
            model_files=summary.get('file_counts', {}).get('models', 0),
            log_files=summary.get('file_counts', {}).get('logs', 0),
            output_files=summary.get('file_counts', {}).get('outputs', 0),
            final_metrics=summary.get('final_metrics', {}),
            total_size_mb=summary.get('total_size_mb', 0.0)
        )
        
        return results
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/download")
async def download_job_results(
    job_id: str,
    format: str = "tar.gz",
    collector: ResultCollector = Depends(get_collector)
):
    """작업 결과 다운로드"""
    try:
        # 결과 존재 확인
        summary = await collector.get_result_summary(job_id)
        if not summary:
            raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")
        
        # 아카이브 생성
        archive_path = await collector.archive_results(job_id, format)
        
        # 파일 다운로드 응답
        return FileResponse(
            path=archive_path,
            filename=f"{job_id}_results.{format}",
            media_type="application/octet-stream"
        )
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/export/{export_format}")
async def export_job_results(
    job_id: str,
    export_format: str,
    collector: ResultCollector = Depends(get_collector)
):
    """작업 결과를 특정 형식으로 내보내기"""
    try:
        if export_format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Supported formats: json, csv")
        
        # 임시 파일 경로 생성
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{export_format}", delete=False) as tmp:
            export_path = tmp.name
        
        # 결과 내보내기
        result_path = await collector.export_results(job_id, export_path, export_format)
        
        # 파일 다운로드 응답
        media_type = "application/json" if export_format == "json" else "text/csv"
        return FileResponse(
            path=result_path,
            filename=f"{job_id}_results.{export_format}",
            media_type=media_type
        )
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Job Comparison and Analytics

@router.post("/jobs/compare")
async def compare_jobs(
    job_ids: List[str],
    collector: ResultCollector = Depends(get_collector)
):
    """여러 작업 결과 비교"""
    try:
        if len(job_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 job IDs required for comparison")
        
        comparison = await collector.compare_results(job_ids)
        return comparison
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Storage Management Endpoints

@router.get("/storage/usage")
async def get_storage_usage(
    collector: ResultCollector = Depends(get_collector)
):
    """스토리지 사용량 조회"""
    try:
        usage = await collector.get_storage_usage()
        return usage
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/results")
async def list_available_results(
    collector: ResultCollector = Depends(get_collector)
):
    """사용 가능한 결과 목록"""
    try:
        results = await collector.list_available_results()
        return {"results": results, "count": len(results)}
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/storage/cleanup")
async def cleanup_expired_results(
    background_tasks: BackgroundTasks,
    retention_days: int = 30,
    collector: ResultCollector = Depends(get_collector)
):
    """만료된 결과 정리"""
    try:
        # 백그라운드에서 정리 작업 실행
        background_tasks.add_task(
            _cleanup_results_background,
            collector,
            retention_days
        )
        
        return {"message": f"Cleanup started for results older than {retention_days} days"}
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


# File Upload Endpoints

@router.post("/jobs/{job_id}/upload/{file_type}")
async def upload_job_file(
    job_id: str,
    file_type: str,
    file: UploadFile = File(...),
    orchestrator: JobOrchestrator = Depends(get_orchestrator)
):
    """작업에 파일 업로드 (데이터, 모델 등)"""
    try:
        if file_type not in ["data", "model", "config"]:
            raise HTTPException(status_code=400, detail="Supported file types: data, model, config")
        
        job = await orchestrator.get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        
        # 파일 업로드 처리
        upload_result = await orchestrator.upload_additional_file(job, file, file_type)
        
        return {
            "job_id": job_id,
            "file_type": file_type,
            "filename": file.filename,
            "uploaded_path": upload_result,
            "message": "File uploaded successfully"
        }
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health and Status Endpoints

@router.get("/health")
async def cluster_health(
    connection: ClusterConnection = Depends(get_cluster_connection)
):
    """클러스터 연결 상태 확인"""
    try:
        is_connected = await connection.test_connection()
        return {
            "cluster_connected": is_connected,
            "timestamp": datetime.now().isoformat(),
            "status": "healthy" if is_connected else "unhealthy"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "cluster_connected": False,
            "timestamp": datetime.now().isoformat(),
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/cluster/info")
async def get_cluster_info(
    connection: ClusterConnection = Depends(get_cluster_connection)
):
    """클러스터 정보 조회"""
    try:
        # 클러스터 기본 정보 수집
        info = await connection.get_cluster_info()
        return info
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background Tasks

async def _collect_results_background(
    collector: ResultCollector,
    job: GPUJob,
    cleanup_remote: bool
):
    """백그라운드 결과 수집 작업"""
    try:
        await collector.collect_results(job, cleanup_remote)
        logger.info(f"Background result collection completed for job {job.job_id}")
    except Exception as e:
        logger.error(f"Background result collection failed for job {job.job_id}: {e}")


async def _cleanup_results_background(
    collector: ResultCollector,
    retention_days: int
):
    """백그라운드 결과 정리 작업"""
    try:
        cleaned_jobs = await collector.cleanup_expired_results(retention_days)
        logger.info(f"Background cleanup completed, cleaned {len(cleaned_jobs)} jobs")
    except Exception as e:
        logger.error(f"Background cleanup failed: {e}")


# Error Handlers (추가 설정 필요)

@router.get("/jobs/{job_id}/debug")
async def debug_job(
    job_id: str,
    monitor: JobMonitor = Depends(get_monitor),
    orchestrator: JobOrchestrator = Depends(get_orchestrator)
):
    """작업 디버깅 정보"""
    try:
        job = await orchestrator.get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        
        debug_info = await monitor.get_debug_info(job)
        return debug_info
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ClusterError as e:
        raise HTTPException(status_code=500, detail=str(e))