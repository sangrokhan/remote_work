"""
GPU Job Orchestrator Service

Manages the complete lifecycle of GPU cluster jobs including submission,
file transfer, execution, and coordination with monitoring services.
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import shlex

from .connection import ClusterConnection
from .models import (
    GPUJobRequest,
    GPUJob,
    JobStatus,
    CodeSource,
    ClusterConfig
)
from .exceptions import (
    JobExecutionError,
    FileTransferError,
    ValidationError,
    ConfigurationError,
    ResourceUnavailableError
)

logger = logging.getLogger(__name__)


class GPUJobOrchestrator:
    """GPU 작업 조정 서비스"""
    
    def __init__(self, cluster_config: ClusterConfig, job_storage_service=None):
        self.config = cluster_config
        self.connection = ClusterConnection(cluster_config)
        self.job_storage = job_storage_service
        
    async def submit_job(self, request: GPUJobRequest, user_id: str) -> GPUJob:
        """작업 제출 및 실행"""
        job_id = str(uuid.uuid4())
        job_path = f"{self.config.base_path}/{job_id}"
        
        logger.info(f"Submitting job {job_id}: {request.job_name}")
        
        try:
            # 1. 연결 확인
            await self.connection.connect()
            
            # 2. 리소스 가용성 사전 확인
            await self._validate_resources(request)
            
            # 3. 작업 디렉토리 생성
            await self._create_job_directories(job_path)
            
            # 4. Job 객체 생성
            job = GPUJob(
                job_id=job_id,
                job_name=request.job_name,
                user_id=user_id,
                cluster_path=job_path,
                gpu_count=request.gpu_count,
                original_request=request,
                status=JobStatus.PREPARING
            )
            
            # 5. 작업 정보 저장
            if self.job_storage:
                await self.job_storage.save_job(job)
            
            # 6. 파일 전송
            await self._transfer_files(request, job_path)
            
            # 7. 코드 준비
            await self._prepare_code(request, job_path)
            
            # 8. 실행 스크립트 생성
            script_path = await self._create_execution_script(request, job_path)
            
            # 9. 상태 업데이트: QUEUED
            job.status = JobStatus.QUEUED
            job.queued_at = datetime.now()
            if self.job_storage:
                await self.job_storage.update_job(job)
            
            # 10. 프로세스 실행
            process_id = await self._execute_job(request, script_path)
            
            # 11. 최종 상태 업데이트
            job.process_id = process_id
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            if self.job_storage:
                await self.job_storage.update_job(job)
            
            logger.info(f"Job {job_id} submitted successfully with process ID {process_id}")
            return job
            
        except Exception as e:
            # 실패 시 정리
            logger.error(f"Failed to submit job {job_id}: {str(e)}")
            
            try:
                await self.connection.cleanup_job_files(job_path, keep_results=False)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup after job submission failure: {cleanup_error}")
            
            # 작업 상태를 실패로 업데이트
            if 'job' in locals() and self.job_storage:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                await self.job_storage.update_job(job)
            
            raise
    
    async def cancel_job(self, job_id: str) -> bool:
        """작업 취소"""
        try:
            # 작업 정보 조회
            if self.job_storage:
                job = await self.job_storage.get_job(job_id)
            else:
                raise JobExecutionError(f"Cannot cancel job without storage service")
            
            if not job:
                raise JobExecutionError(f"Job {job_id} not found")
            
            if job.status not in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]:
                raise JobExecutionError(f"Cannot cancel job in status: {job.status}")
            
            # 프로세스 종료
            if job.process_id:
                await self._kill_process(job.process_id)
            
            # 상태 업데이트
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            job.error_message = "Job cancelled by user"
            
            if self.job_storage:
                await self.job_storage.update_job(job)
            
            logger.info(f"Job {job_id} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {str(e)}")
            raise JobExecutionError(f"Failed to cancel job: {str(e)}", job_id=job_id)
    
    async def _validate_resources(self, request: GPUJobRequest):
        """리소스 요구사항 검증"""
        # GPU 가용성 확인 (실제 클러스터 명령어에 따라 조정 필요)
        stdout, stderr, exit_code = await self.connection.execute_command("nvidia-smi --list-gpus | wc -l")
        
        if exit_code != 0:
            raise ResourceUnavailableError("Cannot query GPU availability")
        
        try:
            available_gpus = int(stdout.strip())
            if request.gpu_count > available_gpus:
                raise ResourceUnavailableError(
                    f"Requested {request.gpu_count} GPUs but only {available_gpus} available",
                    requested_gpus=request.gpu_count,
                    available_gpus=available_gpus
                )
        except ValueError:
            logger.warning("Could not parse GPU count from cluster")
        
        # 디스크 공간 확인
        if request.dataset_path:
            dataset_size = Path(request.dataset_path).stat().st_size / (1024**3)  # GB
            required_space = dataset_size * 2  # 압축 해제 고려
            await self.connection.ensure_disk_space(self.config.base_path, required_space)
    
    async def _create_job_directories(self, job_path: str):
        """작업 디렉토리 생성"""
        directories = [
            job_path,
            f"{job_path}/code",
            f"{job_path}/data", 
            f"{job_path}/models",
            f"{job_path}/outputs",
            f"{job_path}/logs"
        ]
        
        for directory in directories:
            cmd = f"mkdir -p {directory}"
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            if exit_code != 0:
                raise JobExecutionError(f"Failed to create directory {directory}: {stderr}")
        
        logger.debug(f"Created job directories: {job_path}")
    
    async def _transfer_files(self, request: GPUJobRequest, job_path: str):
        """필요한 파일들을 클러스터로 전송"""
        
        # 모델 파일 전송
        if request.model_file_path:
            local_model_path = request.model_file_path
            model_filename = os.path.basename(local_model_path)
            remote_model_path = f"{job_path}/models/{model_filename}"
            
            logger.info(f"Uploading model file: {model_filename}")
            await self.connection.upload_file(local_model_path, remote_model_path)
        
        # 데이터셋 전송
        if request.dataset_path:
            local_dataset_path = request.dataset_path
            dataset_filename = os.path.basename(local_dataset_path)
            remote_dataset_path = f"{job_path}/data/{dataset_filename}"
            
            logger.info(f"Uploading dataset: {dataset_filename}")
            await self.connection.upload_file(local_dataset_path, remote_dataset_path)
            
            # 압축 파일인 경우 자동 압축 해제
            if dataset_filename.endswith(('.tar.gz', '.tgz', '.tar', '.zip')):
                await self._extract_dataset(remote_dataset_path, f"{job_path}/data")
    
    async def _extract_dataset(self, archive_path: str, extract_dir: str):
        """데이터셋 압축 해제"""
        if archive_path.endswith(('.tar.gz', '.tgz')):
            cmd = f"cd {extract_dir} && tar -xzf {os.path.basename(archive_path)}"
        elif archive_path.endswith('.tar'):
            cmd = f"cd {extract_dir} && tar -xf {os.path.basename(archive_path)}"
        elif archive_path.endswith('.zip'):
            cmd = f"cd {extract_dir} && unzip {os.path.basename(archive_path)}"
        else:
            return  # 압축 파일이 아님
        
        stdout, stderr, exit_code = await self.connection.execute_command(cmd)
        if exit_code != 0:
            logger.warning(f"Failed to extract dataset: {stderr}")
        else:
            logger.info(f"Dataset extracted successfully: {archive_path}")
    
    async def _prepare_code(self, request: GPUJobRequest, job_path: str):
        """코드 준비"""
        code_dir = f"{job_path}/code"
        
        if request.code_source == CodeSource.GIT:
            # Git 저장소 클론
            git_url = request.code_path
            # 보안을 위한 URL 검증
            if not self._is_safe_git_url(git_url):
                raise SecurityError(f"Unsafe Git URL: {git_url}")
            
            cmd = f"cd {code_dir} && git clone {shlex.quote(git_url)} ."
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            if exit_code != 0:
                raise JobExecutionError(f"Failed to clone repository: {stderr}")
            
            logger.info(f"Git repository cloned: {git_url}")
            
        elif request.code_source == CodeSource.LOCAL:
            # 로컬 코드 업로드
            local_code_path = request.code_path
            if not os.path.exists(local_code_path):
                raise FileTransferError(f"Local code path not found: {local_code_path}")
            
            if os.path.isdir(local_code_path):
                # 디렉토리 전체 업로드
                await self.connection.upload_directory(local_code_path, code_dir)
            else:
                # 단일 파일 업로드
                filename = os.path.basename(local_code_path)
                remote_file_path = f"{code_dir}/{filename}"
                await self.connection.upload_file(local_code_path, remote_file_path)
            
            logger.info(f"Local code uploaded: {local_code_path}")
            
        elif request.code_source == CodeSource.EXISTING:
            # 기존 클러스터 코드 복사
            existing_path = request.code_path
            cmd = f"cp -r {shlex.quote(existing_path)}/* {code_dir}/"
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            if exit_code != 0:
                raise JobExecutionError(f"Failed to copy existing code: {stderr}")
            
            logger.info(f"Existing code copied: {existing_path}")
        
        # 코드 파일 권한 설정
        await self.connection.execute_command(f"chmod -R 755 {code_dir}")
    
    def _is_safe_git_url(self, url: str) -> bool:
        """Git URL 보안 검증"""
        # 허용된 프로토콜만 사용
        allowed_protocols = ['https://', 'git@']
        if not any(url.startswith(protocol) for protocol in allowed_protocols):
            return False
        
        # 로컬 파일 시스템 접근 방지
        dangerous_patterns = ['file://', 'localhost', '127.0.0.1', '../']
        if any(pattern in url for pattern in dangerous_patterns):
            return False
        
        return True
    
    async def _create_execution_script(self, request: GPUJobRequest, job_path: str) -> str:
        """실행 스크립트 생성"""
        script_content = self._generate_script_content(request, job_path)
        
        # 로컬에 임시 스크립트 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp_file:
            tmp_file.write(script_content)
            local_script_path = tmp_file.name
        
        # 원격으로 스크립트 업로드
        remote_script_path = f"{job_path}/run.sh"
        await self.connection.upload_file(local_script_path, remote_script_path)
        
        # 실행 권한 부여
        await self.connection.execute_command(f"chmod +x {remote_script_path}")
        
        # 임시 파일 정리
        os.unlink(local_script_path)
        
        logger.debug(f"Execution script created: {remote_script_path}")
        return remote_script_path
    
    def _generate_script_content(self, request: GPUJobRequest, job_path: str) -> str:
        """실행 스크립트 내용 생성"""
        
        # 환경 변수 문자열 생성
        env_vars = []
        for key, value in request.environment_vars.items():
            # 보안을 위한 키-값 검증
            safe_key = shlex.quote(key)
            safe_value = shlex.quote(value)
            env_vars.append(f"export {safe_key}={safe_value}")
        
        # 스크립트 인자 문자열 생성
        script_args = []
        for key, value in request.script_args.items():
            safe_key = shlex.quote(key)
            safe_value = shlex.quote(value)
            script_args.append(f"--{safe_key} {safe_value}")
        
        # GPU 개수 설정
        gpu_devices = ",".join(str(i) for i in range(request.gpu_count))
        
        script_content = f'''#!/bin/bash
#
# Auto-generated execution script for GPU cluster
# Job ID: {job_path.split('/')[-1]}
# Job Name: {request.job_name}
# GPU Count: {request.gpu_count}
# Generated at: {datetime.now().isoformat()}
#

set -e  # Exit on any error

# 작업 시작 로깅
echo "Job started at $(date)" | tee {job_path}/logs/start_time.log
echo "Job ID: {job_path.split('/')[-1]}" | tee -a {job_path}/logs/start_time.log
echo "Job Name: {request.job_name}" | tee -a {job_path}/logs/start_time.log

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES={gpu_devices}
export JOB_ID={job_path.split('/')[-1]}
export JOB_NAME="{request.job_name}"
export JOB_PATH={job_path}
export OUTPUT_DIR={job_path}/outputs
export LOG_DIR={job_path}/logs
export DATA_DIR={job_path}/data
export MODEL_DIR={job_path}/models

# 사용자 정의 환경 변수
{chr(10).join(env_vars)}

# 작업 디렉토리로 이동
cd {job_path}/code

# Python 환경 확인 및 의존성 설치
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..." | tee -a $LOG_DIR/setup.log
    pip install -r requirements.txt 2>&1 | tee -a $LOG_DIR/setup.log
fi

# GPU 정보 로깅
echo "=== GPU Information ===" | tee -a $LOG_DIR/gpu_info.log
nvidia-smi 2>&1 | tee -a $LOG_DIR/gpu_info.log

# 실행 시작 시간 기록
echo "Training started at $(date)" | tee -a $LOG_DIR/training.log

# 메인 스크립트 실행
echo "Executing: python {request.entry_script} {' '.join(script_args)}" | tee -a $LOG_DIR/training.log

python {shlex.quote(request.entry_script)} \\
    --output_dir $OUTPUT_DIR \\
    --log_dir $LOG_DIR \\
    {' '.join(script_args)} \\
    2>&1 | tee -a $LOG_DIR/training.log

# 실행 결과 처리
EXIT_CODE=$?
echo "Training completed at $(date) with exit code $EXIT_CODE" | tee -a $LOG_DIR/training.log

if [ $EXIT_CODE -eq 0 ]; then
    echo "success" > $OUTPUT_DIR/status.txt
    echo "Job completed successfully at $(date)" > $OUTPUT_DIR/SUCCESS
    echo "SUCCESS" | tee -a $LOG_DIR/final_status.log
else
    echo "failed" > $OUTPUT_DIR/status.txt
    echo "Job failed at $(date) with exit code $EXIT_CODE" > $OUTPUT_DIR/FAILED
    echo "FAILED" | tee -a $LOG_DIR/final_status.log
    exit $EXIT_CODE
fi
'''
        
        return script_content
    
    async def _execute_job(self, request: GPUJobRequest, script_path: str) -> str:
        """작업 실행 및 프로세스 ID 반환"""
        
        # GPU 작업 실행 명령 생성
        # 실제 클러스터의 명령어에 맞게 수정 필요
        gpu_run_cmd = self._build_gpu_command(request)
        
        # 백그라운드 실행 및 PID 캡처
        full_cmd = f"nohup {gpu_run_cmd} {shlex.quote(script_path)} > /dev/null 2>&1 & echo $!"
        
        logger.info(f"Executing GPU job with command: {gpu_run_cmd}")
        stdout, stderr, exit_code = await self.connection.execute_command(full_cmd)
        
        if exit_code != 0:
            raise JobExecutionError(f"Failed to start GPU job: {stderr}")
        
        process_id = stdout.strip()
        if not process_id:
            raise JobExecutionError("Failed to get process ID from job execution")
        
        logger.info(f"GPU job started with process ID: {process_id}")
        return process_id
    
    def _build_gpu_command(self, request: GPUJobRequest) -> str:
        """GPU 실행 명령 생성"""
        # 이 부분은 실제 클러스터의 명령어에 맞게 수정해야 함
        # 예시: Slurm, PBS, 또는 커스텀 스케줄러
        
        cmd_parts = ["gpu-run"]  # 기본 명령어 (실제 명령어로 교체)
        
        # GPU 개수 지정
        cmd_parts.append(f"--gpus {request.gpu_count}")
        
        # 메모리 지정
        if request.memory_gb:
            cmd_parts.append(f"--memory {request.memory_gb}G")
        
        # 타임아웃 설정
        if request.timeout_hours:
            cmd_parts.append(f"--time {request.timeout_hours}:00:00")
        
        return " ".join(cmd_parts)
    
    async def _kill_process(self, process_id: str):
        """프로세스 강제 종료"""
        # SIGTERM으로 정상 종료 시도
        cmd = f"kill {process_id}"
        stdout, stderr, exit_code = await self.connection.execute_command(cmd)
        
        if exit_code != 0:
            logger.warning(f"SIGTERM failed for process {process_id}, trying SIGKILL")
            # SIGKILL로 강제 종료
            cmd = f"kill -9 {process_id}"
            await self.connection.execute_command(cmd)
        
        logger.info(f"Process {process_id} terminated")
    
    async def retry_job(self, job_id: str) -> GPUJob:
        """작업 재시도"""
        if not self.job_storage:
            raise ConfigurationError("job_storage", "Job storage service not configured")
        
        # 기존 작업 정보 조회
        original_job = await self.job_storage.get_job(job_id)
        if not original_job:
            raise JobExecutionError(f"Original job {job_id} not found")
        
        # 재시도 가능한 상태인지 확인
        if original_job.status not in [JobStatus.FAILED, JobStatus.TIMEOUT, JobStatus.CANCELLED]:
            raise JobExecutionError(f"Cannot retry job in status: {original_job.status}")
        
        # 새로운 작업으로 재제출
        retry_request = original_job.original_request
        retry_request.job_name = f"{retry_request.job_name}_retry"
        
        logger.info(f"Retrying job {job_id} as new job")
        return await self.submit_job(retry_request, original_job.user_id)
    
    async def get_job_details(self, job_id: str) -> Optional[GPUJob]:
        """작업 상세 정보 조회"""
        if not self.job_storage:
            return None
        return await self.job_storage.get_job(job_id)
    
    async def list_user_jobs(
        self, 
        user_id: str, 
        status_filter: Optional[JobStatus] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[GPUJob]:
        """사용자 작업 목록 조회"""
        if not self.job_storage:
            return []
        return await self.job_storage.list_jobs(
            user_id=user_id,
            status=status_filter,
            limit=limit,
            offset=offset
        )
    
    async def cleanup_old_jobs(self, days_old: int = 7, dry_run: bool = True) -> List[str]:
        """오래된 작업 정리"""
        if not self.job_storage:
            return []
        
        # 오래된 완료/실패 작업 조회
        old_jobs = await self.job_storage.get_old_jobs(days_old)
        cleaned_jobs = []
        
        for job in old_jobs:
            try:
                if not dry_run:
                    # 클러스터에서 파일 삭제
                    await self.connection.cleanup_job_files(job.cluster_path, keep_results=False)
                    
                    # 데이터베이스에서 삭제 또는 아카이브 상태로 변경
                    await self.job_storage.archive_job(job.job_id)
                
                cleaned_jobs.append(job.job_id)
                logger.info(f"{'Would clean' if dry_run else 'Cleaned'} job {job.job_id}")
                
            except Exception as e:
                logger.error(f"Failed to clean job {job.job_id}: {e}")
        
        return cleaned_jobs
    
    async def get_cluster_status(self) -> dict:
        """클러스터 상태 정보 조회"""
        try:
            status_info = await self.connection.test_connection()
            
            # 활성 작업 수 조회
            if self.job_storage:
                active_jobs_count = await self.job_storage.count_active_jobs()
                status_info['active_jobs'] = active_jobs_count
            
            # 시스템 메트릭 추가
            system_metrics = await self.connection.get_system_metrics()
            status_info['system_metrics'] = system_metrics
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def estimate_queue_time(self, request: GPUJobRequest) -> Optional[int]:
        """대기 시간 추정 (초)"""
        if not self.job_storage:
            return None
        
        try:
            # 대기 중인 작업 수 조회
            queued_jobs = await self.job_storage.count_jobs_by_status(JobStatus.QUEUED)
            running_jobs = await self.job_storage.count_jobs_by_status(JobStatus.RUNNING)
            
            # 평균 작업 시간 조회
            avg_duration = await self.job_storage.get_average_job_duration()
            
            if avg_duration and queued_jobs >= 0:
                # 간단한 추정: (대기 작업 수) × (평균 실행 시간) / (가용 GPU 슬롯)
                # 실제로는 더 정교한 알고리즘 필요
                estimated_seconds = (queued_jobs * avg_duration) / max(1, request.gpu_count)
                return int(estimated_seconds)
            
        except Exception as e:
            logger.warning(f"Failed to estimate queue time: {e}")
        
        return None
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.connection.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.connection.disconnect()