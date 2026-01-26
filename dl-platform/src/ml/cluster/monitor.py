"""
GPU Job Monitoring Service

Monitors running GPU cluster jobs, tracks status changes, and collects real-time metrics.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, AsyncIterator, Tuple
import time

from .connection import ClusterConnection
from .models import (
    GPUJob,
    JobStatus,
    JobProgress,
    JobLogEntry
)
from .exceptions import (
    JobExecutionError,
    JobNotFoundError,
    CommandExecutionError
)

logger = logging.getLogger(__name__)


class JobMonitor:
    """GPU 작업 모니터링 서비스"""
    
    def __init__(self, connection: ClusterConnection, job_storage_service=None):
        self.connection = connection
        self.job_storage = job_storage_service
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        
    async def check_job_status(self, job: GPUJob) -> JobStatus:
        """작업 상태 확인"""
        if not job.process_id:
            return JobStatus.PENDING
        
        try:
            # 프로세스 존재 여부 확인
            is_running = await self._is_process_running(job.process_id)
            
            if is_running:
                return JobStatus.RUNNING
            else:
                # 프로세스가 종료됨 - 성공/실패 판단
                return await self._determine_completion_status(job)
                
        except Exception as e:
            logger.error(f"Failed to check job status for {job.job_id}: {e}")
            return job.status  # 기존 상태 유지
    
    async def _is_process_running(self, process_id: str) -> bool:
        """프로세스 실행 상태 확인"""
        cmd = f"ps -p {process_id} -o stat --no-headers"
        stdout, stderr, exit_code = await self.connection.execute_command(cmd)
        
        if exit_code != 0:
            return False  # 프로세스가 존재하지 않음
        
        # 프로세스 상태 파싱
        status = stdout.strip()
        # R: Running, S: Sleeping, D: Disk sleep, T: Stopped, Z: Zombie
        active_states = ['R', 'S', 'D']
        return any(state in status for state in active_states)
    
    async def _determine_completion_status(self, job: GPUJob) -> JobStatus:
        """완료된 작업의 성공/실패 판단"""
        try:
            # SUCCESS 마커 파일 확인
            success_file = f"{job.cluster_path}/outputs/SUCCESS"
            cmd = f"test -f {success_file} && echo 'SUCCESS' || echo 'NOT_FOUND'"
            stdout, _, _ = await self.connection.execute_command(cmd)
            
            if "SUCCESS" in stdout:
                return JobStatus.COMPLETED
            
            # FAILED 마커 파일 확인
            failed_file = f"{job.cluster_path}/outputs/FAILED"
            cmd = f"test -f {failed_file} && echo 'FAILED' || echo 'NOT_FOUND'"
            stdout, _, _ = await self.connection.execute_command(cmd)
            
            if "FAILED" in stdout:
                return JobStatus.FAILED
            
            # 마커 파일이 없으면 status.txt 확인
            status_file = f"{job.cluster_path}/outputs/status.txt"
            cmd = f"cat {status_file} 2>/dev/null || echo 'unknown'"
            stdout, _, _ = await self.connection.execute_command(cmd)
            
            status_content = stdout.strip().lower()
            if status_content == "success":
                return JobStatus.COMPLETED
            elif status_content == "failed":
                return JobStatus.FAILED
            
            # 타임아웃 확인
            if self._is_job_timed_out(job):
                return JobStatus.TIMEOUT
            
            # 기본적으로 실패로 간주
            return JobStatus.FAILED
            
        except Exception as e:
            logger.error(f"Failed to determine completion status for job {job.job_id}: {e}")
            return JobStatus.FAILED
    
    def _is_job_timed_out(self, job: GPUJob) -> bool:
        """작업 타임아웃 확인"""
        if not job.started_at or not job.original_request.timeout_hours:
            return False
        
        elapsed_time = datetime.now() - job.started_at
        timeout_duration = timedelta(hours=job.original_request.timeout_hours)
        
        return elapsed_time > timeout_duration
    
    async def get_job_logs(
        self, 
        job: GPUJob, 
        lines: int = 100,
        log_type: str = "training"
    ) -> List[str]:
        """작업 로그 조회"""
        log_file_map = {
            "training": f"{job.cluster_path}/logs/training.log",
            "setup": f"{job.cluster_path}/logs/setup.log",
            "error": f"{job.cluster_path}/logs/error.log",
            "gpu": f"{job.cluster_path}/logs/gpu_info.log"
        }
        
        log_file = log_file_map.get(log_type, log_file_map["training"])
        
        try:
            cmd = f"tail -n {lines} {log_file} 2>/dev/null || echo 'Log file not found'"
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            if "Log file not found" in stdout:
                return [f"Log file not found: {log_file}"]
            
            return stdout.strip().split('\n') if stdout.strip() else []
            
        except Exception as e:
            logger.error(f"Failed to get logs for job {job.job_id}: {e}")
            return [f"Error retrieving logs: {str(e)}"]
    
    async def stream_job_logs(self, job: GPUJob) -> AsyncIterator[str]:
        """실시간 로그 스트리밍"""
        log_file = f"{job.cluster_path}/logs/training.log"
        
        # tail -f 명령으로 실시간 로그 추적
        cmd = f"tail -f {log_file}"
        
        try:
            # SSH를 통한 스트리밍 연결
            ssh = self.connection._ssh_client
            if not ssh:
                raise ConnectionError("SSH connection not available")
            
            transport = ssh.get_transport()
            channel = transport.open_session()
            channel.exec_command(cmd)
            
            # 스트리밍 루프
            while True:
                # 논블로킹으로 데이터 읽기
                if channel.recv_ready():
                    data = channel.recv(1024).decode('utf-8')
                    for line in data.split('\n'):
                        if line.strip():
                            yield line.strip()
                
                # 채널이 닫혔는지 확인
                if channel.exit_status_ready():
                    break
                
                # 짧은 대기
                await asyncio.sleep(0.1)
                
            channel.close()
            
        except Exception as e:
            logger.error(f"Log streaming failed for job {job.job_id}: {e}")
            yield f"Log streaming error: {str(e)}"
    
    async def get_job_progress(self, job: GPUJob) -> Optional[JobProgress]:
        """작업 진행 상황 파싱"""
        try:
            # 최근 로그에서 진행 상황 파싱
            recent_logs = await self.get_job_logs(job, lines=50)
            progress = await self._parse_progress_from_logs(job.job_id, recent_logs)
            
            # GPU 사용률 조회
            gpu_metrics = await self._get_gpu_metrics(job)
            if gpu_metrics:
                progress.gpu_utilization = gpu_metrics.get('utilization')
                progress.memory_usage_gb = gpu_metrics.get('memory_usage_gb')
            
            return progress
            
        except Exception as e:
            logger.error(f"Failed to get progress for job {job.job_id}: {e}")
            return None
    
    async def _parse_progress_from_logs(self, job_id: str, log_lines: List[str]) -> JobProgress:
        """로그에서 진행 상황 파싱"""
        progress = JobProgress(job_id=job_id)
        
        # 일반적인 진행률 패턴들
        patterns = {
            'epoch': re.compile(r'Epoch\s+(\d+)/(\d+)', re.IGNORECASE),
            'step': re.compile(r'Step\s+(\d+)/(\d+)', re.IGNORECASE),
            'loss': re.compile(r'Loss[:=]\s*([\d.]+)', re.IGNORECASE),
            'accuracy': re.compile(r'Acc(?:uracy)?[:=]\s*([\d.]+)', re.IGNORECASE),
            'learning_rate': re.compile(r'(?:lr|learning.rate)[:=]\s*([\d.e-]+)', re.IGNORECASE),
        }
        
        for line in reversed(log_lines):  # 최신 로그부터 파싱
            # Epoch 정보
            if not progress.current_epoch:
                match = patterns['epoch'].search(line)
                if match:
                    progress.current_epoch = int(match.group(1))
                    progress.total_epochs = int(match.group(2))
            
            # Step 정보
            if not progress.current_step:
                match = patterns['step'].search(line)
                if match:
                    progress.current_step = int(match.group(1))
                    progress.total_steps = int(match.group(2))
            
            # Loss 정보
            if not progress.current_loss:
                match = patterns['loss'].search(line)
                if match:
                    progress.current_loss = float(match.group(1))
            
            # Accuracy 정보
            if not progress.current_accuracy:
                match = patterns['accuracy'].search(line)
                if match:
                    accuracy = float(match.group(1))
                    # 0-1 범위로 정규화 (100을 넘으면 퍼센트로 간주)
                    progress.current_accuracy = accuracy / 100 if accuracy > 1 else accuracy
            
            # Learning Rate 정보
            if not progress.learning_rate:
                match = patterns['learning_rate'].search(line)
                if match:
                    progress.learning_rate = float(match.group(1))
        
        # 예상 남은 시간 계산
        if progress.current_epoch and progress.total_epochs:
            progress.estimated_time_remaining = self._estimate_remaining_time(
                progress.current_epoch,
                progress.total_epochs,
                job_id
            )
        
        return progress
    
    async def _get_gpu_metrics(self, job: GPUJob) -> Optional[Dict[str, float]]:
        """GPU 메트릭 조회"""
        try:
            # GPU 사용률 조회
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            if exit_code != 0:
                return None
            
            # 첫 번째 GPU 정보 파싱 (멀티 GPU 환경에서는 평균 계산)
            lines = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
            if not lines:
                return None
            
            total_utilization = 0
            total_memory_used = 0
            total_memory_total = 0
            
            for line in lines[:job.gpu_count]:  # 할당된 GPU 수만큼만
                parts = line.split(',')
                if len(parts) >= 3:
                    utilization = float(parts[0].strip())
                    memory_used = float(parts[1].strip())
                    memory_total = float(parts[2].strip())
                    
                    total_utilization += utilization
                    total_memory_used += memory_used
                    total_memory_total += memory_total
            
            gpu_count = min(len(lines), job.gpu_count)
            return {
                'utilization': total_utilization / gpu_count,
                'memory_usage_gb': total_memory_used / 1024,  # MB to GB
                'memory_total_gb': total_memory_total / 1024,
                'memory_usage_percent': (total_memory_used / total_memory_total) * 100
            }
            
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
            return None
    
    def _estimate_remaining_time(self, current_epoch: int, total_epochs: int, job_id: str) -> Optional[int]:
        """남은 시간 추정"""
        try:
            if not self.job_storage:
                return None
            
            # 작업 시작 시간 기반 계산
            # 더 정교한 알고리즘은 과거 에포크별 시간을 추적해야 함
            # 현재는 단순한 선형 추정 사용
            
            if current_epoch <= 0:
                return None
            
            # 간단한 선형 추정
            progress_ratio = current_epoch / total_epochs
            if progress_ratio <= 0:
                return None
            
            # 실제 구현에서는 job_storage에서 시작 시간을 가져와야 함
            # 현재는 추정값 반환
            estimated_total_time = 3600  # 1시간 가정
            elapsed_ratio = progress_ratio
            remaining_ratio = 1 - elapsed_ratio
            
            return int((remaining_ratio / elapsed_ratio) * (estimated_total_time * elapsed_ratio))
            
        except Exception as e:
            logger.warning(f"Failed to estimate remaining time: {e}")
            return None
    
    async def get_detailed_job_info(self, job: GPUJob) -> Dict:
        """작업 상세 정보 조회"""
        try:
            info = {
                'job_id': job.job_id,
                'status': await self.check_job_status(job),
                'progress': await self.get_job_progress(job),
                'system_metrics': await self._get_job_system_metrics(job),
                'files_info': await self._get_job_files_info(job),
                'last_updated': datetime.now()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get detailed info for job {job.job_id}: {e}")
            return {
                'job_id': job.job_id,
                'status': JobStatus.FAILED,
                'error': str(e),
                'last_updated': datetime.now()
            }
    
    async def _get_job_system_metrics(self, job: GPUJob) -> Dict:
        """작업별 시스템 메트릭 조회"""
        if not job.process_id:
            return {}
        
        try:
            metrics = {}
            
            # 프로세스별 CPU/메모리 사용량
            cmd = f"ps -p {job.process_id} -o %cpu,%mem,vsz,rss --no-headers"
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            if exit_code == 0:
                parts = stdout.strip().split()
                if len(parts) >= 4:
                    metrics.update({
                        'cpu_percent': float(parts[0]),
                        'memory_percent': float(parts[1]),
                        'virtual_memory_kb': int(parts[2]),
                        'resident_memory_kb': int(parts[3])
                    })
            
            # 프로세스 실행 시간
            cmd = f"ps -p {job.process_id} -o etime --no-headers"
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            if exit_code == 0:
                metrics['runtime'] = stdout.strip()
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to get system metrics for job {job.job_id}: {e}")
            return {}
    
    async def _get_job_files_info(self, job: GPUJob) -> Dict:
        """작업 파일 정보 조회"""
        try:
            info = {}
            
            # 출력 디렉토리 크기
            cmd = f"du -sh {job.cluster_path}/outputs 2>/dev/null | cut -f1"
            stdout, _, exit_code = await self.connection.execute_command(cmd)
            if exit_code == 0:
                info['output_size'] = stdout.strip()
            
            # 로그 파일 수
            cmd = f"find {job.cluster_path}/logs -type f | wc -l"
            stdout, _, exit_code = await self.connection.execute_command(cmd)
            if exit_code == 0:
                info['log_file_count'] = int(stdout.strip())
            
            # 출력 파일 수
            cmd = f"find {job.cluster_path}/outputs -type f | wc -l"
            stdout, _, exit_code = await self.connection.execute_command(cmd)
            if exit_code == 0:
                info['output_file_count'] = int(stdout.strip())
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get file info for job {job.job_id}: {e}")
            return {}
    
    async def start_monitoring(self, job: GPUJob, update_interval: int = 30):
        """작업 모니터링 시작"""
        if job.job_id in self._monitoring_tasks:
            logger.warning(f"Monitoring already active for job {job.job_id}")
            return
        
        # 비동기 모니터링 태스크 생성
        task = asyncio.create_task(
            self._monitoring_loop(job, update_interval)
        )
        self._monitoring_tasks[job.job_id] = task
        
        logger.info(f"Started monitoring for job {job.job_id}")
    
    async def stop_monitoring(self, job_id: str):
        """작업 모니터링 중지"""
        if job_id in self._monitoring_tasks:
            task = self._monitoring_tasks[job_id]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self._monitoring_tasks[job_id]
            logger.info(f"Stopped monitoring for job {job_id}")
    
    async def _monitoring_loop(self, job: GPUJob, update_interval: int):
        """모니터링 루프"""
        logger.info(f"Starting monitoring loop for job {job.job_id}")
        
        try:
            while True:
                # 상태 확인
                current_status = await self.check_job_status(job)
                
                # 상태가 변경된 경우 업데이트
                if current_status != job.status:
                    old_status = job.status
                    job.status = current_status
                    
                    # 완료 시간 기록
                    if current_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.TIMEOUT]:
                        job.completed_at = datetime.now()
                    
                    # 데이터베이스 업데이트
                    if self.job_storage:
                        await self.job_storage.update_job(job)
                    
                    logger.info(f"Job {job.job_id} status changed: {old_status} -> {current_status}")
                    
                    # 완료된 경우 모니터링 종료
                    if job.is_finished:
                        break
                
                # 진행 상황 업데이트
                if current_status == JobStatus.RUNNING:
                    progress = await self.get_job_progress(job)
                    if progress and self.job_storage:
                        await self.job_storage.update_job_progress(job.job_id, progress)
                
                # 대기
                await asyncio.sleep(update_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for job {job.job_id}")
            raise
        except Exception as e:
            logger.error(f"Monitoring error for job {job.job_id}: {e}")
            
            # 에러 발생 시 작업을 실패로 마킹
            if self.job_storage:
                job.status = JobStatus.FAILED
                job.error_message = f"Monitoring error: {str(e)}"
                job.completed_at = datetime.now()
                await self.job_storage.update_job(job)
        
        finally:
            # 모니터링 태스크 정리
            if job.job_id in self._monitoring_tasks:
                del self._monitoring_tasks[job.job_id]
            
            logger.info(f"Monitoring ended for job {job.job_id}")
    
    async def get_job_resource_usage(self, job: GPUJob) -> Dict:
        """작업 리소스 사용량 조회"""
        if not job.process_id:
            return {}
        
        try:
            usage = {}
            
            # GPU 메트릭
            gpu_metrics = await self._get_gpu_metrics(job)
            if gpu_metrics:
                usage['gpu'] = gpu_metrics
            
            # 시스템 메트릭
            system_metrics = await self._get_job_system_metrics(job)
            if system_metrics:
                usage['system'] = system_metrics
            
            # 네트워크 I/O (필요시)
            network_metrics = await self._get_network_metrics(job)
            if network_metrics:
                usage['network'] = network_metrics
            
            return usage
            
        except Exception as e:
            logger.error(f"Failed to get resource usage for job {job.job_id}: {e}")
            return {}
    
    async def _get_network_metrics(self, job: GPUJob) -> Optional[Dict]:
        """네트워크 메트릭 조회"""
        try:
            # 프로세스별 네트워크 I/O 조회 (Linux)
            cmd = f"cat /proc/{job.process_id}/io 2>/dev/null"
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            if exit_code != 0:
                return None
            
            io_stats = {}
            for line in stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    io_stats[key.strip()] = int(value.strip())
            
            return {
                'read_bytes': io_stats.get('read_bytes', 0),
                'write_bytes': io_stats.get('write_bytes', 0),
                'read_chars': io_stats.get('rchar', 0),
                'write_chars': io_stats.get('wchar', 0)
            }
            
        except Exception as e:
            logger.debug(f"Could not get network metrics: {e}")
            return None
    
    async def get_monitoring_summary(self) -> Dict:
        """모니터링 중인 작업 요약"""
        active_jobs = len(self._monitoring_tasks)
        
        summary = {
            'active_monitoring_jobs': active_jobs,
            'monitoring_tasks': list(self._monitoring_tasks.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.job_storage:
            # 데이터베이스에서 추가 통계 조회
            try:
                stats = await self.job_storage.get_job_statistics()
                summary['job_statistics'] = stats
            except Exception as e:
                logger.warning(f"Failed to get job statistics: {e}")
        
        return summary
    
    async def force_status_update(self, job_id: str) -> JobStatus:
        """강제 상태 업데이트"""
        if not self.job_storage:
            raise ConfigurationError("job_storage", "Job storage service not configured")
        
        job = await self.job_storage.get_job(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        
        # 강제로 상태 확인
        current_status = await self.check_job_status(job)
        
        if current_status != job.status:
            job.status = current_status
            if job.is_finished and not job.completed_at:
                job.completed_at = datetime.now()
            
            await self.job_storage.update_job(job)
            logger.info(f"Force updated job {job_id} status to {current_status}")
        
        return current_status
    
    async def cleanup_monitoring(self):
        """모든 모니터링 태스크 정리"""
        tasks_to_cancel = list(self._monitoring_tasks.values())
        job_ids = list(self._monitoring_tasks.keys())
        
        # 모든 태스크 취소
        for task in tasks_to_cancel:
            task.cancel()
        
        # 취소 완료 대기
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # 딕셔너리 정리
        self._monitoring_tasks.clear()
        
        logger.info(f"Cleaned up monitoring for {len(job_ids)} jobs")
    
    def __del__(self):
        """소멸자 - 모니터링 태스크 정리"""
        if self._monitoring_tasks:
            try:
                asyncio.create_task(self.cleanup_monitoring())
            except Exception:
                pass  # 이벤트 루프가 없거나 이미 정리됨