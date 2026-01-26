"""
GPU Job Result Collector Service

Collects and manages results from completed GPU cluster jobs including
model files, logs, metrics, and other output artifacts.
"""

import asyncio
import json
import logging
import os
import re
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

from .connection import ClusterConnection
from .models import (
    GPUJob,
    JobResults,
    JobStatus
)
from .exceptions import (
    FileTransferError,
    JobExecutionError,
    ValidationError
)

logger = logging.getLogger(__name__)


class ResultCollector:
    """GPU 작업 결과 수집 서비스"""
    
    def __init__(self, connection: ClusterConnection, local_storage_path: str):
        self.connection = connection
        self.local_storage_path = Path(local_storage_path)
        self.local_storage_path.mkdir(exist_ok=True)
        
    async def collect_results(self, job: GPUJob, cleanup_remote: bool = False) -> JobResults:
        """작업 결과 수집"""
        if job.status != JobStatus.COMPLETED:
            raise ValidationError("status", f"Job must be completed to collect results, current status: {job.status}")
        
        logger.info(f"Collecting results for job {job.job_id}")
        
        # 로컬 결과 디렉토리 생성
        local_job_dir = self.local_storage_path / job.job_id
        local_job_dir.mkdir(exist_ok=True)
        
        try:
            results = JobResults(job_id=job.job_id)
            
            # 1. 모델 파일 수집
            results.model_files = await self._collect_model_files(job, local_job_dir)
            
            # 2. 로그 파일 수집
            results.log_files = await self._collect_log_files(job, local_job_dir)
            
            # 3. 출력 파일 수집
            results.output_files = await self._collect_output_files(job, local_job_dir)
            
            # 4. 메트릭 추출
            results.final_metrics = await self._extract_metrics(job, local_job_dir)
            
            # 5. 학습 히스토리 파싱
            results.training_history = await self._parse_training_history(job, local_job_dir)
            
            # 6. 총 크기 계산
            results.total_size_mb = await self._calculate_total_size(local_job_dir)
            
            # 7. 원격 정리 (옵션)
            if cleanup_remote:
                await self.connection.cleanup_job_files(job.cluster_path, keep_results=False)
                logger.info(f"Cleaned up remote files for job {job.job_id}")
            
            # 8. 결과 메타데이터 저장
            await self._save_results_metadata(job, results, local_job_dir)
            
            logger.info(f"Results collection completed for job {job.job_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to collect results for job {job.job_id}: {e}")
            raise FileTransferError(f"Result collection failed: {str(e)}", job_id=job.job_id)
    
    async def _collect_model_files(self, job: GPUJob, local_dir: Path) -> List[str]:
        """모델 파일 수집"""
        models_dir = local_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # 원격 모델 파일 검색
        model_extensions = ["*.pth", "*.pt", "*.h5", "*.onnx", "*.pkl", "*.joblib"]
        model_files = []
        
        for ext in model_extensions:
            cmd = f"find {job.cluster_path}/outputs -name '{ext}' -type f"
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            if exit_code == 0 and stdout.strip():
                for remote_file in stdout.strip().split('\n'):
                    if remote_file.strip():
                        # 파일 다운로드
                        filename = os.path.basename(remote_file)
                        local_file = models_dir / filename
                        
                        try:
                            await self.connection.download_file(remote_file, str(local_file))
                            model_files.append(str(local_file))
                            logger.debug(f"Downloaded model file: {filename}")
                        except Exception as e:
                            logger.warning(f"Failed to download model file {filename}: {e}")
        
        logger.info(f"Collected {len(model_files)} model files for job {job.job_id}")
        return model_files
    
    async def _collect_log_files(self, job: GPUJob, local_dir: Path) -> List[str]:
        """로그 파일 수집"""
        logs_dir = local_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        try:
            # 로그 디렉토리 전체를 tar로 압축
            remote_logs_dir = f"{job.cluster_path}/logs"
            remote_tar = f"{job.cluster_path}/logs_archive.tar.gz"
            
            # 로그 파일 존재 확인
            cmd = f"test -d {remote_logs_dir} && echo 'EXISTS' || echo 'NOT_FOUND'"
            stdout, _, _ = await self.connection.execute_command(cmd)
            
            if "NOT_FOUND" in stdout:
                logger.warning(f"No logs directory found for job {job.job_id}")
                return []
            
            # tar 압축 생성
            cmd = f"cd {job.cluster_path} && tar -czf logs_archive.tar.gz logs/ 2>/dev/null || true"
            await self.connection.execute_command(cmd)
            
            # 압축 파일 다운로드
            local_tar = local_dir / "logs_archive.tar.gz"
            await self.connection.download_file(remote_tar, str(local_tar))
            
            # 로컬에서 압축 해제
            with tarfile.open(local_tar, 'r:gz') as tar:
                tar.extractall(local_dir)
            
            # 압축 파일 정리
            local_tar.unlink()
            
            # 로그 파일 목록 생성
            log_files = []
            if logs_dir.exists():
                for log_file in logs_dir.rglob('*'):
                    if log_file.is_file():
                        log_files.append(str(log_file))
            
            # 원격 임시 tar 파일 정리
            await self.connection.execute_command(f"rm -f {remote_tar}")
            
            logger.info(f"Collected {len(log_files)} log files for job {job.job_id}")
            return log_files
            
        except Exception as e:
            logger.error(f"Failed to collect log files for job {job.job_id}: {e}")
            return []
    
    async def _collect_output_files(self, job: GPUJob, local_dir: Path) -> List[str]:
        """출력 파일 수집 (모델 파일 제외)"""
        outputs_dir = local_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        try:
            # 모델 파일을 제외한 출력 파일 검색
            exclude_patterns = ["*.pth", "*.pt", "*.h5", "*.onnx", "*.pkl", "*.joblib"]
            exclude_cmd = " ".join([f"-not -name '{pattern}'" for pattern in exclude_patterns])
            
            cmd = f"find {job.cluster_path}/outputs -type f {exclude_cmd}"
            stdout, stderr, exit_code = await self.connection.execute_command(cmd)
            
            output_files = []
            if exit_code == 0 and stdout.strip():
                for remote_file in stdout.strip().split('\n'):
                    if remote_file.strip():
                        # 상대 경로 생성 (출력 디렉토리 구조 유지)
                        rel_path = os.path.relpath(remote_file, f"{job.cluster_path}/outputs")
                        local_file = outputs_dir / rel_path
                        
                        # 하위 디렉토리 생성
                        local_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            await self.connection.download_file(remote_file, str(local_file))
                            output_files.append(str(local_file))
                            logger.debug(f"Downloaded output file: {rel_path}")
                        except Exception as e:
                            logger.warning(f"Failed to download output file {rel_path}: {e}")
            
            logger.info(f"Collected {len(output_files)} output files for job {job.job_id}")
            return output_files
            
        except Exception as e:
            logger.error(f"Failed to collect output files for job {job.job_id}: {e}")
            return []
    
    async def _extract_metrics(self, job: GPUJob, local_dir: Path) -> Dict[str, float]:
        """메트릭 추출"""
        metrics = {}
        
        try:
            # 1. metrics.json 파일에서 추출
            metrics_file = local_dir / "outputs" / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    file_metrics = json.load(f)
                    if isinstance(file_metrics, dict):
                        metrics.update(file_metrics)
            
            # 2. 로그에서 최종 메트릭 파싱
            log_metrics = await self._parse_final_metrics_from_logs(job, local_dir)
            metrics.update(log_metrics)
            
            # 3. TensorBoard 로그에서 추출 (선택사항)
            tb_metrics = await self._extract_tensorboard_metrics(job, local_dir)
            metrics.update(tb_metrics)
            
            logger.info(f"Extracted {len(metrics)} metrics for job {job.job_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to extract metrics for job {job.job_id}: {e}")
            return {}
    
    async def _parse_final_metrics_from_logs(self, job: GPUJob, local_dir: Path) -> Dict[str, float]:
        """로그에서 최종 메트릭 파싱"""
        metrics = {}
        
        try:
            training_log = local_dir / "logs" / "training.log"
            if not training_log.exists():
                return metrics
            
            # 로그 파일의 마지막 몇 줄에서 최종 결과 찾기
            with open(training_log, 'r') as f:
                lines = f.readlines()
            
            # 마지막 100줄에서 메트릭 검색
            recent_lines = lines[-100:] if len(lines) > 100 else lines
            
            for line in reversed(recent_lines):
                line = line.strip()
                
                # 다양한 메트릭 패턴 매칭
                patterns = {
                    'final_loss': re.compile(r'(?:final|test|val).*loss[:=]\s*([\d.]+)', re.IGNORECASE),
                    'final_accuracy': re.compile(r'(?:final|test|val).*acc(?:uracy)?[:=]\s*([\d.]+)', re.IGNORECASE),
                    'final_f1': re.compile(r'(?:final|test|val).*f1[:=]\s*([\d.]+)', re.IGNORECASE),
                    'final_precision': re.compile(r'(?:final|test|val).*precision[:=]\s*([\d.]+)', re.IGNORECASE),
                    'final_recall': re.compile(r'(?:final|test|val).*recall[:=]\s*([\d.]+)', re.IGNORECASE)
                }
                
                for metric_name, pattern in patterns.items():
                    if metric_name not in metrics:
                        match = pattern.search(line)
                        if match:
                            value = float(match.group(1))
                            # 정확도 정규화 (100 초과시 퍼센트로 간주)
                            if 'accuracy' in metric_name and value > 1:
                                value = value / 100
                            metrics[metric_name] = value
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to parse metrics from logs: {e}")
            return {}
    
    async def _parse_training_history(self, job: GPUJob, local_dir: Path) -> Dict[str, List[float]]:
        """학습 히스토리 파싱"""
        history = {}
        
        try:
            # 1. 별도 히스토리 파일에서 로드
            history_file = local_dir / "outputs" / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    file_history = json.load(f)
                    if isinstance(file_history, dict):
                        history.update(file_history)
                        return history
            
            # 2. 로그에서 에포크별 메트릭 파싱
            training_log = local_dir / "logs" / "training.log"
            if not training_log.exists():
                return history
            
            epoch_data = {
                'loss': [],
                'accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            with open(training_log, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Epoch 라인 찾기
                    if 'epoch' in line.lower():
                        # 메트릭 추출
                        metrics = self._extract_metrics_from_line(line)
                        for metric_name, value in metrics.items():
                            if metric_name in epoch_data:
                                epoch_data[metric_name].append(value)
            
            # 빈 리스트 제거
            history = {k: v for k, v in epoch_data.items() if v}
            
            logger.info(f"Parsed training history with {len(history)} metric types for job {job.job_id}")
            return history
            
        except Exception as e:
            logger.error(f"Failed to parse training history for job {job.job_id}: {e}")
            return {}
    
    def _extract_metrics_from_line(self, line: str) -> Dict[str, float]:
        """로그 라인에서 메트릭 추출"""
        metrics = {}
        
        # 일반적인 메트릭 패턴들
        patterns = {
            'loss': re.compile(r'loss[:=]\s*([\d.]+)', re.IGNORECASE),
            'accuracy': re.compile(r'acc(?:uracy)?[:=]\s*([\d.]+)', re.IGNORECASE),
            'val_loss': re.compile(r'val_loss[:=]\s*([\d.]+)', re.IGNORECASE),
            'val_accuracy': re.compile(r'val_acc(?:uracy)?[:=]\s*([\d.]+)', re.IGNORECASE),
            'learning_rate': re.compile(r'lr[:=]\s*([\d.e-]+)', re.IGNORECASE)
        }
        
        for metric_name, pattern in patterns.items():
            match = pattern.search(line)
            if match:
                try:
                    value = float(match.group(1))
                    # 정확도 정규화
                    if 'accuracy' in metric_name and value > 1:
                        value = value / 100
                    metrics[metric_name] = value
                except ValueError:
                    continue
        
        return metrics
    
    async def _extract_tensorboard_metrics(self, job: GPUJob, local_dir: Path) -> Dict[str, float]:
        """TensorBoard 로그에서 메트릭 추출"""
        try:
            tb_dir = local_dir / "logs" / "tensorboard"
            if not tb_dir.exists():
                return {}
            
            # TensorBoard 로그 파싱은 복잡하므로 기본 구현만 제공
            # 실제 환경에서는 tensorboard 라이브러리 사용 권장
            
            metrics = {}
            
            # runs 디렉토리에서 이벤트 파일 찾기
            event_files = list(tb_dir.rglob("events.out.tfevents.*"))
            
            if event_files:
                logger.info(f"Found {len(event_files)} TensorBoard event files")
                # 실제 파싱 로직은 tensorboard 라이브러리 필요
                # 여기서는 플레이스홀더
                metrics['tensorboard_events_found'] = len(event_files)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to extract TensorBoard metrics: {e}")
            return {}
    
    async def _calculate_total_size(self, directory: Path) -> float:
        """디렉토리 총 크기 계산 (MB)"""
        try:
            total_size = 0
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)  # Bytes to MB
            
        except Exception as e:
            logger.warning(f"Failed to calculate directory size: {e}")
            return 0.0
    
    async def _save_results_metadata(self, job: GPUJob, results: JobResults, local_dir: Path):
        """결과 메타데이터 저장"""
        try:
            metadata = {
                'job_id': job.job_id,
                'job_name': job.job_name,
                'collected_at': results.collected_at.isoformat(),
                'total_size_mb': results.total_size_mb,
                'file_counts': {
                    'models': len(results.model_files),
                    'logs': len(results.log_files),
                    'outputs': len(results.output_files)
                },
                'final_metrics': results.final_metrics,
                'training_history_metrics': list(results.training_history.keys())
            }
            
            metadata_file = local_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Saved results metadata for job {job.job_id}")
            
        except Exception as e:
            logger.warning(f"Failed to save metadata for job {job.job_id}: {e}")
    
    async def archive_results(self, job_id: str, archive_format: str = "tar.gz") -> str:
        """결과를 아카이브 파일로 압축"""
        local_job_dir = self.local_storage_path / job_id
        
        if not local_job_dir.exists():
            raise FileTransferError(f"Results directory not found for job {job_id}")
        
        # 아카이브 파일 경로
        archive_name = f"{job_id}_results.{archive_format}"
        archive_path = self.local_storage_path / archive_name
        
        try:
            if archive_format == "tar.gz":
                with tarfile.open(archive_path, 'w:gz') as tar:
                    tar.add(local_job_dir, arcname=job_id)
            elif archive_format == "zip":
                import zipfile
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in local_job_dir.rglob('*'):
                        if file_path.is_file():
                            rel_path = file_path.relative_to(local_job_dir)
                            zipf.write(file_path, f"{job_id}/{rel_path}")
            else:
                raise ValidationError("archive_format", f"Unsupported archive format: {archive_format}")
            
            logger.info(f"Created archive for job {job_id}: {archive_path}")
            return str(archive_path)
            
        except Exception as e:
            logger.error(f"Failed to create archive for job {job_id}: {e}")
            raise FileTransferError(f"Archive creation failed: {str(e)}")
    
    async def get_result_summary(self, job_id: str) -> Optional[Dict]:
        """결과 요약 정보 조회"""
        local_job_dir = self.local_storage_path / job_id
        
        if not local_job_dir.exists():
            return None
        
        try:
            # 메타데이터 파일에서 요약 정보 로드
            metadata_file = local_job_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            
            # 메타데이터 파일이 없으면 직접 계산
            summary = {
                'job_id': job_id,
                'total_size_mb': await self._calculate_total_size(local_job_dir),
                'file_counts': {},
                'collected_at': datetime.now().isoformat()
            }
            
            # 파일 개수 계산
            for subdir in ['models', 'logs', 'outputs']:
                subdir_path = local_job_dir / subdir
                if subdir_path.exists():
                    file_count = len([f for f in subdir_path.rglob('*') if f.is_file()])
                    summary['file_counts'][subdir] = file_count
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get result summary for job {job_id}: {e}")
            return None
    
    async def cleanup_local_results(self, job_id: str, older_than_days: int = 30) -> bool:
        """로컬 결과 정리"""
        local_job_dir = self.local_storage_path / job_id
        
        if not local_job_dir.exists():
            return True
        
        try:
            # 수정 시간 확인
            dir_mtime = datetime.fromtimestamp(local_job_dir.stat().st_mtime)
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            if dir_mtime < cutoff_date:
                shutil.rmtree(local_job_dir)
                logger.info(f"Cleaned up local results for job {job_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cleanup local results for job {job_id}: {e}")
            return False
    
    async def get_storage_usage(self) -> Dict:
        """로컬 스토리지 사용량 조회"""
        try:
            total_size = 0
            job_count = 0
            
            for job_dir in self.local_storage_path.iterdir():
                if job_dir.is_dir():
                    job_count += 1
                    for file_path in job_dir.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
            
            total_size_mb = total_size / (1024 * 1024)
            
            # 디스크 공간 정보
            disk_usage = shutil.disk_usage(self.local_storage_path)
            
            return {
                'total_results_size_mb': total_size_mb,
                'job_count': job_count,
                'disk_total_gb': disk_usage.total / (1024**3),
                'disk_used_gb': disk_usage.used / (1024**3),
                'disk_free_gb': disk_usage.free / (1024**3),
                'storage_path': str(self.local_storage_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage usage: {e}")
            return {}
    
    async def list_available_results(self) -> List[Dict]:
        """사용 가능한 결과 목록"""
        results = []
        
        try:
            for job_dir in self.local_storage_path.iterdir():
                if job_dir.is_dir() and job_dir.name != '.':
                    job_id = job_dir.name
                    summary = await self.get_result_summary(job_id)
                    if summary:
                        results.append(summary)
            
            # 수집 시간순 정렬
            results.sort(key=lambda x: x.get('collected_at', ''), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to list available results: {e}")
            return []
    
    async def export_results(self, job_id: str, export_path: str, format: str = "json") -> str:
        """결과를 특정 형식으로 내보내기"""
        local_job_dir = self.local_storage_path / job_id
        
        if not local_job_dir.exists():
            raise FileTransferError(f"Results not found for job {job_id}")
        
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                # JSON 형식으로 메타데이터 내보내기
                summary = await self.get_result_summary(job_id)
                with open(export_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                    
            elif format == "csv":
                # CSV 형식으로 메트릭 내보내기
                import csv
                
                metadata_file = local_job_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    metrics = metadata.get('final_metrics', {})
                    
                    with open(export_file, 'w', newline='') as f:
                        if metrics:
                            writer = csv.DictWriter(f, fieldnames=metrics.keys())
                            writer.writeheader()
                            writer.writerow(metrics)
                        
            else:
                raise ValidationError("format", f"Unsupported export format: {format}")
            
            logger.info(f"Exported results for job {job_id} to {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"Failed to export results for job {job_id}: {e}")
            raise FileTransferError(f"Export failed: {str(e)}")
    
    async def compare_results(self, job_ids: List[str]) -> Dict:
        """여러 작업 결과 비교"""
        if len(job_ids) < 2:
            raise ValidationError("job_ids", "At least 2 jobs required for comparison")
        
        try:
            comparison = {
                'jobs': [],
                'metrics_comparison': {},
                'performance_ranking': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # 각 작업의 결과 수집
            job_results = []
            for job_id in job_ids:
                summary = await self.get_result_summary(job_id)
                if summary:
                    job_results.append(summary)
                    comparison['jobs'].append({
                        'job_id': job_id,
                        'metrics': summary.get('final_metrics', {}),
                        'collected_at': summary.get('collected_at')
                    })
            
            # 공통 메트릭 비교
            if job_results:
                common_metrics = set(job_results[0].get('final_metrics', {}).keys())
                for result in job_results[1:]:
                    common_metrics &= set(result.get('final_metrics', {}).keys())
                
                for metric in common_metrics:
                    values = []
                    for i, result in enumerate(job_results):
                        value = result.get('final_metrics', {}).get(metric)
                        if value is not None:
                            values.append({
                                'job_id': job_ids[i],
                                'value': value
                            })
                    
                    comparison['metrics_comparison'][metric] = {
                        'values': values,
                        'best': max(values, key=lambda x: x['value']) if values else None,
                        'worst': min(values, key=lambda x: x['value']) if values else None
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare results: {e}")
            raise
    
    async def cleanup_expired_results(self, retention_days: int = 30) -> List[str]:
        """만료된 결과 정리"""
        cleaned_jobs = []
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            for job_dir in self.local_storage_path.iterdir():
                if job_dir.is_dir() and job_dir.name != '.':
                    # 디렉토리 수정 시간 확인
                    dir_mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
                    
                    if dir_mtime < cutoff_date:
                        # 아카이브 생성 후 삭제
                        try:
                            archive_path = await self.archive_results(job_dir.name)
                            shutil.rmtree(job_dir)
                            cleaned_jobs.append(job_dir.name)
                            logger.info(f"Archived and cleaned job {job_dir.name}")
                        except Exception as e:
                            logger.error(f"Failed to archive job {job_dir.name}: {e}")
            
            logger.info(f"Cleaned up {len(cleaned_jobs)} expired result directories")
            return cleaned_jobs
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired results: {e}")
            return []