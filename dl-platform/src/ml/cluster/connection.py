"""
GPU Cluster Connection Manager

Handles SSH and SFTP connections to GPU clusters for job execution and file transfer.
"""

import asyncio
import hashlib
import logging
import os
import stat
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple, List, AsyncIterator
import tarfile
import tempfile

import paramiko
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .exceptions import (
    ConnectionError,
    AuthenticationError, 
    FileTransferError,
    CommandExecutionError,
    SecurityError,
    DiskSpaceError
)
from .models import ClusterConfig

logger = logging.getLogger(__name__)


class ClusterConnection:
    """GPU 클러스터 연결 관리 클래스"""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self._ssh_client: Optional[paramiko.SSHClient] = None
        self._sftp_client: Optional[paramiko.SFTPClient] = None
        self._is_connected = False
        
    async def connect(self) -> bool:
        """클러스터에 연결"""
        try:
            # SSH 클라이언트 초기화
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # SSH 키 파일 권한 확인
            self._verify_key_permissions()
            
            # 연결 시도
            self._ssh_client.connect(
                hostname=self.config.host,
                port=self.config.port,
                username=self.config.username,
                key_filename=self.config.key_path,
                timeout=self.config.timeout,
                banner_timeout=self.config.timeout
            )
            
            # SFTP 클라이언트 초기화
            self._sftp_client = self._ssh_client.open_sftp()
            
            self._is_connected = True
            logger.info(f"Successfully connected to cluster {self.config.host}")
            return True
            
        except paramiko.AuthenticationException as e:
            raise AuthenticationError(f"SSH authentication failed: {str(e)}", host=self.config.host)
        except paramiko.SSHException as e:
            raise ConnectionError(f"SSH connection failed: {str(e)}", host=self.config.host)
        except Exception as e:
            raise ConnectionError(f"Unexpected connection error: {str(e)}", host=self.config.host)
    
    def _verify_key_permissions(self):
        """SSH 키 파일 권한 확인"""
        key_path = Path(self.config.key_path)
        
        if not key_path.exists():
            raise SecurityError(f"SSH key file not found: {self.config.key_path}")
        
        # 권한 확인 (Unix 시스템만)
        if hasattr(os, 'stat'):
            file_stat = key_path.stat()
            if stat.S_IMODE(file_stat.st_mode) & 0o077:
                logger.warning(f"SSH key file has overly permissive permissions: {self.config.key_path}")
    
    async def disconnect(self):
        """연결 종료"""
        try:
            if self._sftp_client:
                self._sftp_client.close()
                self._sftp_client = None
                
            if self._ssh_client:
                self._ssh_client.close()
                self._ssh_client = None
                
            self._is_connected = False
            logger.info(f"Disconnected from cluster {self.config.host}")
            
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
    
    @property
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._is_connected and self._ssh_client and self._ssh_client.get_transport().is_active()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, paramiko.SSHException))
    )
    async def execute_command(self, command: str, timeout: Optional[int] = None) -> Tuple[str, str, int]:
        """명령 실행 (재시도 포함)"""
        if not self.is_connected:
            await self.connect()
            
        try:
            logger.debug(f"Executing command on {self.config.host}: {command}")
            
            # 명령 실행
            stdin, stdout, stderr = self._ssh_client.exec_command(
                command,
                timeout=timeout or self.config.timeout
            )
            
            # 결과 읽기
            stdout_data = stdout.read().decode('utf-8')
            stderr_data = stderr.read().decode('utf-8')
            exit_code = stdout.channel.recv_exit_status()
            
            logger.debug(f"Command completed with exit code {exit_code}")
            
            if exit_code != 0:
                logger.warning(f"Command failed: {command}\nstderr: {stderr_data}")
            
            return stdout_data, stderr_data, exit_code
            
        except paramiko.SSHException as e:
            raise CommandExecutionError(
                command=command,
                stderr=str(e)
            )
        except Exception as e:
            raise ConnectionError(f"Command execution failed: {str(e)}")
    
    async def execute_command_async(self, command: str, timeout: Optional[int] = None) -> Tuple[str, str, int]:
        """비동기 명령 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: asyncio.run(self.execute_command(command, timeout))
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(FileTransferError)
    )
    async def upload_file(self, local_path: str, remote_path: str, verify_integrity: bool = True) -> bool:
        """파일 업로드 (체크섬 검증 포함)"""
        if not self.is_connected:
            await self.connect()
            
        try:
            local_file = Path(local_path)
            if not local_file.exists():
                raise FileTransferError(f"Local file not found: {local_path}")
            
            # 파일 크기 확인
            file_size_gb = local_file.stat().st_size / (1024**3)
            if file_size_gb > self.config.max_file_size_gb:
                raise FileTransferError(
                    f"File too large: {file_size_gb:.2f}GB (max: {self.config.max_file_size_gb}GB)",
                    source_path=local_path
                )
            
            # 원격 디렉토리 생성
            remote_dir = os.path.dirname(remote_path)
            await self._ensure_remote_directory(remote_dir)
            
            # 체크섬 계산 (검증용)
            local_checksum = None
            if verify_integrity:
                local_checksum = await self._calculate_file_checksum(local_path)
            
            logger.info(f"Uploading file: {local_path} -> {remote_path}")
            
            # 파일 전송
            self._sftp_client.put(
                local_path, 
                remote_path,
                callback=self._transfer_progress_callback,
                confirm=True
            )
            
            # 무결성 검증
            if verify_integrity and local_checksum:
                remote_checksum = await self._get_remote_file_checksum(remote_path)
                if local_checksum != remote_checksum:
                    # 파일 삭제 후 예외 발생
                    await self.execute_command(f"rm -f {remote_path}")
                    raise FileTransferError(
                        "File integrity check failed",
                        source_path=local_path,
                        dest_path=remote_path
                    )
            
            logger.info(f"File upload completed: {remote_path}")
            return True
            
        except paramiko.SFTPError as e:
            raise FileTransferError(
                f"SFTP error: {str(e)}",
                source_path=local_path,
                dest_path=remote_path
            )
        except Exception as e:
            raise FileTransferError(
                f"Unexpected upload error: {str(e)}",
                source_path=local_path,
                dest_path=remote_path
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(FileTransferError)
    )
    async def download_file(self, remote_path: str, local_path: str, verify_integrity: bool = True) -> bool:
        """파일 다운로드 (체크섬 검증 포함)"""
        if not self.is_connected:
            await self.connect()
            
        try:
            # 로컬 디렉토리 생성
            local_dir = os.path.dirname(local_path)
            os.makedirs(local_dir, exist_ok=True)
            
            # 원격 파일 존재 확인
            try:
                self._sftp_client.stat(remote_path)
            except FileNotFoundError:
                raise FileTransferError(f"Remote file not found: {remote_path}")
            
            # 체크섬 계산 (검증용)
            remote_checksum = None
            if verify_integrity:
                remote_checksum = await self._get_remote_file_checksum(remote_path)
            
            logger.info(f"Downloading file: {remote_path} -> {local_path}")
            
            # 파일 다운로드
            self._sftp_client.get(
                remote_path,
                local_path,
                callback=self._transfer_progress_callback
            )
            
            # 무결성 검증
            if verify_integrity and remote_checksum:
                local_checksum = await self._calculate_file_checksum(local_path)
                if local_checksum != remote_checksum:
                    # 로컬 파일 삭제 후 예외 발생
                    os.remove(local_path)
                    raise FileTransferError(
                        "File integrity check failed",
                        source_path=remote_path,
                        dest_path=local_path
                    )
            
            logger.info(f"File download completed: {local_path}")
            return True
            
        except paramiko.SFTPError as e:
            raise FileTransferError(
                f"SFTP error: {str(e)}",
                source_path=remote_path,
                dest_path=local_path
            )
        except Exception as e:
            raise FileTransferError(
                f"Unexpected download error: {str(e)}",
                source_path=remote_path,
                dest_path=local_path
            )
    
    async def upload_directory(self, local_dir: str, remote_dir: str) -> bool:
        """디렉토리 전체 업로드 (tar 압축 사용)"""
        try:
            # 임시 tar 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
                tar_path = tmp_file.name
            
            # 디렉토리를 tar.gz로 압축
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(local_dir, arcname='.')
            
            # 압축 파일 업로드
            remote_tar = f"{remote_dir}.tar.gz"
            await self.upload_file(tar_path, remote_tar)
            
            # 원격에서 압축 해제
            await self.execute_command(f"mkdir -p {remote_dir}")
            await self.execute_command(f"cd {remote_dir} && tar -xzf {remote_tar} && rm {remote_tar}")
            
            # 임시 파일 정리
            os.unlink(tar_path)
            
            logger.info(f"Directory upload completed: {local_dir} -> {remote_dir}")
            return True
            
        except Exception as e:
            # 정리 작업
            if 'tar_path' in locals() and os.path.exists(tar_path):
                os.unlink(tar_path)
            raise FileTransferError(f"Directory upload failed: {str(e)}")
    
    async def download_directory(self, remote_dir: str, local_dir: str) -> bool:
        """디렉토리 전체 다운로드 (tar 압축 사용)"""
        try:
            # 원격에서 tar.gz 생성
            remote_tar = f"{remote_dir}.tar.gz"
            await self.execute_command(f"cd {remote_dir} && tar -czf {remote_tar} .")
            
            # 로컬 디렉토리 생성
            os.makedirs(local_dir, exist_ok=True)
            
            # 압축 파일 다운로드
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
                tar_path = tmp_file.name
            
            await self.download_file(remote_tar, tar_path)
            
            # 압축 해제
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(local_dir)
            
            # 정리
            os.unlink(tar_path)
            await self.execute_command(f"rm -f {remote_tar}")
            
            logger.info(f"Directory download completed: {remote_dir} -> {local_dir}")
            return True
            
        except Exception as e:
            # 정리 작업
            if 'tar_path' in locals() and os.path.exists(tar_path):
                os.unlink(tar_path)
            raise FileTransferError(f"Directory download failed: {str(e)}")
    
    async def list_directory(self, remote_path: str) -> List[str]:
        """원격 디렉토리 내용 조회"""
        if not self.is_connected:
            await self.connect()
            
        try:
            return self._sftp_client.listdir(remote_path)
        except Exception as e:
            raise FileTransferError(f"Failed to list directory {remote_path}: {str(e)}")
    
    async def check_disk_space(self, path: str) -> Tuple[float, float]:
        """디스크 공간 확인 (사용량, 전체)"""
        cmd = f"df -BG {path} | tail -1 | awk '{{print $3, $2}}'"
        stdout, stderr, exit_code = await self.execute_command(cmd)
        
        if exit_code != 0:
            raise CommandExecutionError(f"Failed to check disk space: {stderr}")
        
        try:
            used_str, total_str = stdout.strip().split()
            used_gb = float(used_str.rstrip('G'))
            total_gb = float(total_str.rstrip('G'))
            return used_gb, total_gb
        except ValueError as e:
            raise CommandExecutionError(f"Failed to parse disk space output: {stdout}")
    
    async def ensure_disk_space(self, path: str, required_gb: float):
        """필요한 디스크 공간 확인"""
        used_gb, total_gb = await self.check_disk_space(path)
        available_gb = total_gb - used_gb
        
        if available_gb < required_gb:
            raise DiskSpaceError(
                path=path,
                required_gb=required_gb,
                available_gb=available_gb
            )
    
    async def _ensure_remote_directory(self, remote_path: str):
        """원격 디렉토리 생성 확인"""
        cmd = f"mkdir -p {remote_path}"
        await self.execute_command(cmd)
    
    async def _calculate_file_checksum(self, file_path: str) -> str:
        """파일 체크섬 계산 (SHA256)"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _get_remote_file_checksum(self, remote_path: str) -> str:
        """원격 파일 체크섬 조회"""
        cmd = f"sha256sum {remote_path} | cut -d' ' -f1"
        stdout, stderr, exit_code = await self.execute_command(cmd)
        
        if exit_code != 0:
            raise CommandExecutionError(f"Failed to get remote checksum: {stderr}")
        
        return stdout.strip()
    
    def _transfer_progress_callback(self, bytes_transferred: int, total_bytes: int):
        """파일 전송 진행률 콜백"""
        progress = (bytes_transferred / total_bytes) * 100
        logger.debug(f"Transfer progress: {progress:.1f}% ({bytes_transferred}/{total_bytes} bytes)")
    
    async def test_connection(self) -> dict:
        """연결 테스트 및 클러스터 정보 수집"""
        try:
            await self.connect()
            
            # 기본 정보 수집
            info = {}
            
            # 호스트 정보
            stdout, _, _ = await self.execute_command("hostname")
            info['hostname'] = stdout.strip()
            
            # OS 정보
            stdout, _, _ = await self.execute_command("uname -a")
            info['os_info'] = stdout.strip()
            
            # GPU 정보
            stdout, stderr, exit_code = await self.execute_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
            if exit_code == 0:
                gpu_info = []
                for line in stdout.strip().split('\n'):
                    if line.strip():
                        name, memory = line.split(',')
                        gpu_info.append({
                            'name': name.strip(),
                            'memory_mb': int(memory.strip())
                        })
                info['gpu_info'] = gpu_info
            else:
                info['gpu_info'] = []
                logger.warning("Could not retrieve GPU information")
            
            # 디스크 공간
            used_gb, total_gb = await self.check_disk_space(self.config.base_path)
            info['disk_space'] = {
                'used_gb': used_gb,
                'total_gb': total_gb,
                'available_gb': total_gb - used_gb
            }
            
            # 기본 경로 확인/생성
            await self._ensure_remote_directory(self.config.base_path)
            
            return {
                'status': 'connected',
                'cluster_info': info,
                'connection_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'connection_time': datetime.now().isoformat()
            }
    
    @asynccontextmanager
    async def connection_context(self):
        """연결 컨텍스트 매니저"""
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()
    
    async def cleanup_job_files(self, job_path: str, keep_results: bool = True):
        """작업 파일 정리"""
        try:
            if keep_results:
                # 결과를 제외한 임시 파일만 삭제
                cleanup_dirs = ['data', 'models', 'code']
                for dir_name in cleanup_dirs:
                    cmd = f"rm -rf {job_path}/{dir_name}"
                    await self.execute_command(cmd)
                logger.info(f"Cleaned up temporary files for job: {job_path}")
            else:
                # 전체 작업 디렉토리 삭제
                cmd = f"rm -rf {job_path}"
                await self.execute_command(cmd)
                logger.info(f"Removed entire job directory: {job_path}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup job files: {e}")
    
    async def get_system_metrics(self) -> dict:
        """시스템 메트릭 조회"""
        metrics = {}
        
        try:
            # CPU 사용률
            stdout, _, exit_code = await self.execute_command("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1")
            if exit_code == 0:
                metrics['cpu_usage'] = float(stdout.strip())
            
            # 메모리 사용률
            stdout, _, exit_code = await self.execute_command("free | grep Mem | awk '{printf \"%.1f\", $3/$2 * 100.0}'")
            if exit_code == 0:
                metrics['memory_usage'] = float(stdout.strip())
            
            # GPU 사용률
            stdout, _, exit_code = await self.execute_command("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
            if exit_code == 0:
                gpu_usage = [float(line.strip()) for line in stdout.strip().split('\n') if line.strip()]
                metrics['gpu_usage'] = gpu_usage
                metrics['avg_gpu_usage'] = sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0
            
            # 로드 애버리지
            stdout, _, exit_code = await self.execute_command("uptime | awk -F'load average:' '{print $2}' | cut -d',' -f1")
            if exit_code == 0:
                metrics['load_average'] = float(stdout.strip())
                
        except Exception as e:
            logger.warning(f"Failed to collect some system metrics: {e}")
        
        return metrics
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        asyncio.create_task(self.disconnect())
    
    def __del__(self):
        """소멸자"""
        if self._is_connected:
            try:
                asyncio.create_task(self.disconnect())
            except Exception:
                pass  # 이미 정리되었거나 이벤트 루프가 없음