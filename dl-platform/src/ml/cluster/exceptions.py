"""
GPU Cluster Exception Classes

Custom exceptions for GPU cluster operations.
"""


class ClusterError(Exception):
    """GPU 클러스터 관련 기본 예외"""
    
    def __init__(self, message: str, job_id: str = None, details: dict = None):
        self.message = message
        self.job_id = job_id
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.job_id:
            return f"[Job {self.job_id}] {self.message}"
        return self.message


class ConnectionError(ClusterError):
    """SSH/SCP 연결 관련 예외"""
    
    def __init__(self, message: str, host: str = None, **kwargs):
        self.host = host
        super().__init__(message, **kwargs)
        
    def __str__(self):
        if self.host:
            return f"Connection to {self.host}: {self.message}"
        return f"Connection error: {self.message}"


class AuthenticationError(ClusterError):
    """인증 관련 예외"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, **kwargs)


class FileTransferError(ClusterError):
    """파일 전송 관련 예외"""
    
    def __init__(self, message: str, source_path: str = None, dest_path: str = None, **kwargs):
        self.source_path = source_path
        self.dest_path = dest_path
        super().__init__(message, **kwargs)
        
    def __str__(self):
        if self.source_path and self.dest_path:
            return f"File transfer error ({self.source_path} -> {self.dest_path}): {self.message}"
        return f"File transfer error: {self.message}"


class JobExecutionError(ClusterError):
    """작업 실행 관련 예외"""
    
    def __init__(self, message: str, exit_code: int = None, **kwargs):
        self.exit_code = exit_code
        super().__init__(message, **kwargs)
        
    def __str__(self):
        if self.exit_code is not None:
            return f"Job execution failed (exit code: {self.exit_code}): {self.message}"
        return f"Job execution failed: {self.message}"


class ResourceUnavailableError(ClusterError):
    """리소스 부족 관련 예외"""
    
    def __init__(self, message: str, requested_gpus: int = None, available_gpus: int = None, **kwargs):
        self.requested_gpus = requested_gpus
        self.available_gpus = available_gpus
        super().__init__(message, **kwargs)
        
    def __str__(self):
        if self.requested_gpus is not None and self.available_gpus is not None:
            return f"Resource unavailable (requested: {self.requested_gpus} GPUs, available: {self.available_gpus}): {self.message}"
        return f"Resource unavailable: {self.message}"


class JobNotFoundError(ClusterError):
    """작업을 찾을 수 없는 예외"""
    
    def __init__(self, job_id: str):
        super().__init__(f"Job not found: {job_id}", job_id=job_id)


class JobTimeoutError(ClusterError):
    """작업 시간 초과 예외"""
    
    def __init__(self, job_id: str, timeout_hours: int):
        message = f"Job timed out after {timeout_hours} hours"
        super().__init__(message, job_id=job_id)
        self.timeout_hours = timeout_hours


class ValidationError(ClusterError):
    """입력 검증 예외"""
    
    def __init__(self, field: str, message: str, **kwargs):
        self.field = field
        super().__init__(f"Validation error for {field}: {message}", **kwargs)


class ClusterUnavailableError(ClusterError):
    """클러스터 전체 사용 불가 예외"""
    
    def __init__(self, cluster_host: str, reason: str = None):
        self.cluster_host = cluster_host
        message = f"Cluster {cluster_host} is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ConfigurationError(ClusterError):
    """설정 관련 예외"""
    
    def __init__(self, config_name: str, message: str):
        self.config_name = config_name
        super().__init__(f"Configuration error for {config_name}: {message}")


class SecurityError(ClusterError):
    """보안 관련 예외"""
    
    def __init__(self, message: str, security_issue: str = None, **kwargs):
        self.security_issue = security_issue
        super().__init__(f"Security error: {message}", **kwargs)


class DiskSpaceError(ClusterError):
    """디스크 공간 부족 예외"""
    
    def __init__(self, path: str, required_gb: float, available_gb: float, **kwargs):
        self.path = path
        self.required_gb = required_gb
        self.available_gb = available_gb
        message = f"Insufficient disk space at {path} (required: {required_gb}GB, available: {available_gb}GB)"
        super().__init__(message, **kwargs)


class CommandExecutionError(ClusterError):
    """명령 실행 예외"""
    
    def __init__(self, command: str, stdout: str = "", stderr: str = "", exit_code: int = None, **kwargs):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        
        message = f"Command execution failed: {command}"
        if stderr:
            message += f"\nError output: {stderr}"
            
        super().__init__(message, **kwargs)


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAP = {
    ValidationError: 400,
    JobNotFoundError: 404,
    AuthenticationError: 401,
    SecurityError: 403,
    ResourceUnavailableError: 429,
    JobTimeoutError: 408,
    ClusterUnavailableError: 503,
    ConfigurationError: 500,
    ConnectionError: 502,
    FileTransferError: 502,
    JobExecutionError: 500,
    DiskSpaceError: 507,
    CommandExecutionError: 500,
    ClusterError: 500
}


def get_http_status(exception: Exception) -> int:
    """예외에 대응하는 HTTP 상태 코드 반환"""
    for exc_class, status_code in EXCEPTION_STATUS_MAP.items():
        if isinstance(exception, exc_class):
            return status_code
    return 500  # Internal Server Error for unknown exceptions


def create_error_response(exception: Exception) -> dict:
    """예외를 API 응답 형태로 변환"""
    return {
        "error": exception.__class__.__name__,
        "message": str(exception),
        "details": getattr(exception, 'details', {}),
        "job_id": getattr(exception, 'job_id', None)
    }