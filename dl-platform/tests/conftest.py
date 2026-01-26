"""
PyTest Configuration and Global Fixtures

Shared test configuration, fixtures, and utilities for the GPU cluster
integration test suite.
"""

import asyncio
import os
import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

# Set environment variables before importing app
os.environ.update({
    'STORAGE_RESULTS_PATH': '/tmp/test_results',
    'STORAGE_UPLOADS_PATH': '/tmp/test_uploads', 
    'STORAGE_TEMP_PATH': '/tmp/test_temp',
    'CLUSTER_HOST': 'test-cluster.example.com',
    'CLUSTER_USERNAME': 'testuser',
    'CLUSTER_PASSWORD': 'testpass',
    'CLUSTER_WORK_DIR': '/tmp/test_work',
    'CLUSTER_SSH_PORT': '22',
    'CLUSTER_CONNECTION_TIMEOUT': '30',
    'CLUSTER_RETRY_ATTEMPTS': '3',
    'SECRET_KEY': 'test-secret-key-for-testing-only',
    'SECURITY_SECRET_KEY': 'test-secret-key-for-testing-only',
    'ALGORITHM': 'HS256',
    'ACCESS_TOKEN_EXPIRE_MINUTES': '60',
    'DATABASE_URL': 'sqlite+aiosqlite:///./test.db',
    'REDIS_URL': 'redis://localhost:6379/1'
})

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Test imports
from src.api.main import app
from src.core.database import Base, get_db
from src.core.config import AppSettings, ClusterSettings, DatabaseSettings, StorageSettings
from src.ml.cluster.connection import ClusterConnection
from src.ml.cluster.models import GPUJobRequest, GPUJob, JobStatus, CodeSource


# Test Database Configuration

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_db.sqlite"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = async_sessionmaker(test_engine, expire_on_commit=False)
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_client(test_session) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client with database dependency override."""
    
    async def override_get_db():
        yield test_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()


# Configuration Fixtures

@pytest.fixture
def test_cluster_config() -> ClusterSettings:
    """Test cluster configuration."""
    return ClusterSettings(
        host="test-cluster.local",
        port=22,
        username="testuser",
        private_key_path=None,
        password="testpass",
        base_path="/tmp/test_jobs",
        max_concurrent_jobs=5,
        max_gpu_per_job=4,
        connection_timeout=10,
        command_timeout=30
    )


@pytest.fixture
def test_storage_config(tmp_path) -> StorageSettings:
    """Test storage configuration with temporary paths."""
    return StorageSettings(
        results_path=str(tmp_path / "results"),
        uploads_path=str(tmp_path / "uploads"),
        temp_path=str(tmp_path / "temp"),
        max_upload_size_mb=100,
        result_retention_days=7
    )


@pytest.fixture
def test_app_settings(test_cluster_config, test_storage_config) -> AppSettings:
    """Test application settings."""
    return AppSettings(
        app_name="Test DL Platform",
        debug=True,
        environment="testing",
        database=DatabaseSettings(database_url=TEST_DATABASE_URL),
        cluster=test_cluster_config,
        storage=test_storage_config,
        security={"secret_key": "test-secret-key"}
    )


# Mock SSH Connection Fixtures

@pytest.fixture
def mock_paramiko_client():
    """Mock paramiko SSH client."""
    mock_client = MagicMock()
    mock_stdin = MagicMock()
    mock_stdout = MagicMock()
    mock_stderr = MagicMock()
    
    # Configure mock responses
    mock_stdout.read.return_value = b"test output"
    mock_stderr.read.return_value = b""
    mock_stdout.channel.recv_exit_status.return_value = 0
    
    mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)
    
    return mock_client


@pytest.fixture
def mock_scp_client():
    """Mock SCP client."""
    mock_scp = MagicMock()
    mock_scp.put.return_value = None
    mock_scp.get.return_value = None
    return mock_scp


@pytest.fixture
async def mock_cluster_connection(test_cluster_config, mock_paramiko_client, mock_scp_client):
    """Mock cluster connection with configured responses."""
    
    class MockClusterConnection:
        def __init__(self, config):
            self.config = config
            self._client = mock_paramiko_client
            self._scp = mock_scp_client
            self.connected = False
        
        async def connect(self):
            self.connected = True
            return True
        
        async def disconnect(self):
            self.connected = False
        
        async def execute_command(self, command, timeout=None):
            if "test -f" in command:
                return "file exists", "", 0
            elif "find" in command and "*.pth" in command:
                return "/tmp/test/model.pth\n/tmp/test/best_model.pth", "", 0
            elif "ps aux" in command:
                return "user 12345 python train.py", "", 0
            else:
                return "success", "", 0
        
        async def upload_file(self, local_path, remote_path, verify_integrity=True):
            return True
        
        async def download_file(self, remote_path, local_path, verify_integrity=True):
            # Create mock file
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_text("mock file content")
            return True
        
        async def test_connection(self):
            return True
        
        async def close(self):
            await self.disconnect()
    
    connection = MockClusterConnection(test_cluster_config)
    await connection.connect()
    
    return connection


# Test Data Fixtures

@pytest.fixture
def sample_job_request() -> GPUJobRequest:
    """Sample GPU job request for testing."""
    return GPUJobRequest(
        job_name="test_training_job",
        gpu_count=2,
        code_source=CodeSource.GIT,
        git_url="https://github.com/user/ml-project.git",
        git_branch="main",
        entry_script="train.py",
        model_files={"pretrained": "models/pretrained.pth"},
        data_files={"train": "data/train.csv", "val": "data/val.csv"},
        environment_vars={"CUDA_VISIBLE_DEVICES": "0,1"},
        python_packages=["torch", "transformers"],
        docker_image="pytorch/pytorch:latest"
    )


@pytest.fixture
def sample_gpu_job(sample_job_request) -> GPUJob:
    """Sample GPU job for testing."""
    return GPUJob(
        job_id="test_job_12345",
        **sample_job_request.dict(),
        status=JobStatus.PENDING,
        cluster_path="/tmp/test_jobs/test_job_12345",
        created_at=datetime.now()
    )


@pytest.fixture
def sample_completed_job(sample_gpu_job) -> GPUJob:
    """Sample completed GPU job for testing."""
    job = sample_gpu_job.copy()
    job.status = JobStatus.COMPLETED
    job.started_at = datetime.now()
    job.completed_at = datetime.now()
    job.final_metrics = {
        "final_loss": 0.123,
        "final_accuracy": 0.967,
        "training_time_hours": 2.5
    }
    return job


# File System Fixtures

@pytest.fixture
def temp_workspace(tmp_path) -> Path:
    """Create temporary workspace for file operations."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    
    # Create sample files
    (workspace / "train.py").write_text("""
import torch
print("Training started")
for epoch in range(10):
    loss = 1.0 / (epoch + 1)
    acc = min(0.9, epoch * 0.1)
    print(f"Epoch {epoch}: loss={loss:.3f}, acc={acc:.3f}")
print("Training completed")
""")
    
    (workspace / "requirements.txt").write_text("torch\nnumpy\n")
    
    models_dir = workspace / "models"
    models_dir.mkdir()
    (models_dir / "model.pth").write_bytes(b"fake model data")
    
    return workspace


@pytest.fixture
def mock_file_operations():
    """Mock file system operations."""
    with patch('os.path.exists', return_value=True), \
         patch('os.makedirs'), \
         patch('shutil.copy2'), \
         patch('tarfile.open'), \
         patch('pathlib.Path.mkdir'), \
         patch('pathlib.Path.write_text'), \
         patch('pathlib.Path.write_bytes'):
        yield


# SSH Mock Responses

@pytest.fixture
def ssh_responses() -> Dict[str, tuple]:
    """Standard SSH command responses for testing."""
    return {
        "test -f /tmp/file.txt": ("", "", 0),
        "mkdir -p /tmp/test": ("", "", 0),
        "python3 --version": ("Python 3.9.0", "", 0),
        "pip install torch": ("Successfully installed torch", "", 0),
        "ps aux | grep python": ("user 12345 python train.py", "", 0),
        "find /tmp -name '*.pth'": ("/tmp/model.pth\n/tmp/best_model.pth", "", 0),
        "du -sh /tmp/test": ("1.2G\t/tmp/test", "", 0),
        "cat /tmp/test/train.log | tail -10": (
            "Epoch 8: loss=0.125, acc=0.89\n"
            "Epoch 9: loss=0.111, acc=0.91\n"
            "Final results: loss=0.095, accuracy=0.94",
            "", 0
        )
    }


# Performance Testing Fixtures

@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Performance testing thresholds."""
    return {
        "api_response_time_ms": 500,
        "file_transfer_rate_mbps": 10,
        "job_submission_time_ms": 1000,
        "status_check_time_ms": 100
    }


# Factory Functions

def create_test_job(job_id: str = None, status: JobStatus = JobStatus.PENDING) -> GPUJob:
    """Factory function for creating test jobs."""
    return GPUJob(
        job_id=job_id or f"test_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        job_name="Test Training Job",
        gpu_count=1,
        code_source=CodeSource.GIT,
        git_url="https://github.com/test/repo.git",
        entry_script="train.py",
        status=status,
        created_at=datetime.now()
    )


def create_test_user() -> Dict:
    """Factory function for creating test users."""
    return {
        "username": f"testuser_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "email": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}@example.com",
        "full_name": "Test User",
        "is_active": True
    }


# Cleanup Fixtures

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Auto cleanup temporary test files after each test."""
    yield
    
    # Cleanup any test files created during tests
    test_dirs = ["/tmp/test_gpu_jobs", "/tmp/test_results"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)


# Pytest configuration

pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


# Async test configuration
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"