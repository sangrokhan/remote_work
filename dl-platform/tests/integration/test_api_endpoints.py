"""
Integration Tests for GPU Cluster API Endpoints

Tests for API endpoint functionality, request/response handling,
error conditions, and end-to-end workflows.
"""

import pytest
from unittest.mock import AsyncMock, patch
import json
from httpx import AsyncClient
from fastapi.testclient import TestClient

from src.api.main import app
from src.ml.cluster.models import GPUJobRequest, JobStatus, CodeSource
from tests.fixtures.test_data import SAMPLE_JOB_REQUESTS, TEST_USERS


class TestJobManagementEndpoints:
    """Test job management API endpoints."""
    
    @pytest.fixture
    def mock_services(self):
        """Mock all cluster services for API testing."""
        services = {
            'connection': AsyncMock(),
            'orchestrator': AsyncMock(),
            'monitor': AsyncMock(),
            'collector': AsyncMock()
        }
        
        # Configure default behaviors
        services['connection'].connected = True
        services['connection'].test_connection.return_value = True
        
        return services

    async def test_submit_job_git_source(self, test_client: AsyncClient, mock_services):
        """Test job submission with Git source."""
        job_request = SAMPLE_JOB_REQUESTS[0]  # Image classification job
        
        # Mock successful job submission
        from tests.fixtures.test_data import CompletedJobFactory
        mock_job = CompletedJobFactory(**job_request)
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_job.return_value = mock_job
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.post("/api/v1/gpu/jobs", json=job_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_name"] == job_request["job_name"]
        assert data["status"] == "completed"
        assert data["gpu_count"] == job_request["gpu_count"]

    async def test_submit_job_manual_source(self, test_client: AsyncClient):
        """Test job submission with manual file upload."""
        job_request = SAMPLE_JOB_REQUESTS[1]  # NLP fine-tuning job
        
        from tests.fixtures.test_data import GPUJobFactory
        mock_job = GPUJobFactory(**job_request)
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_job.return_value = mock_job
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.post("/api/v1/gpu/jobs", json=job_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["code_source"] == "manual"
        assert data["job_name"] == job_request["job_name"]

    async def test_submit_job_validation_error(self, test_client: AsyncClient):
        """Test job submission with validation errors."""
        invalid_request = {
            "job_name": "",  # Empty name
            "gpu_count": 10,  # Too many GPUs
            "code_source": "invalid_source",
            "entry_script": ""
        }
        
        response = await test_client.post("/api/v1/gpu/jobs", json=invalid_request)
        
        assert response.status_code == 422  # Validation error

    async def test_get_job_success(self, test_client: AsyncClient):
        """Test retrieving job information."""
        job_id = "test_job_123"
        
        from tests.fixtures.test_data import RunningJobFactory
        mock_job = RunningJobFactory(job_id=job_id)
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_job.return_value = mock_job
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "running"

    async def test_get_job_not_found(self, test_client: AsyncClient):
        """Test retrieving non-existent job."""
        job_id = "nonexistent_job"
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_job.return_value = None
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}")
        
        assert response.status_code == 404

    async def test_list_jobs(self, test_client: AsyncClient):
        """Test listing jobs with filters."""
        from tests.fixtures.test_data import GPUJobFactory
        mock_jobs = [
            GPUJobFactory(status=JobStatus.RUNNING),
            GPUJobFactory(status=JobStatus.COMPLETED),
            GPUJobFactory(status=JobStatus.QUEUED)
        ]
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.list_jobs.return_value = mock_jobs
            mock_get_orch.return_value = mock_orchestrator
            
            # Test listing all jobs
            response = await test_client.get("/api/v1/gpu/jobs")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 3
            
            # Test filtering by status
            response = await test_client.get("/api/v1/gpu/jobs?status=running")
            assert response.status_code == 200

    async def test_cancel_job_success(self, test_client: AsyncClient):
        """Test successful job cancellation."""
        job_id = "test_job_123"
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.cancel_job.return_value = True
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.delete(f"/api/v1/gpu/jobs/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "cancelled successfully" in data["message"]

    async def test_cancel_job_not_found(self, test_client: AsyncClient):
        """Test cancelling non-existent job."""
        job_id = "nonexistent_job"
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.cancel_job.return_value = False
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.delete(f"/api/v1/gpu/jobs/{job_id}")
        
        assert response.status_code == 404


class TestJobMonitoringEndpoints:
    """Test job monitoring API endpoints."""
    
    async def test_get_job_status(self, test_client: AsyncClient):
        """Test getting job status."""
        job_id = "test_job_123"
        
        from tests.fixtures.test_data import RunningJobFactory
        mock_job = RunningJobFactory(job_id=job_id)
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch, \
             patch('src.api.endpoints.gpu_cluster.get_monitor') as mock_get_monitor:
            
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_job.return_value = mock_job
            mock_get_orch.return_value = mock_orchestrator
            
            mock_monitor = AsyncMock()
            mock_monitor.check_job_status.return_value = JobStatus.RUNNING
            mock_get_monitor.return_value = mock_monitor
            
            response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}/status")
        
        assert response.status_code == 200
        assert response.json() == "running"

    async def test_get_job_progress(self, test_client: AsyncClient):
        """Test getting job progress."""
        job_id = "test_job_123"
        
        from tests.fixtures.test_data import RunningJobFactory, JobProgressFactory
        mock_job = RunningJobFactory(job_id=job_id)
        mock_progress = JobProgressFactory(job_id=job_id)
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch, \
             patch('src.api.endpoints.gpu_cluster.get_monitor') as mock_get_monitor:
            
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_job.return_value = mock_job
            mock_get_orch.return_value = mock_orchestrator
            
            mock_monitor = AsyncMock()
            mock_monitor.get_job_progress.return_value = mock_progress
            mock_get_monitor.return_value = mock_monitor
            
            response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}/progress")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "progress_percent" in data

    async def test_get_job_logs(self, test_client: AsyncClient):
        """Test getting job logs."""
        job_id = "test_job_123"
        
        from tests.fixtures.test_data import RunningJobFactory
        mock_job = RunningJobFactory(job_id=job_id)
        mock_logs = [
            "2024-08-25 10:00:01 - INFO - Training started",
            "2024-08-25 10:05:00 - INFO - Epoch 1/10: loss=0.856"
        ]
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch, \
             patch('src.api.endpoints.gpu_cluster.get_monitor') as mock_get_monitor:
            
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_job.return_value = mock_job
            mock_get_orch.return_value = mock_orchestrator
            
            mock_monitor = AsyncMock()
            mock_monitor.get_job_logs.return_value = mock_logs
            mock_get_monitor.return_value = mock_monitor
            
            response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}/logs")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert len(data["logs"]) == 2

    async def test_get_job_logs_with_parameters(self, test_client: AsyncClient):
        """Test getting job logs with query parameters."""
        job_id = "test_job_123"
        
        from tests.fixtures.test_data import RunningJobFactory
        mock_job = RunningJobFactory(job_id=job_id)
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch, \
             patch('src.api.endpoints.gpu_cluster.get_monitor') as mock_get_monitor:
            
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_job.return_value = mock_job
            mock_get_orch.return_value = mock_orchestrator
            
            mock_monitor = AsyncMock()
            mock_monitor.get_job_logs.return_value = []
            mock_get_monitor.return_value = mock_monitor
            
            response = await test_client.get(
                f"/api/v1/gpu/jobs/{job_id}/logs?log_type=error&lines=50"
            )
        
        assert response.status_code == 200
        # Verify parameters were passed correctly
        mock_monitor.get_job_logs.assert_called_with(mock_job, "error", 50)


class TestResultCollectionEndpoints:
    """Test result collection API endpoints."""
    
    async def test_collect_job_results(self, test_client: AsyncClient):
        """Test starting result collection."""
        job_id = "test_job_123"
        
        from tests.fixtures.test_data import CompletedJobFactory
        mock_job = CompletedJobFactory(job_id=job_id)
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch, \
             patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_job.return_value = mock_job
            mock_get_orch.return_value = mock_orchestrator
            
            mock_collector = AsyncMock()
            mock_get_collector.return_value = mock_collector
            
            response = await test_client.post(f"/api/v1/gpu/jobs/{job_id}/collect")
        
        assert response.status_code == 200
        data = response.json()
        assert "collection started" in data["message"].lower()

    async def test_get_job_results(self, test_client: AsyncClient):
        """Test retrieving collected job results."""
        job_id = "test_job_123"
        
        mock_summary = {
            "job_id": job_id,
            "file_counts": {"models": 2, "logs": 3, "outputs": 5},
            "final_metrics": {"accuracy": 0.95, "loss": 0.123},
            "total_size_mb": 256.7,
            "collected_at": "2024-08-25T10:00:00"
        }
        
        with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_result_summary.return_value = mock_summary
            mock_get_collector.return_value = mock_collector
            
            response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}/results")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id

    async def test_get_job_results_not_found(self, test_client: AsyncClient):
        """Test retrieving results for job with no collected results."""
        job_id = "test_job_123"
        
        with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_result_summary.return_value = None
            mock_get_collector.return_value = mock_collector
            
            response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}/results")
        
        assert response.status_code == 404

    async def test_download_job_results(self, test_client: AsyncClient):
        """Test downloading job results archive."""
        job_id = "test_job_123"
        
        mock_summary = {"job_id": job_id, "total_size_mb": 100.0}
        archive_path = f"/tmp/{job_id}_results.tar.gz"
        
        with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_result_summary.return_value = mock_summary
            mock_collector.archive_results.return_value = archive_path
            mock_get_collector.return_value = mock_collector
            
            # Mock FileResponse
            with patch('src.api.endpoints.gpu_cluster.FileResponse') as mock_file_response:
                mock_file_response.return_value = "mock_file_response"
                
                response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}/download")
        
        # Should attempt to create archive and return file
        mock_collector.archive_results.assert_called_once_with(job_id, "tar.gz")

    async def test_export_job_results_json(self, test_client: AsyncClient):
        """Test exporting job results in JSON format."""
        job_id = "test_job_123"
        
        export_path = f"/tmp/{job_id}_results.json"
        
        with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.export_results.return_value = export_path
            mock_get_collector.return_value = mock_collector
            
            with patch('src.api.endpoints.gpu_cluster.FileResponse'):
                response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}/export/json")
        
        mock_collector.export_results.assert_called_with(job_id, mock.ANY, "json")

    async def test_compare_jobs(self, test_client: AsyncClient):
        """Test comparing multiple job results."""
        job_ids = ["job_1", "job_2", "job_3"]
        
        mock_comparison = {
            "jobs": [{"job_id": job_id} for job_id in job_ids],
            "metrics_comparison": {
                "accuracy": {
                    "best": {"job_id": "job_2", "value": 0.95},
                    "worst": {"job_id": "job_1", "value": 0.88}
                }
            }
        }
        
        with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.compare_results.return_value = mock_comparison
            mock_get_collector.return_value = mock_collector
            
            response = await test_client.post(
                "/api/v1/gpu/jobs/compare", 
                json=job_ids
            )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 3

    async def test_compare_jobs_insufficient_jobs(self, test_client: AsyncClient):
        """Test job comparison with insufficient jobs."""
        response = await test_client.post("/api/v1/gpu/jobs/compare", json=["single_job"])
        
        assert response.status_code == 400


class TestStorageManagementEndpoints:
    """Test storage management API endpoints."""
    
    async def test_get_storage_usage(self, test_client: AsyncClient):
        """Test getting storage usage information."""
        mock_usage = {
            "total_results_size_mb": 2048.5,
            "job_count": 15,
            "disk_total_gb": 1000.0,
            "disk_used_gb": 250.0,
            "disk_free_gb": 750.0
        }
        
        with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_storage_usage.return_value = mock_usage
            mock_get_collector.return_value = mock_collector
            
            response = await test_client.get("/api/v1/gpu/storage/usage")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_results_size_mb"] == 2048.5
        assert data["job_count"] == 15

    async def test_list_available_results(self, test_client: AsyncClient):
        """Test listing available results."""
        mock_results = [
            {"job_id": "job_1", "total_size_mb": 100.0},
            {"job_id": "job_2", "total_size_mb": 200.0},
            {"job_id": "job_3", "total_size_mb": 150.0}
        ]
        
        with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.list_available_results.return_value = mock_results
            mock_get_collector.return_value = mock_collector
            
            response = await test_client.get("/api/v1/gpu/storage/results")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert len(data["results"]) == 3

    async def test_cleanup_expired_results(self, test_client: AsyncClient):
        """Test cleaning up expired results."""
        with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_get_collector.return_value = mock_collector
            
            response = await test_client.delete("/api/v1/gpu/storage/cleanup")
        
        assert response.status_code == 200
        data = response.json()
        assert "cleanup started" in data["message"].lower()


class TestHealthAndStatusEndpoints:
    """Test health and status API endpoints."""
    
    async def test_cluster_health_check(self, test_client: AsyncClient):
        """Test cluster health endpoint."""
        with patch('src.api.endpoints.gpu_cluster.get_cluster_connection') as mock_get_conn:
            mock_connection = AsyncMock()
            mock_connection.test_connection.return_value = True
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            mock_get_conn.return_value.__aexit__.return_value = None
            
            response = await test_client.get("/api/v1/gpu/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_connected"] is True
        assert data["status"] == "healthy"

    async def test_cluster_health_check_unhealthy(self, test_client: AsyncClient):
        """Test cluster health endpoint when cluster is down."""
        with patch('src.api.endpoints.gpu_cluster.get_cluster_connection') as mock_get_conn:
            mock_connection = AsyncMock()
            mock_connection.test_connection.side_effect = Exception("Connection failed")
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            mock_get_conn.return_value.__aexit__.return_value = None
            
            response = await test_client.get("/api/v1/gpu/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_connected"] is False
        assert data["status"] == "unhealthy"

    async def test_get_cluster_info(self, test_client: AsyncClient):
        """Test getting cluster information."""
        mock_info = {
            "cluster_name": "test-cluster",
            "total_nodes": 4,
            "active_nodes": 4,
            "total_gpus": 16,
            "available_gpus": 12
        }
        
        with patch('src.api.endpoints.gpu_cluster.get_cluster_connection') as mock_get_conn:
            mock_connection = AsyncMock()
            mock_connection.get_cluster_info.return_value = mock_info
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            mock_get_conn.return_value.__aexit__.return_value = None
            
            response = await test_client.get("/api/v1/gpu/cluster/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_name"] == "test-cluster"
        assert data["total_gpus"] == 16


class TestFileUploadEndpoints:
    """Test file upload API endpoints."""
    
    async def test_upload_job_file(self, test_client: AsyncClient):
        """Test uploading additional files to job."""
        job_id = "test_job_123"
        
        from tests.fixtures.test_data import RunningJobFactory
        mock_job = RunningJobFactory(job_id=job_id)
        upload_path = "/cluster/path/uploaded_file.csv"
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_job.return_value = mock_job
            mock_orchestrator.upload_additional_file.return_value = upload_path
            mock_get_orch.return_value = mock_orchestrator
            
            # Create mock file
            file_content = b"col1,col2\n1,2\n3,4\n"
            files = {"file": ("test_data.csv", file_content, "text/csv")}
            
            response = await test_client.post(
                f"/api/v1/gpu/jobs/{job_id}/upload/data",
                files=files
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["file_type"] == "data"
        assert data["filename"] == "test_data.csv"

    async def test_upload_file_invalid_type(self, test_client: AsyncClient):
        """Test uploading file with invalid type."""
        job_id = "test_job_123"
        
        file_content = b"test content"
        files = {"file": ("test.txt", file_content, "text/plain")}
        
        response = await test_client.post(
            f"/api/v1/gpu/jobs/{job_id}/upload/invalid_type",
            files=files
        )
        
        assert response.status_code == 400


class TestConfigurationEndpoint:
    """Test configuration API endpoint."""
    
    async def test_get_configuration(self, test_client: AsyncClient):
        """Test retrieving system configuration."""
        response = await test_client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include app information
        assert "app" in data
        assert "cluster" in data
        assert "storage" in data
        assert "validation" in data


class TestErrorHandling:
    """Test error handling across API endpoints."""
    
    async def test_internal_server_error_handling(self, test_client: AsyncClient):
        """Test handling of internal server errors."""
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_job.side_effect = Exception("Internal error")
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.post("/api/v1/gpu/jobs", json=SAMPLE_JOB_REQUESTS[0])
        
        assert response.status_code == 500

    async def test_validation_error_handling(self, test_client: AsyncClient):
        """Test handling of validation errors."""
        from src.ml.cluster.exceptions import ValidationError
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_job.side_effect = ValidationError("gpu_count", "Too many GPUs")
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.post("/api/v1/gpu/jobs", json=SAMPLE_JOB_REQUESTS[0])
        
        assert response.status_code == 400

    async def test_resource_unavailable_error(self, test_client: AsyncClient):
        """Test handling of resource unavailable errors."""
        from src.ml.cluster.exceptions import ResourceUnavailableError
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_job.side_effect = ResourceUnavailableError(
                "No GPUs available", requested_gpus=4, available_gpus=0
            )
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.post("/api/v1/gpu/jobs", json=SAMPLE_JOB_REQUESTS[0])
        
        assert response.status_code == 429


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    async def test_complete_job_lifecycle(self, test_client: AsyncClient):
        """Test complete job lifecycle from submission to result collection."""
        job_request = SAMPLE_JOB_REQUESTS[0]
        
        # 1. Submit job
        from tests.fixtures.test_data import GPUJobFactory, RunningJobFactory, CompletedJobFactory
        submitted_job = GPUJobFactory(**job_request)
        running_job = RunningJobFactory(job_id=submitted_job.job_id)
        completed_job = CompletedJobFactory(job_id=submitted_job.job_id)
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            
            # Submit job
            mock_orchestrator.submit_job.return_value = submitted_job
            mock_get_orch.return_value = mock_orchestrator
            
            response = await test_client.post("/api/v1/gpu/jobs", json=job_request)
            assert response.status_code == 200
            job_id = response.json()["job_id"]
            
            # 2. Check job status (running)
            mock_orchestrator.get_job.return_value = running_job
            
            with patch('src.api.endpoints.gpu_cluster.get_monitor') as mock_get_monitor:
                mock_monitor = AsyncMock()
                mock_monitor.check_job_status.return_value = JobStatus.RUNNING
                mock_get_monitor.return_value = mock_monitor
                
                response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}/status")
                assert response.status_code == 200
                assert response.json() == "running"
            
            # 3. Check job status (completed)
            mock_orchestrator.get_job.return_value = completed_job
            
            with patch('src.api.endpoints.gpu_cluster.get_monitor') as mock_get_monitor:
                mock_monitor = AsyncMock()
                mock_monitor.check_job_status.return_value = JobStatus.COMPLETED
                mock_get_monitor.return_value = mock_monitor
                
                response = await test_client.get(f"/api/v1/gpu/jobs/{job_id}/status")
                assert response.status_code == 200
                assert response.json() == "completed"
            
            # 4. Collect results
            with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
                mock_collector = AsyncMock()
                mock_get_collector.return_value = mock_collector
                
                response = await test_client.post(f"/api/v1/gpu/jobs/{job_id}/collect")
                assert response.status_code == 200

    async def test_batch_job_operations(self, test_client: AsyncClient):
        """Test batch operations on multiple jobs."""
        # Submit multiple jobs
        job_ids = []
        
        from tests.fixtures.test_data import GPUJobFactory
        
        with patch('src.api.endpoints.gpu_cluster.get_orchestrator') as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_get_orch.return_value = mock_orchestrator
            
            for i in range(3):
                job_request = SAMPLE_JOB_REQUESTS[0].copy()
                job_request["job_name"] = f"batch_job_{i}"
                
                mock_job = GPUJobFactory(**job_request)
                mock_orchestrator.submit_job.return_value = mock_job
                
                response = await test_client.post("/api/v1/gpu/jobs", json=job_request)
                assert response.status_code == 200
                job_ids.append(response.json()["job_id"])
            
            # List all jobs
            mock_jobs = [GPUJobFactory(job_id=job_id) for job_id in job_ids]
            mock_orchestrator.list_jobs.return_value = mock_jobs
            
            response = await test_client.get("/api/v1/gpu/jobs")
            assert response.status_code == 200
            assert len(response.json()) == 3
        
        # Compare results (after completion)
        with patch('src.api.endpoints.gpu_cluster.get_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_comparison = {
                "jobs": [{"job_id": job_id} for job_id in job_ids],
                "metrics_comparison": {}
            }
            mock_collector.compare_results.return_value = mock_comparison
            mock_get_collector.return_value = mock_collector
            
            response = await test_client.post("/api/v1/gpu/jobs/compare", json=job_ids)
            assert response.status_code == 200
            assert len(response.json()["jobs"]) == 3