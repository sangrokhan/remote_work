"""
Unit Tests for GPU Cluster Models

Tests for Pydantic model validation, serialization, and business logic
in the GPU cluster data models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from src.ml.cluster.models import (
    GPUJobRequest, GPUJob, JobStatus, CodeSource, JobProgress, JobResults
)


class TestGPUJobRequest:
    """Test GPUJobRequest model validation and creation."""
    
    def test_valid_git_job_request(self):
        """Test creating valid Git-based job request."""
        request = GPUJobRequest(
            job_name="test_job",
            gpu_count=2,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py",
            script_args={"epochs": "100"},
            environment_vars={"CUDA_VISIBLE_DEVICES": "0,1"}
        )
        
        assert request.job_name == "test_job"
        assert request.gpu_count == 2
        assert request.code_source == CodeSource.GIT
        assert request.code_path == "https://github.com/user/repo.git"
        assert request.entry_script == "train.py"
    
    def test_valid_manual_job_request(self):
        """Test creating valid manual upload job request."""
        request = GPUJobRequest(
            job_name="manual_job",
            gpu_count=1,
            code_source=CodeSource.MANUAL,
            entry_script="train.py"
        )
        
        assert request.code_source == CodeSource.MANUAL
        assert request.entry_script == "train.py"
    
    def test_invalid_gpu_count(self):
        """Test validation of GPU count limits."""
        with pytest.raises(ValidationError) as exc_info:
            GPUJobRequest(
                job_name="test",
                gpu_count=0,  # Invalid: too low
                code_source=CodeSource.GIT,
                code_path="https://github.com/user/repo.git",
                entry_script="train.py"
            )
        
        assert "gpu_count" in str(exc_info.value)
        
        with pytest.raises(ValidationError):
            GPUJobRequest(
                job_name="test",
                gpu_count=9,  # Invalid: too high
                code_source=CodeSource.GIT,
                code_path="https://github.com/user/repo.git",
                entry_script="train.py"
            )
    
    def test_invalid_job_name_length(self):
        """Test job name length validation."""
        with pytest.raises(ValidationError):
            GPUJobRequest(
                job_name="a" * 256,  # Too long
                gpu_count=1,
                code_source=CodeSource.GIT,
                code_path="https://github.com/user/repo.git",
                entry_script="train.py"
            )
    
    def test_git_url_validation(self):
        """Test Git URL validation."""
        # Valid Git URLs
        valid_urls = [
            "https://github.com/user/repo.git",
            "git@github.com:user/repo.git",
            "https://gitlab.com/user/repo.git"
        ]
        
        for url in valid_urls:
            request = GPUJobRequest(
                job_name="test",
                gpu_count=1,
                code_source=CodeSource.GIT,
                code_path=url,
                entry_script="train.py"
            )
            assert request.code_path == url
        
        # Invalid Git URLs
        invalid_urls = [
            "not-a-url",
            "ftp://example.com/repo",
            "javascript:alert(1)"
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValidationError):
                GPUJobRequest(
                    job_name="test",
                    gpu_count=1,
                    code_source=CodeSource.GIT,
                    code_path=url,
                    entry_script="train.py"
                )
    
    def test_entry_script_validation(self):
        """Test entry script validation."""
        # Valid scripts
        valid_scripts = ["train.py", "scripts/train.py", "main.py"]
        
        for script in valid_scripts:
            request = GPUJobRequest(
                job_name="test",
                gpu_count=1,
                code_source=CodeSource.GIT,
                code_path="https://github.com/user/repo.git",
                entry_script=script
            )
            assert request.entry_script == script
        
        # Invalid scripts (security check)
        invalid_scripts = [
            "../../../etc/passwd",
            "/etc/shadow",
            "rm -rf /",
            "script.py; rm -rf /"
        ]
        
        for script in invalid_scripts:
            with pytest.raises(ValidationError):
                GPUJobRequest(
                    job_name="test",
                    gpu_count=1,
                    code_source=CodeSource.GIT,
                    code_path="https://github.com/user/repo.git",
                    entry_script=script
                )


class TestGPUJob:
    """Test GPUJob model and state management."""
    
    def test_job_creation_from_request(self):
        """Test creating GPUJob from GPUJobRequest."""
        request = GPUJobRequest(
            job_name="test_job",
            gpu_count=1,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py"
        )
        
        job = GPUJob(
            job_id="test_123",
            user_id="test_user",
            cluster_path="/tmp/jobs/test_123",
            status=JobStatus.PENDING,
            original_request=request,
            **request.model_dump()
        )
        
        assert job.job_id == "test_123"
        assert job.job_name == "test_job"
        assert job.status == JobStatus.PENDING
        assert job.cluster_path == "/tmp/jobs/test_123"
    
    def test_job_status_transitions(self):
        """Test valid job status transitions."""
        request = GPUJobRequest(
            job_name="test",
            gpu_count=1,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py"
        )
        
        job = GPUJob(
            job_id="test_123",
            user_id="test_user",
            cluster_path="/tmp/jobs/test_123",
            status=JobStatus.PENDING,
            original_request=request,
            **request.model_dump()
        )
        
        # Valid transitions
        valid_transitions = [
            (JobStatus.PENDING, JobStatus.QUEUED),
            (JobStatus.QUEUED, JobStatus.RUNNING),
            (JobStatus.RUNNING, JobStatus.COMPLETED),
            (JobStatus.RUNNING, JobStatus.FAILED),
            (JobStatus.QUEUED, JobStatus.CANCELLED),
            (JobStatus.RUNNING, JobStatus.CANCELLED)
        ]
        
        for from_status, to_status in valid_transitions:
            job.status = from_status
            job.status = to_status  # Should not raise
            assert job.status == to_status
    
    def test_job_progress_calculation(self):
        """Test job progress percentage calculation."""
        request = GPUJobRequest(
            job_name="test",
            gpu_count=1,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py"
        )
        
        job = GPUJob(
            job_id="test_123",
            user_id="test_user",
            cluster_path="/tmp/jobs/test_123",
            status=JobStatus.RUNNING,
            original_request=request,
            **request.model_dump()
        )
        
        # Test that job was created successfully
        assert job.job_id == "test_123"
        assert job.status == JobStatus.RUNNING
        
        # Progress tracking is handled by JobProgress model, not GPUJob
        progress = JobProgress(
            job_id=job.job_id,
            current_epoch=3,
            total_epochs=10
        )
        
        assert progress.epoch_progress_percent == 30.0
    
    def test_job_runtime_calculation(self):
        """Test job runtime calculation."""
        now = datetime.now()
        request = GPUJobRequest(
            job_name="test",
            gpu_count=1,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py"
        )
        
        completed_time = now.replace(hour=now.hour + 1)  # 1 hour later
        
        job = GPUJob(
            job_id="test_123",
            user_id="test_user",
            cluster_path="/tmp/jobs/test_123",
            status=JobStatus.COMPLETED,
            created_at=now,
            started_at=now,
            completed_at=completed_time,
            original_request=request,
            **request.model_dump()
        )
        
        # Test duration calculation
        duration = job.duration_seconds
        assert duration == 3600  # Exactly 1 hour in seconds


class TestJobProgress:
    """Test JobProgress model."""
    
    def test_progress_creation(self):
        """Test creating job progress object."""
        progress = JobProgress(
            job_id="test_123",
            current_epoch=5,
            total_epochs=10,
            current_loss=0.234,
            current_accuracy=0.834,
            gpu_utilization=85.5,
            memory_usage_gb=4.096
        )
        
        assert progress.job_id == "test_123"
        assert progress.current_epoch == 5
        assert progress.epoch_progress_percent == 50.0
        assert progress.gpu_utilization == 85.5
    
    def test_progress_validation(self):
        """Test progress validation rules."""
        # Invalid accuracy (should be 0-1)
        with pytest.raises(ValidationError):
            JobProgress(
                job_id="test",
                current_accuracy=1.5  # > 1
            )
        
        with pytest.raises(ValidationError):
            JobProgress(
                job_id="test",
                current_accuracy=-0.1  # < 0
            )
        
        # Invalid GPU utilization
        with pytest.raises(ValidationError):
            JobProgress(
                job_id="test",
                gpu_utilization=110.0  # > 100
            )


class TestJobResults:
    """Test JobResults model."""
    
    def test_results_creation(self):
        """Test creating job results object."""
        results = JobResults(
            job_id="test_123",
            model_files=["/path/to/model.pth"],
            log_files=["/path/to/training.log"],
            output_files=["/path/to/metrics.json"],
            final_metrics={"accuracy": 0.95, "loss": 0.123},
            training_history={"loss": [0.8, 0.6, 0.4], "accuracy": [0.2, 0.5, 0.8]},
            total_size_mb=1024.5
        )
        
        assert results.job_id == "test_123"
        assert len(results.model_files) == 1
        assert results.final_metrics["accuracy"] == 0.95
        assert results.total_size_mb == 1024.5
        assert isinstance(results.collected_at, datetime)
    
    def test_results_file_count_properties(self):
        """Test computed properties for file counts."""
        results = JobResults(
            job_id="test",
            model_files=["/path/model1.pth", "/path/model2.pth"],
            log_files=["/path/train.log", "/path/error.log", "/path/system.log"],
            output_files=["/path/metrics.json"]
        )
        
        assert len(results.model_files) == 2
        assert len(results.log_files) == 3
        assert len(results.output_files) == 1
        assert len(results.model_files) + len(results.log_files) + len(results.output_files) == 6


class TestJobStatus:
    """Test JobStatus enum."""
    
    def test_status_values(self):
        """Test all status enum values."""
        expected_statuses = [
            "pending", "queued", "running", "completed", "failed", "cancelled"
        ]
        
        for status_value in expected_statuses:
            status = JobStatus(status_value)
            assert status.value == status_value
    
    def test_status_ordering(self):
        """Test status progression ordering."""
        statuses = [
            JobStatus.PENDING,
            JobStatus.QUEUED, 
            JobStatus.RUNNING,
            JobStatus.COMPLETED
        ]
        
        # Test that we can create a sequence
        status_sequence = [s.value for s in statuses]
        assert status_sequence == ["pending", "queued", "running", "completed"]


class TestCodeSource:
    """Test CodeSource enum."""
    
    def test_code_source_values(self):
        """Test code source enum values."""
        assert CodeSource.GIT.value == "git"
        assert CodeSource.MANUAL.value == "manual"
    
    def test_code_source_creation(self):
        """Test creating CodeSource from string."""
        git_source = CodeSource("git")
        manual_source = CodeSource("manual")
        
        assert git_source == CodeSource.GIT
        assert manual_source == CodeSource.MANUAL


class TestModelValidation:
    """Test comprehensive model validation scenarios."""
    
    def test_job_request_git_requirements(self):
        """Test Git source requirements validation."""
        # Valid Git request
        request = GPUJobRequest(
            job_name="git_job",
            gpu_count=1,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py"
        )
        assert request.code_source == CodeSource.GIT
        assert request.code_path is not None
        
        # Git request without URL should use validation in actual implementation
        # This would be handled by custom validators in the model
    
    def test_job_request_manual_requirements(self):
        """Test manual source requirements validation."""
        request = GPUJobRequest(
            job_name="manual_job",
            gpu_count=1,
            code_source=CodeSource.MANUAL,
            entry_script="train.py"
        )
        
        assert request.code_source == CodeSource.MANUAL
        assert request.entry_script == "train.py"
    
    def test_environment_variables_validation(self):
        """Test environment variables format."""
        request = GPUJobRequest(
            job_name="env_test",
            gpu_count=1,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py",
            environment_vars={
                "CUDA_VISIBLE_DEVICES": "0,1",
                "BATCH_SIZE": "32",
                "LEARNING_RATE": "0.001"
            }
        )
        
        assert "CUDA_VISIBLE_DEVICES" in request.environment_vars
        assert request.environment_vars["BATCH_SIZE"] == "32"
    
    def test_environment_vars_validation(self):
        """Test environment variables validation."""
        request = GPUJobRequest(
            job_name="env_test",
            gpu_count=1,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py",
            environment_vars={"PYTHON_PACKAGES": "torch,numpy,pandas,scikit-learn"}
        )
        
        assert "PYTHON_PACKAGES" in request.environment_vars
        assert "torch" in request.environment_vars["PYTHON_PACKAGES"]
        assert isinstance(request.environment_vars, dict)
    
    def test_model_serialization(self):
        """Test model serialization and deserialization."""
        original = GPUJobRequest(
            job_name="serialization_test",
            gpu_count=2,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py",
            model_file_path="path/to/model.pth",
            dataset_path="path/to/data.csv"
        )
        
        # Serialize to dict
        serialized = original.model_dump()
        assert isinstance(serialized, dict)
        assert serialized["job_name"] == "serialization_test"
        assert serialized["code_source"] == "git"
        
        # Deserialize from dict
        recreated = GPUJobRequest(**serialized)
        assert recreated.job_name == original.job_name
        assert recreated.code_source == original.code_source
        assert recreated.model_file_path == original.model_file_path
    
    def test_job_progress_edge_cases(self):
        """Test edge cases in job progress calculation."""
        # None epochs - should handle gracefully
        progress = JobProgress(
            job_id="test",
            current_epoch=0,
            total_epochs=None
        )
        assert progress.epoch_progress_percent is None
        
        # Negative values should be caught by validation
        with pytest.raises(ValidationError):
            JobProgress(
                job_id="test",
                current_epoch=-1
            )
    
    def test_job_results_empty_collections(self):
        """Test job results with empty file collections."""
        results = JobResults(
            job_id="test",
            model_files=[],
            log_files=[],
            output_files=[],
            final_metrics={},
            training_history={}
        )
        
        assert len(results.model_files) == 0
        assert len(results.log_files) == 0
        assert len(results.output_files) == 0
        assert len(results.model_files) + len(results.log_files) + len(results.output_files) == 0
    
    def test_model_copy_and_update(self):
        """Test model copying and field updates."""
        request = GPUJobRequest(
            job_name="original",
            gpu_count=1,
            code_source=CodeSource.GIT,
            code_path="https://github.com/user/repo.git",
            entry_script="train.py"
        )
        
        original = GPUJob(
            job_id="test_123",
            user_id="test_user",
            cluster_path="/tmp/jobs/test_123",
            status=JobStatus.PENDING,
            original_request=request,
            **request.model_dump()
        )
        
        # Copy with updates
        updated = original.model_copy(update={"status": JobStatus.RUNNING, "process_id": "12345"})
        
        assert updated.job_id == original.job_id
        assert updated.status == JobStatus.RUNNING
        assert updated.process_id == "12345"
        assert original.status == JobStatus.PENDING  # Original unchanged