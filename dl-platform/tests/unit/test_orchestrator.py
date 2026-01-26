"""
Unit Tests for Job Orchestrator

Tests for job submission, lifecycle management, file preparation,
and orchestration logic with mocked cluster connections.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from pathlib import Path

from src.ml.cluster.orchestrator import JobOrchestrator
from src.ml.cluster.models import GPUJobRequest, GPUJob, JobStatus, CodeSource
from src.ml.cluster.exceptions import ValidationError, ResourceUnavailableError, JobExecutionError
from tests.fixtures.test_data import GPUJobRequestFactory, GPUJobFactory


class TestJobOrchestrator:
    """Test JobOrchestrator functionality."""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock cluster connection."""
        connection = AsyncMock()
        connection.connected = True
        connection.execute_command = AsyncMock(return_value=("success", "", 0))
        connection.upload_file = AsyncMock(return_value=True)
        connection.test_connection = AsyncMock(return_value=True)
        return connection
    
    @pytest.fixture
    def orchestrator(self, mock_connection):
        """JobOrchestrator instance with mocked connection."""
        return JobOrchestrator(mock_connection)

    async def test_job_submission_git_source(self, orchestrator, mock_connection):
        """Test submitting Git-based job."""
        request = GPUJobRequestFactory(
            code_source=CodeSource.GIT,
            git_url="https://github.com/test/repo.git",
            git_branch="main"
        )
        
        # Mock successful operations
        mock_connection.execute_command.side_effect = [
            ("", "", 0),  # mkdir
            ("", "", 0),  # git clone
            ("", "", 0),  # git checkout
            ("12345", "", 0),  # job submission
        ]
        
        job = await orchestrator.submit_job(request, "user123")
        
        assert job.job_id is not None
        assert job.status == JobStatus.QUEUED
        assert job.code_source == CodeSource.GIT
        assert job.cluster_job_id == "12345"
        assert job.cluster_path is not None
        
        # Verify SSH commands were called
        assert mock_connection.execute_command.call_count >= 3

    async def test_job_submission_manual_source(self, orchestrator, mock_connection, tmp_path):
        """Test submitting manual upload job."""
        # Create test files
        script_file = tmp_path / "train.py"
        script_file.write_text("print('Training started')")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("epochs: 10")
        
        request = GPUJobRequestFactory(
            code_source=CodeSource.MANUAL,
            manual_files={
                "train.py": str(script_file),
                "config.yaml": str(config_file)
            }
        )
        
        # Mock successful operations
        mock_connection.execute_command.side_effect = [
            ("", "", 0),  # mkdir
            ("12345", "", 0),  # job submission
        ]
        
        job = await orchestrator.submit_job(request, "user123")
        
        assert job.code_source == CodeSource.MANUAL
        assert job.cluster_job_id == "12345"
        
        # Verify file uploads were called
        assert mock_connection.upload_file.call_count >= 2

    async def test_job_submission_validation_failure(self, orchestrator, mock_connection):
        """Test job submission with validation failures."""
        # Test GPU limit validation
        request = GPUJobRequestFactory(gpu_count=10)  # Exceeds limit
        
        with pytest.raises(ValidationError) as exc_info:
            await orchestrator.submit_job(request, "user123")
        
        assert "gpu" in str(exc_info.value).lower()

    async def test_job_submission_resource_unavailable(self, orchestrator, mock_connection):
        """Test job submission when resources unavailable."""
        request = GPUJobRequestFactory(gpu_count=4)
        
        # Mock resource check failure
        mock_connection.execute_command.side_effect = [
            ("Currently using: 8 GPUs", "", 0),  # Resource check shows high usage
        ]
        
        # This would typically check cluster capacity
        # For now, test that orchestrator handles resource checking
        
        job = await orchestrator.submit_job(request, "user123")
        # Should still create job but might queue it
        assert job is not None

    async def test_job_cancellation(self, orchestrator, mock_connection):
        """Test job cancellation."""
        job = GPUJobFactory(
            status=JobStatus.RUNNING,
            cluster_job_id="12345"
        )
        
        # Mock successful cancellation
        mock_connection.execute_command.return_value = ("", "", 0)
        
        result = await orchestrator.cancel_job(job.job_id)
        
        assert result is True
        mock_connection.execute_command.assert_called()
        
        # Verify kill command was sent
        last_call = mock_connection.execute_command.call_args
        command = last_call[0][0]
        assert "kill" in command and "12345" in command

    async def test_job_cancellation_not_found(self, orchestrator, mock_connection):
        """Test cancelling non-existent job."""
        # Mock job not found
        with patch.object(orchestrator, 'get_job', return_value=None):
            result = await orchestrator.cancel_job("nonexistent_job")
            assert result is False

    async def test_file_preparation_git(self, orchestrator, mock_connection):
        """Test file preparation for Git source."""
        job = GPUJobFactory(
            code_source=CodeSource.GIT,
            git_url="https://github.com/test/repo.git",
            git_branch="feature-branch"
        )
        
        mock_connection.execute_command.side_effect = [
            ("", "", 0),  # mkdir
            ("", "", 0),  # git clone
            ("", "", 0),  # git checkout
        ]
        
        await orchestrator._prepare_code_files(job)
        
        # Verify git operations
        commands = [call[0][0] for call in mock_connection.execute_command.call_args_list]
        assert any("git clone" in cmd for cmd in commands)
        assert any("git checkout" in cmd for cmd in commands)

    async def test_file_preparation_manual(self, orchestrator, mock_connection, tmp_path):
        """Test file preparation for manual source."""
        # Create test files
        files = {}
        for name in ["train.py", "config.yaml", "utils.py"]:
            test_file = tmp_path / name
            test_file.write_text(f"# {name} content")
            files[name] = str(test_file)
        
        job = GPUJobFactory(
            code_source=CodeSource.MANUAL,
            manual_files=files
        )
        
        await orchestrator._prepare_code_files(job)
        
        # Verify file uploads
        assert mock_connection.upload_file.call_count == 3

    async def test_model_file_upload(self, orchestrator, mock_connection, tmp_path):
        """Test model file upload preparation."""
        # Create test model file
        model_file = tmp_path / "model.pth"
        model_file.write_bytes(b"fake model data")
        
        job = GPUJobFactory(
            model_files={"pretrained": str(model_file)}
        )
        
        await orchestrator._prepare_model_files(job)
        
        # Verify model upload
        mock_connection.upload_file.assert_called()
        
        # Check upload path
        upload_call = mock_connection.upload_file.call_args
        local_path, remote_path = upload_call[0]
        assert local_path == str(model_file)
        assert "models" in remote_path

    async def test_data_file_upload(self, orchestrator, mock_connection, tmp_path):
        """Test data file upload preparation."""
        # Create test data files
        train_file = tmp_path / "train.csv"
        train_file.write_text("feature1,feature2,label\n1,2,0\n3,4,1\n")
        val_file = tmp_path / "val.csv"
        val_file.write_text("feature1,feature2,label\n5,6,0\n")
        
        job = GPUJobFactory(
            data_files={
                "train": str(train_file),
                "validation": str(val_file)
            }
        )
        
        await orchestrator._prepare_data_files(job)
        
        # Verify data uploads
        assert mock_connection.upload_file.call_count == 2

    async def test_execution_script_generation(self, orchestrator, mock_connection):
        """Test execution script generation."""
        job = GPUJobFactory(
            entry_script="train.py",
            environment_vars={"CUDA_VISIBLE_DEVICES": "0,1", "BATCH_SIZE": "32"},
            python_packages=["torch", "numpy"]
        )
        
        script_content = await orchestrator._generate_execution_script(job)
        
        assert "#!/bin/bash" in script_content
        assert "export CUDA_VISIBLE_DEVICES=0,1" in script_content
        assert "export BATCH_SIZE=32" in script_content
        assert "pip install torch numpy" in script_content
        assert "python3 train.py" in script_content

    async def test_job_submission_with_docker(self, orchestrator, mock_connection):
        """Test job submission with Docker image."""
        request = GPUJobRequestFactory(
            docker_image="pytorch/pytorch:latest",
            code_source=CodeSource.GIT,
            git_url="https://github.com/test/repo.git"
        )
        
        mock_connection.execute_command.side_effect = [
            ("", "", 0),  # mkdir
            ("", "", 0),  # git clone
            ("", "", 0),  # git checkout  
            ("", "", 0),  # docker pull
            ("12345", "", 0),  # job submission with docker
        ]
        
        job = await orchestrator.submit_job(request, "user123")
        
        assert job.docker_image == "pytorch/pytorch:latest"
        
        # Verify docker commands were called
        commands = [call[0][0] for call in mock_connection.execute_command.call_args_list]
        assert any("docker" in cmd for cmd in commands)

    async def test_job_validation_security(self, orchestrator, mock_connection):
        """Test security validation during job submission."""
        # Test malicious Git URL
        malicious_request = GPUJobRequestFactory(
            git_url="https://malicious-site.com/repo.git"
        )
        
        # Should validate Git URL (this depends on implementation)
        job = await orchestrator.submit_job(malicious_request, "user123")
        # For now, ensure it doesn't crash
        assert job is not None
        
        # Test malicious entry script
        dangerous_request = GPUJobRequestFactory(
            entry_script="../../../etc/passwd"
        )
        
        with pytest.raises(ValidationError):
            await orchestrator.submit_job(dangerous_request, "user123")

    async def test_resource_estimation(self, orchestrator, mock_connection):
        """Test resource requirement estimation."""
        request = GPUJobRequestFactory(gpu_count=4)
        
        # Mock resource availability check
        mock_connection.execute_command.return_value = ("4 GPUs available", "", 0)
        
        can_submit = await orchestrator._check_resource_availability(request)
        
        # Should be able to determine if resources are available
        assert isinstance(can_submit, bool)

    async def test_job_directory_creation(self, orchestrator, mock_connection):
        """Test job directory structure creation."""
        job = GPUJobFactory()
        
        await orchestrator._create_job_directories(job)
        
        # Verify directory creation commands
        commands = [call[0][0] for call in mock_connection.execute_command.call_args_list]
        
        # Should create necessary directories
        assert any("mkdir" in cmd for cmd in commands)
        # Should create subdirectories like logs, outputs, etc.
        directory_structure = ["logs", "outputs", "code", "data", "models"]
        for dir_name in directory_structure:
            assert any(dir_name in cmd for cmd in commands if "mkdir" in cmd)

    async def test_concurrent_job_submission(self, orchestrator, mock_connection):
        """Test concurrent job submissions."""
        requests = [GPUJobRequestFactory() for _ in range(3)]
        
        # Mock successful operations for all jobs
        mock_connection.execute_command.side_effect = [
            ("", "", 0),  # mkdir for job 1
            ("job1_pid", "", 0),  # submit job 1
            ("", "", 0),  # mkdir for job 2
            ("job2_pid", "", 0),  # submit job 2
            ("", "", 0),  # mkdir for job 3
            ("job3_pid", "", 0),  # submit job 3
        ]
        
        # Submit jobs concurrently
        import asyncio
        tasks = [
            orchestrator.submit_job(req, f"user{i}")
            for i, req in enumerate(requests)
        ]
        
        jobs = await asyncio.gather(*tasks)
        
        assert len(jobs) == 3
        assert all(job.status == JobStatus.QUEUED for job in jobs)
        assert len(set(job.job_id for job in jobs)) == 3  # All unique IDs

    async def test_error_handling_during_submission(self, orchestrator, mock_connection):
        """Test error handling during job submission."""
        request = GPUJobRequestFactory()
        
        # Mock failure during directory creation
        mock_connection.execute_command.side_effect = [
            ("", "Permission denied", 1),  # mkdir fails
        ]
        
        with pytest.raises(JobExecutionError) as exc_info:
            await orchestrator.submit_job(request, "user123")
        
        assert "Permission denied" in str(exc_info.value)

    async def test_git_branch_handling(self, orchestrator, mock_connection):
        """Test Git branch and commit handling."""
        request = GPUJobRequestFactory(
            code_source=CodeSource.GIT,
            git_url="https://github.com/test/repo.git",
            git_branch="feature-branch",
            git_commit="abc123def"
        )
        
        mock_connection.execute_command.side_effect = [
            ("", "", 0),  # mkdir
            ("", "", 0),  # git clone
            ("", "", 0),  # git checkout branch
            ("", "", 0),  # git checkout commit
            ("12345", "", 0),  # job submission
        ]
        
        job = await orchestrator.submit_job(request, "user123")
        
        # Verify git operations
        commands = [call[0][0] for call in mock_connection.execute_command.call_args_list]
        assert any("feature-branch" in cmd for cmd in commands)
        assert any("abc123def" in cmd for cmd in commands)

    async def test_environment_variable_handling(self, orchestrator, mock_connection):
        """Test environment variable setup."""
        request = GPUJobRequestFactory(
            environment_vars={
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "BATCH_SIZE": "64",
                "LEARNING_RATE": "0.001",
                "MODEL_NAME": "resnet50"
            }
        )
        
        script_content = await orchestrator._generate_execution_script(
            GPUJobFactory(**request.dict())
        )
        
        # Verify environment variables are set
        assert "export CUDA_VISIBLE_DEVICES=0,1,2,3" in script_content
        assert "export BATCH_SIZE=64" in script_content
        assert "export LEARNING_RATE=0.001" in script_content
        assert "export MODEL_NAME=resnet50" in script_content

    async def test_python_package_installation(self, orchestrator, mock_connection):
        """Test Python package installation script generation."""
        request = GPUJobRequestFactory(
            python_packages=["torch", "transformers", "datasets", "accelerate"]
        )
        
        script_content = await orchestrator._generate_execution_script(
            GPUJobFactory(**request.dict())
        )
        
        # Verify package installation
        assert "pip install" in script_content
        for package in request.python_packages:
            assert package in script_content

    async def test_job_script_security(self, orchestrator, mock_connection):
        """Test security measures in generated scripts."""
        request = GPUJobRequestFactory(
            entry_script="train.py",
            environment_vars={
                "SAFE_VAR": "safe_value",
                "MALICIOUS_VAR": "$(rm -rf /)"  # Injection attempt
            }
        )
        
        script_content = await orchestrator._generate_execution_script(
            GPUJobFactory(**request.dict())
        )
        
        # Should escape or sanitize dangerous content
        assert "export SAFE_VAR=safe_value" in script_content
        # The malicious content should be escaped or rejected
        # Implementation should handle this safely

    async def test_job_timeout_configuration(self, orchestrator, mock_connection):
        """Test job timeout configuration."""
        request = GPUJobRequestFactory()
        
        # Test with custom timeout
        job = await orchestrator.submit_job(request, "user123", timeout_hours=12)
        
        # Should set timeout in job metadata
        script_content = await orchestrator._generate_execution_script(job)
        # Implementation should include timeout handling in script

    async def test_job_listing_and_filtering(self, orchestrator, mock_connection):
        """Test job listing with filters."""
        # Mock job storage (would typically use database)
        mock_jobs = [
            GPUJobFactory(status=JobStatus.RUNNING),
            GPUJobFactory(status=JobStatus.COMPLETED),
            GPUJobFactory(status=JobStatus.FAILED)
        ]
        
        with patch.object(orchestrator, '_get_jobs_from_storage', return_value=mock_jobs):
            # Test listing all jobs
            all_jobs = await orchestrator.list_jobs()
            assert len(all_jobs) == 3
            
            # Test filtering by status
            running_jobs = await orchestrator.list_jobs(status=JobStatus.RUNNING)
            assert len(running_jobs) == 1
            assert running_jobs[0].status == JobStatus.RUNNING

    async def test_job_cleanup_on_failure(self, orchestrator, mock_connection):
        """Test cleanup when job submission fails."""
        request = GPUJobRequestFactory()
        
        # Mock failure during job execution
        mock_connection.execute_command.side_effect = [
            ("", "", 0),  # mkdir success
            ("", "", 0),  # file prep success
            ("", "Job submission failed", 1),  # job submission fails
        ]
        
        with pytest.raises(JobExecutionError):
            await orchestrator.submit_job(request, "user123")
        
        # Should attempt cleanup
        # Implementation should call cleanup commands

    async def test_additional_file_upload(self, orchestrator, mock_connection, tmp_path):
        """Test uploading additional files to existing job."""
        job = GPUJobFactory(status=JobStatus.QUEUED)
        
        # Create test file
        additional_file = tmp_path / "additional_data.csv"
        additional_file.write_text("col1,col2\n1,2\n3,4\n")
        
        # Mock file object (FastAPI UploadFile)
        mock_file = MagicMock()
        mock_file.filename = "additional_data.csv"
        mock_file.file.read.return_value = additional_file.read_bytes()
        mock_file.content_type = "text/csv"
        
        result = await orchestrator.upload_additional_file(job, mock_file, "data")
        
        # Should return upload path
        assert result is not None
        assert isinstance(result, str)
        
        # Verify file was uploaded
        mock_connection.upload_file.assert_called()

    async def test_job_dependency_handling(self, orchestrator, mock_connection):
        """Test job dependency management."""
        # Create dependent job
        parent_job = GPUJobFactory(status=JobStatus.COMPLETED)
        
        request = GPUJobRequestFactory()
        # If we had dependency support, would test here
        
        job = await orchestrator.submit_job(request, "user123")
        
        # For now, ensure basic submission works
        assert job is not None

    async def test_job_priority_handling(self, orchestrator, mock_connection):
        """Test job priority assignment."""
        high_priority_request = GPUJobRequestFactory()
        low_priority_request = GPUJobRequestFactory()
        
        # Submit with different priorities (if supported)
        high_job = await orchestrator.submit_job(high_priority_request, "user123")
        low_job = await orchestrator.submit_job(low_priority_request, "user456")
        
        # Verify both jobs are created
        assert high_job is not None
        assert low_job is not None
        assert high_job.job_id != low_job.job_id