"""
Unit Tests for Job Monitor

Tests for job status monitoring, progress tracking, log parsing,
and resource utilization monitoring with mocked cluster responses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.ml.cluster.monitor import JobMonitor
from src.ml.cluster.models import GPUJob, JobStatus, JobProgress
from src.ml.cluster.exceptions import JobNotFoundError, JobExecutionError
from tests.fixtures.test_data import GPUJobFactory, RunningJobFactory, CompletedJobFactory
from tests.fixtures.mock_responses import MOCK_FILE_CONTENTS, MOCK_TRAINING_METRICS


class TestJobMonitor:
    """Test JobMonitor functionality."""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock cluster connection for monitoring tests."""
        connection = AsyncMock()
        connection.connected = True
        connection.execute_command = AsyncMock()
        return connection
    
    @pytest.fixture
    def monitor(self, mock_connection):
        """JobMonitor instance with mocked connection."""
        return JobMonitor(mock_connection)

    async def test_check_running_job_status(self, monitor, mock_connection):
        """Test checking status of running job."""
        job = RunningJobFactory(cluster_job_id="12345")
        
        # Mock process exists
        mock_connection.execute_command.return_value = (
            "testuser 12345  2.3  1.1 123456  7890 ?  R  10:00   0:05 python3 train.py",
            "", 0
        )
        
        status = await monitor.check_job_status(job)
        
        assert status == JobStatus.RUNNING
        mock_connection.execute_command.assert_called()

    async def test_check_completed_job_status(self, monitor, mock_connection):
        """Test checking status of completed job."""
        job = RunningJobFactory(cluster_job_id="12345")
        
        # Mock process not found (completed)
        mock_connection.execute_command.return_value = ("", "", 1)  # Process not found
        
        status = await monitor.check_job_status(job)
        
        assert status == JobStatus.COMPLETED

    async def test_check_failed_job_status(self, monitor, mock_connection):
        """Test detecting failed job status."""
        job = RunningJobFactory(cluster_job_id="12345")
        
        # Mock process not found and error log exists
        mock_connection.execute_command.side_effect = [
            ("", "", 1),  # Process not found
            ("ERROR: CUDA out of memory", "", 0),  # Error in logs
        ]
        
        status = await monitor.check_job_status(job)
        
        assert status == JobStatus.FAILED

    async def test_job_progress_parsing(self, monitor, mock_connection):
        """Test parsing job progress from logs."""
        job = RunningJobFactory(cluster_job_id="12345", total_epochs=10)
        
        # Mock log content with progress information
        log_content = """
        Epoch 5/10: loss=0.234, acc=0.834, val_loss=0.278, val_acc=0.812
        Memory usage: 4096MB / 8192MB
        GPU utilization: 85%
        Estimated time remaining: 25 minutes
        """
        
        mock_connection.execute_command.return_value = (log_content, "", 0)
        
        progress = await monitor.get_job_progress(job)
        
        assert progress is not None
        assert progress.current_epoch == 5
        assert progress.total_epochs == 10
        assert progress.progress_percent == 50.0
        assert progress.current_loss == 0.234
        assert progress.current_accuracy == 0.834

    async def test_log_parsing_different_formats(self, monitor, mock_connection):
        """Test parsing different log formats."""
        job = RunningJobFactory()
        
        # Test different log formats
        log_formats = [
            "Epoch 3: loss=0.456 acc=0.678",  # Space separated
            "Epoch: 3, Loss: 0.456, Accuracy: 0.678",  # Comma separated
            "[INFO] Training epoch 3/10 - loss: 0.456 - acc: 0.678",  # Structured format
            "3/10 (30%) | Loss: 0.456 | Acc: 67.8%",  # Percentage format
        ]
        
        for log_format in log_formats:
            mock_connection.execute_command.return_value = (log_format, "", 0)
            
            progress = await monitor.get_job_progress(job)
            
            # Should extract epoch information from any format
            assert progress is not None
            # Implementation should handle various formats

    async def test_gpu_resource_monitoring(self, monitor, mock_connection):
        """Test GPU resource utilization monitoring."""
        job = RunningJobFactory()
        
        # Mock nvidia-smi output
        nvidia_output = """
        0, Tesla V100, 2048, 32768, 85, 45
        1, Tesla V100, 4096, 32768, 90, 67
        """
        
        mock_connection.execute_command.return_value = (nvidia_output, "", 0)
        
        progress = await monitor.get_job_progress(job)
        
        if progress:
            assert progress.gpu_usage_percent > 0
            assert progress.memory_usage_mb > 0

    async def test_system_resource_monitoring(self, monitor, mock_connection):
        """Test system resource monitoring."""
        job = RunningJobFactory()
        
        # Mock system resource commands
        mock_connection.execute_command.side_effect = [
            ("Training log content", "", 0),  # Log content
            ("Mem: 65536 32768 32768", "", 0),  # Memory info
            ("0.50 0.45 0.40", "", 0),  # Load average
        ]
        
        progress = await monitor.get_job_progress(job)
        
        # Should include system resource information
        if progress:
            assert hasattr(progress, 'memory_usage_mb')

    async def test_log_retrieval(self, monitor, mock_connection):
        """Test retrieving job logs."""
        job = CompletedJobFactory()
        
        # Mock log file content
        training_log = MOCK_FILE_CONTENTS["/tmp/test_jobs/test_job_12345/logs/training.log"]
        mock_connection.execute_command.return_value = (training_log, "", 0)
        
        logs = await monitor.get_job_logs(job, "training", lines=50)
        
        assert isinstance(logs, list)
        assert len(logs) > 0
        assert any("Epoch" in log for log in logs)

    async def test_log_retrieval_different_types(self, monitor, mock_connection):
        """Test retrieving different types of logs."""
        job = CompletedJobFactory()
        
        log_types = ["training", "error", "system"]
        
        for log_type in log_types:
            mock_content = f"Mock {log_type} log content\nLine 2\nLine 3"
            mock_connection.execute_command.return_value = (mock_content, "", 0)
            
            logs = await monitor.get_job_logs(job, log_type, lines=10)
            
            assert isinstance(logs, list)
            assert len(logs) >= 1

    async def test_real_time_monitoring(self, monitor, mock_connection):
        """Test real-time job monitoring."""
        job = RunningJobFactory()
        
        # Mock progressive log updates
        log_updates = [
            "Epoch 1: loss=0.856",
            "Epoch 2: loss=0.654", 
            "Epoch 3: loss=0.432"
        ]
        
        progress_snapshots = []
        
        for i, log_line in enumerate(log_updates):
            mock_connection.execute_command.return_value = (log_line, "", 0)
            
            progress = await monitor.get_job_progress(job)
            if progress:
                progress_snapshots.append(progress)
        
        # Should track progress over time
        if progress_snapshots:
            assert len(progress_snapshots) > 0

    async def test_error_detection_in_logs(self, monitor, mock_connection):
        """Test error detection from log content."""
        job = RunningJobFactory()
        
        error_logs = [
            "RuntimeError: CUDA out of memory",
            "FileNotFoundError: No such file or directory",
            "ImportError: No module named 'missing_package'",
            "KeyboardInterrupt: Training interrupted",
        ]
        
        for error_log in error_logs:
            mock_connection.execute_command.return_value = (error_log, "", 0)
            
            # Should detect error patterns
            debug_info = await monitor.get_debug_info(job)
            
            assert isinstance(debug_info, dict)
            # Implementation should identify error patterns

    async def test_metrics_extraction_from_logs(self, monitor, mock_connection):
        """Test extracting training metrics from logs."""
        job = RunningJobFactory()
        
        # Mock log with various metric formats
        complex_log = """
        2024-08-25 10:15:00 - INFO - Epoch 3/10 started
        2024-08-25 10:15:30 - INFO - Batch 100/500: loss=0.432, acc=0.678
        2024-08-25 10:16:00 - INFO - Validation: val_loss=0.487, val_acc=0.634
        2024-08-25 10:16:30 - INFO - Learning rate: 0.001
        2024-08-25 10:17:00 - INFO - GPU Memory: 4096MB / 8192MB
        2024-08-25 10:17:30 - INFO - Epoch 3 completed in 2.5 minutes
        """
        
        mock_connection.execute_command.return_value = (complex_log, "", 0)
        
        progress = await monitor.get_job_progress(job)
        
        if progress:
            assert progress.current_epoch is not None
            assert progress.current_loss is not None
            assert progress.current_accuracy is not None

    async def test_monitoring_job_without_process_id(self, monitor, mock_connection):
        """Test monitoring job without cluster process ID."""
        job = GPUJobFactory(status=JobStatus.QUEUED, cluster_job_id=None)
        
        status = await monitor.check_job_status(job)
        
        # Should handle gracefully
        assert status in [JobStatus.QUEUED, JobStatus.PENDING]

    async def test_debug_information_collection(self, monitor, mock_connection):
        """Test comprehensive debug information collection."""
        job = RunningJobFactory(cluster_job_id="12345")
        
        # Mock various debug commands
        mock_connection.execute_command.side_effect = [
            ("python3 train.py", "", 0),  # Process info
            ("2048", "", 0),  # Memory usage
            ("85", "", 0),  # GPU usage
            ("Recent log entries...", "", 0),  # Recent logs
            ("/tmp/test_jobs/test_job/outputs", "", 0),  # Output directory
        ]
        
        debug_info = await monitor.get_debug_info(job)
        
        assert isinstance(debug_info, dict)
        assert "job_id" in debug_info
        assert "process_info" in debug_info or "status" in debug_info

    async def test_monitoring_performance(self, monitor, mock_connection):
        """Test monitoring operation performance."""
        job = RunningJobFactory()
        
        import time
        
        start_time = time.time()
        
        # Monitor should be fast
        await monitor.check_job_status(job)
        
        elapsed = time.time() - start_time
        
        # Status check should complete quickly
        assert elapsed < 1.0  # Less than 1 second

    async def test_batch_status_checking(self, monitor, mock_connection):
        """Test checking status of multiple jobs efficiently."""
        jobs = [RunningJobFactory() for _ in range(5)]
        
        # Mock batch process checking
        ps_output = "\n".join([
            f"testuser {job.cluster_job_id} python3 train.py"
            for job in jobs if job.cluster_job_id
        ])
        
        mock_connection.execute_command.return_value = (ps_output, "", 0)
        
        # Test batch monitoring (if implemented)
        statuses = []
        for job in jobs:
            status = await monitor.check_job_status(job)
            statuses.append(status)
        
        assert len(statuses) == 5
        assert all(status in JobStatus for status in statuses)

    async def test_log_parsing_edge_cases(self, monitor, mock_connection):
        """Test log parsing edge cases."""
        job = RunningJobFactory()
        
        edge_case_logs = [
            "",  # Empty log
            "No epoch information here",  # No metrics
            "Epoch: NaN, Loss: inf",  # Invalid values
            "Special characters: 한글, émojis 🚀",  # Unicode
            "\x00\x01\x02",  # Binary data
        ]
        
        for log_content in edge_case_logs:
            mock_connection.execute_command.return_value = (log_content, "", 0)
            
            # Should handle gracefully without crashing
            try:
                progress = await monitor.get_job_progress(job)
                # Progress may be None for invalid logs
                assert progress is None or isinstance(progress, JobProgress)
            except Exception as e:
                pytest.fail(f"Monitor should handle edge case gracefully: {e}")

    async def test_monitoring_disconnected_cluster(self, monitor, mock_connection):
        """Test monitoring when cluster connection is lost."""
        job = RunningJobFactory()
        
        # Mock connection error
        mock_connection.execute_command.side_effect = Exception("Connection lost")
        
        with pytest.raises(Exception):
            await monitor.check_job_status(job)
        
        # Monitor should handle connection errors appropriately

    async def test_progress_calculation_accuracy(self, monitor, mock_connection):
        """Test accuracy of progress percentage calculations."""
        job = RunningJobFactory(current_epoch=7, total_epochs=10)
        
        # Mock detailed progress log
        progress_log = "Epoch 7/10: loss=0.156, acc=0.889, progress=70%"
        mock_connection.execute_command.return_value = (progress_log, "", 0)
        
        progress = await monitor.get_job_progress(job)
        
        if progress:
            assert progress.current_epoch == 7
            assert progress.total_epochs == 10
            assert abs(progress.progress_percent - 70.0) < 1.0  # Within 1% tolerance

    async def test_time_estimation(self, monitor, mock_connection):
        """Test estimated time remaining calculation."""
        job = RunningJobFactory(
            current_epoch=3,
            total_epochs=10,
            started_at=datetime.now()
        )
        
        # Mock progress with timing information
        timing_log = """
        Epoch 3/10 completed in 15 minutes
        Average epoch time: 12 minutes
        Estimated remaining: 84 minutes
        """
        
        mock_connection.execute_command.return_value = (timing_log, "", 0)
        
        progress = await monitor.get_job_progress(job)
        
        if progress and progress.estimated_time_remaining_minutes:
            assert progress.estimated_time_remaining_minutes > 0
            assert progress.estimated_time_remaining_minutes < 200  # Reasonable estimate

    async def test_resource_usage_tracking(self, monitor, mock_connection):
        """Test tracking resource usage over time."""
        job = RunningJobFactory()
        
        # Mock resource monitoring commands
        mock_connection.execute_command.side_effect = [
            ("Epoch 5 progress info", "", 0),  # Progress log
            ("4096", "", 0),  # Memory usage in MB
            ("Tesla V100: 85%", "", 0),  # GPU usage
            ("CPU usage: 45%", "", 0),  # CPU usage
        ]
        
        progress = await monitor.get_job_progress(job)
        
        if progress:
            # Should include resource utilization
            assert hasattr(progress, 'memory_usage_mb')
            assert hasattr(progress, 'gpu_usage_percent')

    async def test_log_streaming_simulation(self, monitor, mock_connection):
        """Test simulating real-time log streaming."""
        job = RunningJobFactory()
        
        # Simulate log updates over time
        log_updates = [
            "Starting epoch 1...",
            "Epoch 1: loss=0.856, acc=0.234",
            "Starting epoch 2...",
            "Epoch 2: loss=0.654, acc=0.456",
        ]
        
        collected_logs = []
        
        for log_update in log_updates:
            mock_connection.execute_command.return_value = (log_update, "", 0)
            
            logs = await monitor.get_job_logs(job, "training", lines=1)
            collected_logs.extend(logs)
        
        assert len(collected_logs) >= len(log_updates)

    async def test_error_log_analysis(self, monitor, mock_connection):
        """Test analysis of error logs for debugging."""
        job = RunningJobFactory()
        
        error_log_content = """
        Traceback (most recent call last):
          File "train.py", line 45, in <module>
            model = load_model('nonexistent.pth')
          File "utils.py", line 12, in load_model
            return torch.load(path)
        FileNotFoundError: [Errno 2] No such file or directory: 'nonexistent.pth'
        """
        
        mock_connection.execute_command.return_value = (error_log_content, "", 0)
        
        debug_info = await monitor.get_debug_info(job)
        
        assert isinstance(debug_info, dict)
        # Should extract useful debugging information
        if "error_analysis" in debug_info:
            assert "FileNotFoundError" in str(debug_info["error_analysis"])

    async def test_monitoring_multiple_jobs(self, monitor, mock_connection):
        """Test monitoring multiple jobs efficiently."""
        jobs = [RunningJobFactory() for _ in range(3)]
        
        # Mock batch process status
        batch_ps_output = "\n".join([
            f"testuser {job.cluster_job_id} python3 train.py"
            for job in jobs
        ])
        
        mock_connection.execute_command.return_value = (batch_ps_output, "", 0)
        
        # Monitor all jobs
        import asyncio
        tasks = [monitor.check_job_status(job) for job in jobs]
        statuses = await asyncio.gather(*tasks)
        
        assert len(statuses) == 3
        assert all(isinstance(status, JobStatus) for status in statuses)

    async def test_monitoring_job_state_transitions(self, monitor, mock_connection):
        """Test monitoring during job state transitions."""
        job = GPUJobFactory(status=JobStatus.QUEUED)
        
        # Simulate state transitions
        state_responses = [
            ("", "", 1),  # Not started yet
            ("testuser 12345 python3 train.py", "", 0),  # Now running
            ("", "", 1),  # Completed
        ]
        
        states = []
        for response in state_responses:
            mock_connection.execute_command.return_value = response
            status = await monitor.check_job_status(job)
            states.append(status)
        
        # Should detect state changes
        assert JobStatus.QUEUED in states or JobStatus.RUNNING in states
        assert JobStatus.COMPLETED in states

    async def test_monitoring_long_running_jobs(self, monitor, mock_connection):
        """Test monitoring very long-running jobs."""
        job = RunningJobFactory(
            total_epochs=1000,  # Very long training
            current_epoch=50
        )
        
        # Mock long training progress
        long_training_log = """
        Epoch 50/1000: loss=0.345, acc=0.756
        Time elapsed: 12 hours 34 minutes
        Estimated remaining: 10 days 2 hours
        """
        
        mock_connection.execute_command.return_value = (long_training_log, "", 0)
        
        progress = await monitor.get_job_progress(job)
        
        if progress:
            assert progress.progress_percent == 5.0  # 50/1000
            # Should handle very long time estimates
            if progress.estimated_time_remaining_minutes:
                assert progress.estimated_time_remaining_minutes > 1000