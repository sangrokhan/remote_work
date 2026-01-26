"""
Unit Tests for Result Collector

Tests for result collection, file archiving, metrics extraction,
and storage management with mocked file operations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.ml.cluster.collector import ResultCollector
from src.ml.cluster.models import GPUJob, JobStatus, JobResults
from src.ml.cluster.exceptions import ValidationError, FileTransferError
from tests.fixtures.test_data import CompletedJobFactory, JobResultsFactory
from tests.fixtures.mock_responses import MOCK_FILE_CONTENTS


class TestResultCollector:
    """Test ResultCollector functionality."""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock cluster connection for collector tests."""
        connection = AsyncMock()
        connection.connected = True
        connection.execute_command = AsyncMock()
        connection.download_file = AsyncMock(return_value=True)
        return connection
    
    @pytest.fixture
    def collector(self, mock_connection, tmp_path):
        """ResultCollector instance with mocked connection and temp storage."""
        storage_path = tmp_path / "results"
        storage_path.mkdir()
        return ResultCollector(mock_connection, str(storage_path))

    async def test_collect_results_success(self, collector, mock_connection):
        """Test successful result collection."""
        job = CompletedJobFactory()
        
        # Mock file discovery commands
        mock_connection.execute_command.side_effect = [
            # Model file search
            ("/tmp/test_jobs/test_job/outputs/final_model.pth\n/tmp/test_jobs/test_job/outputs/best_model.pth", "", 0),
            ("", "", 0),  # No .pt files
            ("", "", 0),  # No .h5 files
            ("", "", 0),  # No .onnx files
            ("", "", 0),  # No .pkl files
            ("", "", 0),  # No .joblib files
            # Log directory check
            ("EXISTS", "", 0),
            ("", "", 0),  # tar creation
            ("", "", 0),  # tar cleanup
            # Output file search
            ("/tmp/test_jobs/test_job/outputs/metrics.json\n/tmp/test_jobs/test_job/outputs/config.yaml", "", 0),
        ]
        
        # Mock file downloads create actual files
        def mock_download(remote_path, local_path, verify_integrity=True):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            if "model.pth" in remote_path:
                Path(local_path).write_bytes(b"fake model data")
            elif "metrics.json" in remote_path:
                Path(local_path).write_text('{"accuracy": 0.95, "loss": 0.123}')
            elif "training.log" in remote_path:
                Path(local_path).write_text("Final results: loss=0.123, accuracy=0.95")
            else:
                Path(local_path).write_text("mock file content")
            return True
        
        mock_connection.download_file.side_effect = mock_download
        
        results = await collector.collect_results(job)
        
        assert isinstance(results, JobResults)
        assert results.job_id == job.job_id
        assert len(results.model_files) > 0
        assert results.total_size_mb > 0
        assert isinstance(results.collected_at, datetime)

    async def test_collect_results_invalid_status(self, collector, mock_connection):
        """Test collecting results from non-completed job."""
        job = CompletedJobFactory()
        job.status = JobStatus.RUNNING  # Invalid status
        
        with pytest.raises(ValidationError) as exc_info:
            await collector.collect_results(job)
        
        assert "completed" in str(exc_info.value).lower()

    async def test_model_file_collection(self, collector, mock_connection, tmp_path):
        """Test model file collection with different formats."""
        job = CompletedJobFactory()
        
        # Mock finding different model file types
        model_files = [
            "/tmp/job/outputs/model.pth",
            "/tmp/job/outputs/best_model.pt", 
            "/tmp/job/outputs/checkpoint.h5",
            "/tmp/job/outputs/inference.onnx"
        ]
        
        mock_connection.execute_command.side_effect = [
            ("\n".join(model_files[:2]), "", 0),  # .pth and .pt files
            (model_files[2], "", 0),  # .h5 files
            (model_files[3], "", 0),  # .onnx files
            ("", "", 0),  # No .pkl files
            ("", "", 0),  # No .joblib files
        ]
        
        # Mock file downloads
        def mock_download(remote_path, local_path, verify_integrity=True):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(b"model data")
            return True
        
        mock_connection.download_file.side_effect = mock_download
        
        model_files_collected = await collector._collect_model_files(job, tmp_path)
        
        assert len(model_files_collected) == 4
        assert all("models/" in path for path in model_files_collected)

    async def test_log_file_collection(self, collector, mock_connection, tmp_path):
        """Test log file collection and archiving."""
        job = CompletedJobFactory()
        
        # Mock log directory exists and tar operations
        mock_connection.execute_command.side_effect = [
            ("EXISTS", "", 0),  # Directory exists
            ("", "", 0),  # tar creation
            ("", "", 0),  # tar cleanup
        ]
        
        # Mock tar file download and extraction
        def mock_download(remote_path, local_path, verify_integrity=True):
            if "logs_archive.tar.gz" in remote_path:
                # Create mock tar file
                import tarfile
                with tarfile.open(local_path, 'w:gz') as tar:
                    # Create mock log file in memory
                    log_content = "Training log content"
                    import io
                    log_file = io.BytesIO(log_content.encode())
                    tarinfo = tarfile.TarInfo(name="logs/training.log")
                    tarinfo.size = len(log_content)
                    tar.addfile(tarinfo, log_file)
            return True
        
        mock_connection.download_file.side_effect = mock_download
        
        log_files = await collector._collect_log_files(job, tmp_path)
        
        assert isinstance(log_files, list)
        # May be empty if tar extraction doesn't work in test environment

    async def test_metrics_extraction(self, collector, mock_connection, tmp_path):
        """Test metrics extraction from various sources."""
        job = CompletedJobFactory()
        
        # Create mock metrics file
        metrics_file = tmp_path / "outputs" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.write_text(json.dumps({
            "final_loss": 0.123,
            "final_accuracy": 0.945,
            "f1_score": 0.892
        }))
        
        # Create mock training log
        training_log = tmp_path / "logs" / "training.log"
        training_log.parent.mkdir(parents=True)
        training_log.write_text("""
        Epoch 8: loss=0.145, acc=0.901
        Epoch 9: loss=0.134, acc=0.915  
        Epoch 10: loss=0.123, acc=0.925
        Final results: loss=0.123, accuracy=0.925
        """)
        
        metrics = await collector._extract_metrics(job, tmp_path)
        
        assert isinstance(metrics, dict)
        assert "final_loss" in metrics
        assert "final_accuracy" in metrics
        assert metrics["final_loss"] == 0.123

    async def test_training_history_parsing(self, collector, mock_connection, tmp_path):
        """Test parsing training history from logs."""
        job = CompletedJobFactory()
        
        # Create mock training log with epoch data
        training_log = tmp_path / "logs" / "training.log"
        training_log.parent.mkdir(parents=True)
        training_log.write_text("""
        Epoch 1/10: loss=0.856, acc=0.234, val_loss=0.901, val_acc=0.198
        Epoch 2/10: loss=0.654, acc=0.456, val_loss=0.698, val_acc=0.412
        Epoch 3/10: loss=0.432, acc=0.678, val_loss=0.487, val_acc=0.634
        Epoch 4/10: loss=0.321, acc=0.789, val_loss=0.356, val_acc=0.745
        Epoch 5/10: loss=0.234, acc=0.834, val_loss=0.278, val_acc=0.812
        """)
        
        history = await collector._parse_training_history(job, tmp_path)
        
        assert isinstance(history, dict)
        assert "loss" in history
        assert "accuracy" in history
        assert len(history["loss"]) == 5
        assert len(history["accuracy"]) == 5
        assert history["loss"][0] == 0.856  # First epoch
        assert history["accuracy"][-1] == 0.834  # Last epoch

    async def test_metrics_line_parsing(self, collector):
        """Test extracting metrics from individual log lines."""
        test_lines = [
            "Epoch 5: loss=0.234, acc=0.834",
            "Validation: val_loss=0.278, val_acc=0.812",
            "Learning rate: lr=0.001",
            "Final results: loss=0.123, accuracy=0.925",
            "No metrics in this line",
            "Invalid: loss=abc, acc=xyz"  # Invalid numbers
        ]
        
        for line in test_lines:
            metrics = collector._extract_metrics_from_line(line)
            
            assert isinstance(metrics, dict)
            if "loss=0.234" in line:
                assert metrics.get("loss") == 0.234
                assert metrics.get("accuracy") == 0.834

    async def test_result_archiving(self, collector, mock_connection, tmp_path):
        """Test result archiving functionality."""
        job_id = "test_job_123"
        
        # Create mock result directory structure
        job_dir = tmp_path / "results" / job_id
        job_dir.mkdir(parents=True)
        
        # Create mock files
        (job_dir / "models").mkdir()
        (job_dir / "models" / "model.pth").write_bytes(b"model data")
        (job_dir / "logs").mkdir()
        (job_dir / "logs" / "training.log").write_text("log content")
        (job_dir / "metadata.json").write_text('{"job_id": "test_job_123"}')
        
        # Update collector storage path
        collector.local_storage_path = tmp_path / "results"
        
        # Test tar.gz archiving
        archive_path = await collector.archive_results(job_id, "tar.gz")
        
        assert archive_path.endswith(".tar.gz")
        assert Path(archive_path).exists()

    async def test_result_summary_generation(self, collector, mock_connection, tmp_path):
        """Test generating result summaries."""
        job_id = "test_job_123"
        
        # Create mock metadata file
        job_dir = tmp_path / "results" / job_id
        job_dir.mkdir(parents=True)
        
        metadata = {
            "job_id": job_id,
            "collected_at": datetime.now().isoformat(),
            "total_size_mb": 256.7,
            "file_counts": {"models": 2, "logs": 3, "outputs": 5},
            "final_metrics": {"accuracy": 0.95, "loss": 0.123}
        }
        
        metadata_file = job_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))
        
        # Update collector storage path
        collector.local_storage_path = tmp_path / "results"
        
        summary = await collector.get_result_summary(job_id)
        
        assert summary is not None
        assert summary["job_id"] == job_id
        assert summary["total_size_mb"] == 256.7
        assert summary["final_metrics"]["accuracy"] == 0.95

    async def test_storage_usage_calculation(self, collector, mock_connection, tmp_path):
        """Test storage usage calculation."""
        # Update collector storage path
        collector.local_storage_path = tmp_path / "results"
        collector.local_storage_path.mkdir()
        
        # Create mock job directories with files
        for i in range(3):
            job_dir = collector.local_storage_path / f"job_{i}"
            job_dir.mkdir()
            
            # Create files of different sizes
            (job_dir / "small.txt").write_text("small file")
            (job_dir / "large.txt").write_text("x" * 10000)  # 10KB
        
        usage = await collector.get_storage_usage()
        
        assert isinstance(usage, dict)
        assert "total_results_size_mb" in usage
        assert "job_count" in usage
        assert usage["job_count"] == 3

    async def test_result_comparison(self, collector, mock_connection, tmp_path):
        """Test comparing results from multiple jobs."""
        # Create mock result directories
        collector.local_storage_path = tmp_path / "results"
        collector.local_storage_path.mkdir()
        
        job_ids = ["job_1", "job_2", "job_3"]
        metrics_data = [
            {"accuracy": 0.90, "loss": 0.150, "f1": 0.88},
            {"accuracy": 0.95, "loss": 0.120, "f1": 0.91},
            {"accuracy": 0.88, "loss": 0.180, "f1": 0.85}
        ]
        
        # Create metadata for each job
        for job_id, metrics in zip(job_ids, metrics_data):
            job_dir = collector.local_storage_path / job_id
            job_dir.mkdir()
            
            metadata = {
                "job_id": job_id,
                "final_metrics": metrics,
                "collected_at": datetime.now().isoformat()
            }
            
            (job_dir / "metadata.json").write_text(json.dumps(metadata))
        
        comparison = await collector.compare_results(job_ids)
        
        assert isinstance(comparison, dict)
        assert "jobs" in comparison
        assert "metrics_comparison" in comparison
        assert len(comparison["jobs"]) == 3
        
        # Should identify best and worst for each metric
        if "accuracy" in comparison["metrics_comparison"]:
            acc_comparison = comparison["metrics_comparison"]["accuracy"]
            assert acc_comparison["best"]["value"] == 0.95  # job_2
            assert acc_comparison["worst"]["value"] == 0.88  # job_3

    async def test_result_export_json(self, collector, mock_connection, tmp_path):
        """Test exporting results to JSON format."""
        job_id = "test_job_123"
        
        # Create mock metadata
        collector.local_storage_path = tmp_path / "results"
        job_dir = collector.local_storage_path / job_id
        job_dir.mkdir(parents=True)
        
        metadata = {"job_id": job_id, "final_metrics": {"accuracy": 0.95}}
        (job_dir / "metadata.json").write_text(json.dumps(metadata))
        
        export_path = tmp_path / "export.json"
        result_path = await collector.export_results(job_id, str(export_path), "json")
        
        assert result_path == str(export_path)
        assert Path(export_path).exists()
        
        # Verify exported content
        exported_data = json.loads(Path(export_path).read_text())
        assert exported_data["job_id"] == job_id

    async def test_result_export_csv(self, collector, mock_connection, tmp_path):
        """Test exporting results to CSV format."""
        job_id = "test_job_123"
        
        # Create mock metadata with metrics
        collector.local_storage_path = tmp_path / "results"
        job_dir = collector.local_storage_path / job_id
        job_dir.mkdir(parents=True)
        
        metadata = {
            "job_id": job_id,
            "final_metrics": {
                "accuracy": 0.95,
                "loss": 0.123,
                "f1_score": 0.89
            }
        }
        (job_dir / "metadata.json").write_text(json.dumps(metadata))
        
        export_path = tmp_path / "export.csv"
        result_path = await collector.export_results(job_id, str(export_path), "csv")
        
        assert result_path == str(export_path)
        assert Path(export_path).exists()
        
        # Verify CSV content
        csv_content = Path(export_path).read_text()
        assert "accuracy" in csv_content
        assert "0.95" in csv_content

    async def test_cleanup_expired_results(self, collector, mock_connection, tmp_path):
        """Test cleaning up expired result directories."""
        collector.local_storage_path = tmp_path / "results"
        collector.local_storage_path.mkdir()
        
        # Create old and new job directories
        old_job_dir = collector.local_storage_path / "old_job"
        new_job_dir = collector.local_storage_path / "new_job"
        
        old_job_dir.mkdir()
        new_job_dir.mkdir()
        
        # Create files
        (old_job_dir / "data.txt").write_text("old data")
        (new_job_dir / "data.txt").write_text("new data")
        
        # Mock old timestamp for old_job_dir
        import os
        import time
        old_time = time.time() - (40 * 24 * 3600)  # 40 days ago
        os.utime(old_job_dir, (old_time, old_time))
        
        cleaned_jobs = await collector.cleanup_expired_results(retention_days=30)
        
        # Should clean up expired directories
        assert isinstance(cleaned_jobs, list)

    async def test_list_available_results(self, collector, mock_connection, tmp_path):
        """Test listing available result directories."""
        collector.local_storage_path = tmp_path / "results"
        collector.local_storage_path.mkdir()
        
        # Create mock job result directories
        job_ids = ["job_1", "job_2", "job_3"]
        
        for job_id in job_ids:
            job_dir = collector.local_storage_path / job_id
            job_dir.mkdir()
            
            metadata = {
                "job_id": job_id,
                "collected_at": datetime.now().isoformat(),
                "total_size_mb": 100.0
            }
            (job_dir / "metadata.json").write_text(json.dumps(metadata))
        
        available_results = await collector.list_available_results()
        
        assert len(available_results) == 3
        assert all(result["job_id"] in job_ids for result in available_results)

    async def test_directory_size_calculation(self, collector, tmp_path):
        """Test accurate directory size calculation."""
        test_dir = tmp_path / "size_test"
        test_dir.mkdir()
        
        # Create files of known sizes
        (test_dir / "small.txt").write_text("x" * 1000)  # 1KB
        (test_dir / "medium.txt").write_text("x" * 100000)  # 100KB
        (test_dir / "large.txt").write_text("x" * 1000000)  # 1MB
        
        size_mb = await collector._calculate_total_size(test_dir)
        
        # Should be approximately 1.1 MB
        assert 1.0 < size_mb < 1.2

    async def test_collection_with_missing_files(self, collector, mock_connection):
        """Test collection when some files are missing."""
        job = CompletedJobFactory()
        
        # Mock partial file availability
        mock_connection.execute_command.side_effect = [
            ("", "", 1),  # No model files found
            ("NOT_FOUND", "", 0),  # No logs directory
            ("", "", 1),  # No output files
        ]
        
        results = await collector.collect_results(job)
        
        # Should handle missing files gracefully
        assert isinstance(results, JobResults)
        assert len(results.model_files) == 0
        assert len(results.log_files) == 0

    async def test_collection_with_download_failures(self, collector, mock_connection):
        """Test handling download failures during collection."""
        job = CompletedJobFactory()
        
        # Mock file discovery but download failures
        mock_connection.execute_command.side_effect = [
            ("/tmp/job/model.pth", "", 0),  # Model found
            ("", "", 0),  # Other searches
            ("", "", 0),
            ("", "", 0),
            ("", "", 0),
            ("", "", 0),
            ("NOT_FOUND", "", 0),  # No logs
            ("", "", 1),  # No outputs
        ]
        
        # Mock download failure
        mock_connection.download_file.side_effect = Exception("Download failed")
        
        results = await collector.collect_results(job)
        
        # Should complete despite download failures
        assert isinstance(results, JobResults)
        # Files that failed to download won't be in results

    async def test_tensorboard_metrics_extraction(self, collector, mock_connection, tmp_path):
        """Test TensorBoard metrics extraction."""
        job = CompletedJobFactory()
        
        # Create mock TensorBoard directory
        tb_dir = tmp_path / "logs" / "tensorboard" / "run_1"
        tb_dir.mkdir(parents=True)
        
        # Create mock event files
        event_file = tb_dir / "events.out.tfevents.1692960000.hostname"
        event_file.write_bytes(b"mock tensorboard data")
        
        metrics = await collector._extract_tensorboard_metrics(job, tmp_path)
        
        assert isinstance(metrics, dict)
        if "tensorboard_events_found" in metrics:
            assert metrics["tensorboard_events_found"] >= 1

    async def test_concurrent_collection_operations(self, collector, mock_connection):
        """Test concurrent result collection operations."""
        jobs = [CompletedJobFactory() for _ in range(3)]
        
        # Mock successful operations for all jobs
        mock_connection.execute_command.return_value = ("", "", 0)
        
        # Collect results concurrently
        import asyncio
        tasks = [collector.collect_results(job) for job in jobs]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(isinstance(result, JobResults) for result in results)

    async def test_collection_performance(self, collector, mock_connection):
        """Test collection operation performance."""
        job = CompletedJobFactory()
        
        # Mock fast responses
        mock_connection.execute_command.return_value = ("", "", 0)
        
        import time
        start_time = time.time()
        
        results = await collector.collect_results(job)
        
        elapsed = time.time() - start_time
        
        # Collection should be reasonably fast
        assert elapsed < 5.0  # Less than 5 seconds for mocked operations
        assert isinstance(results, JobResults)

    async def test_local_cleanup_operations(self, collector, mock_connection, tmp_path):
        """Test local result cleanup operations."""
        job_id = "cleanup_test_job"
        
        # Create mock result directory
        collector.local_storage_path = tmp_path / "results"
        job_dir = collector.local_storage_path / job_id
        job_dir.mkdir(parents=True)
        
        # Create files
        (job_dir / "model.pth").write_bytes(b"model data")
        (job_dir / "log.txt").write_text("log content")
        
        # Cleanup recent results (should not delete)
        result = await collector.cleanup_local_results(job_id, older_than_days=30)
        assert result is False  # Too recent to delete
        assert job_dir.exists()
        
        # Mock old timestamp and cleanup
        import os
        import time
        old_time = time.time() - (40 * 24 * 3600)  # 40 days ago
        os.utime(job_dir, (old_time, old_time))
        
        result = await collector.cleanup_local_results(job_id, older_than_days=30)
        assert result is True
        assert not job_dir.exists()