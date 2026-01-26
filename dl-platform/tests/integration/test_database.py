"""
Integration Tests for Database Repositories

Tests for database operations, repository pattern implementation,
and data persistence with real database transactions.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.repositories import (
    GPUJobRepository, JobResultRepository, JobMetricRepository,
    ClusterConfigRepository, UserRepository
)
from src.core.db_models import GPUJobDB, User, JobResultDB, ClusterConfigDB
from src.ml.cluster.models import GPUJob, JobStatus, CodeSource, JobResults
from tests.fixtures.test_data import GPUJobFactory, JobResultsFactory, TEST_USERS


class TestGPUJobRepository:
    """Test GPU job repository operations."""
    
    @pytest.fixture
    def job_repo(self, test_session: AsyncSession):
        """GPU job repository instance."""
        return GPUJobRepository(test_session)
    
    @pytest.fixture
    async def test_user(self, test_session: AsyncSession):
        """Create test user for job operations."""
        user_repo = UserRepository(test_session)
        user_data = TEST_USERS[0].copy()
        user = await user_repo.create_user(user_data)
        return user

    async def test_create_job(self, job_repo, test_user):
        """Test creating a new job in database."""
        job = GPUJobFactory(
            job_name="test_create_job",
            gpu_count=2,
            status=JobStatus.PENDING
        )
        
        db_job = await job_repo.create_job(job, test_user.id)
        
        assert db_job.job_id == job.job_id
        assert db_job.job_name == "test_create_job"
        assert db_job.gpu_count == 2
        assert db_job.user_id == test_user.id
        assert db_job.status == JobStatus.PENDING.value

    async def test_get_job(self, job_repo, test_user):
        """Test retrieving job from database."""
        # Create job first
        job = GPUJobFactory()
        db_job = await job_repo.create_job(job, test_user.id)
        
        # Retrieve job
        retrieved_job = await job_repo.get_job(job.job_id)
        
        assert retrieved_job is not None
        assert retrieved_job.job_id == job.job_id
        assert retrieved_job.user.username == test_user.username

    async def test_get_nonexistent_job(self, job_repo):
        """Test retrieving non-existent job."""
        result = await job_repo.get_job("nonexistent_job_id")
        assert result is None

    async def test_update_job_status(self, job_repo, test_user):
        """Test updating job status."""
        # Create job
        job = GPUJobFactory(status=JobStatus.PENDING)
        await job_repo.create_job(job, test_user.id)
        
        # Update status
        success = await job_repo.update_job_status(
            job.job_id, 
            JobStatus.RUNNING,
            cluster_job_id="12345",
            started_at=datetime.utcnow()
        )
        
        assert success is True
        
        # Verify update
        updated_job = await job_repo.get_job(job.job_id)
        assert updated_job.status == JobStatus.RUNNING.value
        assert updated_job.cluster_job_id == "12345"
        assert updated_job.started_at is not None

    async def test_update_job_progress(self, job_repo, test_user):
        """Test updating job progress information."""
        # Create running job
        job = GPUJobFactory(status=JobStatus.RUNNING)
        await job_repo.create_job(job, test_user.id)
        
        # Update progress
        progress_data = {
            "current_epoch": 5,
            "total_epochs": 10,
            "current_loss": 0.234,
            "current_accuracy": 0.834,
            "memory_usage_mb": 4096.0,
            "gpu_usage_percent": 85.5
        }
        
        success = await job_repo.update_job_progress(job.job_id, progress_data)
        assert success is True
        
        # Verify update
        updated_job = await job_repo.get_job(job.job_id)
        assert updated_job.current_epoch == 5
        assert updated_job.current_loss == 0.234
        assert updated_job.gpu_usage_percent == 85.5

    async def test_list_jobs_basic(self, job_repo, test_user):
        """Test listing jobs without filters."""
        # Create multiple jobs
        jobs = [GPUJobFactory() for _ in range(3)]
        for job in jobs:
            await job_repo.create_job(job, test_user.id)
        
        # List jobs
        retrieved_jobs = await job_repo.list_jobs(limit=10)
        
        assert len(retrieved_jobs) == 3
        assert all(job.user_id == test_user.id for job in retrieved_jobs)

    async def test_list_jobs_with_status_filter(self, job_repo, test_user):
        """Test listing jobs filtered by status."""
        # Create jobs with different statuses
        pending_job = GPUJobFactory(status=JobStatus.PENDING)
        running_job = GPUJobFactory(status=JobStatus.RUNNING)
        completed_job = GPUJobFactory(status=JobStatus.COMPLETED)
        
        await job_repo.create_job(pending_job, test_user.id)
        await job_repo.create_job(running_job, test_user.id)
        await job_repo.create_job(completed_job, test_user.id)
        
        # Filter by running status
        running_jobs = await job_repo.list_jobs(status=JobStatus.RUNNING)
        
        assert len(running_jobs) == 1
        assert running_jobs[0].status == JobStatus.RUNNING.value

    async def test_list_jobs_with_user_filter(self, job_repo, test_session):
        """Test listing jobs filtered by user."""
        # Create multiple users
        user_repo = UserRepository(test_session)
        user1 = await user_repo.create_user(TEST_USERS[0])
        user2 = await user_repo.create_user(TEST_USERS[1])
        
        # Create jobs for different users
        job1 = GPUJobFactory()
        job2 = GPUJobFactory()
        job3 = GPUJobFactory()
        
        await job_repo.create_job(job1, user1.id)
        await job_repo.create_job(job2, user1.id)
        await job_repo.create_job(job3, user2.id)
        
        # Filter by user1
        user1_jobs = await job_repo.list_jobs(user_id=user1.id)
        
        assert len(user1_jobs) == 2
        assert all(job.user_id == user1.id for job in user1_jobs)

    async def test_list_jobs_pagination(self, job_repo, test_user):
        """Test job listing pagination."""
        # Create many jobs
        jobs = [GPUJobFactory() for _ in range(10)]
        for job in jobs:
            await job_repo.create_job(job, test_user.id)
        
        # Test pagination
        page1 = await job_repo.list_jobs(limit=5, offset=0)
        page2 = await job_repo.list_jobs(limit=5, offset=5)
        
        assert len(page1) == 5
        assert len(page2) == 5
        
        # Ensure no overlap
        page1_ids = {job.job_id for job in page1}
        page2_ids = {job.job_id for job in page2}
        assert page1_ids.isdisjoint(page2_ids)

    async def test_get_user_job_count(self, job_repo, test_user):
        """Test getting user job count."""
        # Create jobs with different statuses
        jobs = [
            GPUJobFactory(status=JobStatus.PENDING),
            GPUJobFactory(status=JobStatus.RUNNING),
            GPUJobFactory(status=JobStatus.COMPLETED),
            GPUJobFactory(status=JobStatus.COMPLETED)
        ]
        
        for job in jobs:
            await job_repo.create_job(job, test_user.id)
        
        # Test total count
        total_count = await job_repo.get_user_job_count(test_user.id)
        assert total_count == 4
        
        # Test count by status
        completed_count = await job_repo.get_user_job_count(test_user.id, JobStatus.COMPLETED)
        assert completed_count == 2

    async def test_get_active_gpu_usage(self, job_repo, test_user):
        """Test calculating active GPU usage."""
        # Create jobs using different GPU counts
        jobs = [
            GPUJobFactory(status=JobStatus.RUNNING, gpu_count=2),
            GPUJobFactory(status=JobStatus.QUEUED, gpu_count=4),
            GPUJobFactory(status=JobStatus.COMPLETED, gpu_count=8),  # Should not count
            GPUJobFactory(status=JobStatus.RUNNING, gpu_count=1)
        ]
        
        for job in jobs:
            await job_repo.create_job(job, test_user.id)
        
        active_gpus = await job_repo.get_active_gpu_usage()
        
        # Should count running (2+1) + queued (4) = 7 GPUs
        assert active_gpus == 7

    async def test_delete_job(self, job_repo, test_user):
        """Test deleting job from database."""
        job = GPUJobFactory()
        await job_repo.create_job(job, test_user.id)
        
        # Verify job exists
        retrieved = await job_repo.get_job(job.job_id)
        assert retrieved is not None
        
        # Delete job
        success = await job_repo.delete_job(job.job_id)
        assert success is True
        
        # Verify job is deleted
        deleted_job = await job_repo.get_job(job.job_id)
        assert deleted_job is None


class TestJobResultRepository:
    """Test job result repository operations."""
    
    @pytest.fixture
    def result_repo(self, test_session: AsyncSession):
        """Job result repository instance."""
        return JobResultRepository(test_session)
    
    @pytest.fixture
    async def test_job_with_user(self, test_session: AsyncSession):
        """Create test job with user for result operations."""
        user_repo = UserRepository(test_session)
        job_repo = GPUJobRepository(test_session)
        
        user = await user_repo.create_user(TEST_USERS[0])
        job = GPUJobFactory(status=JobStatus.COMPLETED)
        db_job = await job_repo.create_job(job, user.id)
        
        return db_job

    async def test_create_result(self, result_repo, test_job_with_user):
        """Test creating job result record."""
        results = JobResultsFactory(job_id=test_job_with_user.job_id)
        
        db_result = await result_repo.create_result(results)
        
        assert db_result.job_id == test_job_with_user.job_id
        assert len(db_result.model_files) > 0
        assert db_result.total_size_mb > 0

    async def test_get_result(self, result_repo, test_job_with_user):
        """Test retrieving job result."""
        # Create result first
        results = JobResultsFactory(job_id=test_job_with_user.job_id)
        await result_repo.create_result(results)
        
        # Retrieve result
        retrieved = await result_repo.get_result(test_job_with_user.job_id)
        
        assert retrieved is not None
        assert retrieved.job_id == test_job_with_user.job_id
        assert retrieved.job is not None  # Should load relationship

    async def test_update_result(self, result_repo, test_job_with_user):
        """Test updating job result."""
        # Create result
        results = JobResultsFactory(job_id=test_job_with_user.job_id)
        await result_repo.create_result(results)
        
        # Update result
        update_data = {
            "total_size_mb": 2048.0,
            "archive_path": "/archives/job_archive.tar.gz"
        }
        
        success = await result_repo.update_result(test_job_with_user.job_id, update_data)
        assert success is True
        
        # Verify update
        updated = await result_repo.get_result(test_job_with_user.job_id)
        assert updated.total_size_mb == 2048.0
        assert updated.archive_path == "/archives/job_archive.tar.gz"

    async def test_list_results(self, result_repo, test_session):
        """Test listing job results."""
        # Create multiple jobs and results
        user_repo = UserRepository(test_session)
        job_repo = GPUJobRepository(test_session)
        
        user = await user_repo.create_user(TEST_USERS[0])
        
        for i in range(5):
            job = GPUJobFactory()
            await job_repo.create_job(job, user.id)
            
            results = JobResultsFactory(job_id=job.job_id)
            await result_repo.create_result(results)
        
        # List results
        all_results = await result_repo.list_results(limit=10)
        
        assert len(all_results) == 5
        assert all(result.job is not None for result in all_results)

    async def test_delete_result(self, result_repo, test_job_with_user):
        """Test deleting job result."""
        # Create result
        results = JobResultsFactory(job_id=test_job_with_user.job_id)
        await result_repo.create_result(results)
        
        # Verify exists
        existing = await result_repo.get_result(test_job_with_user.job_id)
        assert existing is not None
        
        # Delete result
        success = await result_repo.delete_result(test_job_with_user.job_id)
        assert success is True
        
        # Verify deletion
        deleted = await result_repo.get_result(test_job_with_user.job_id)
        assert deleted is None


class TestJobMetricRepository:
    """Test job metric repository operations."""
    
    @pytest.fixture
    def metric_repo(self, test_session: AsyncSession):
        """Job metric repository instance."""
        return JobMetricRepository(test_session)
    
    @pytest.fixture
    async def test_job_with_user(self, test_session: AsyncSession):
        """Create test job for metric operations."""
        user_repo = UserRepository(test_session)
        job_repo = GPUJobRepository(test_session)
        
        user = await user_repo.create_user(TEST_USERS[0])
        job = GPUJobFactory()
        db_job = await job_repo.create_job(job, user.id)
        
        return db_job

    async def test_add_metric(self, metric_repo, test_job_with_user):
        """Test adding job metric."""
        metric = await metric_repo.add_metric(
            job_id=test_job_with_user.job_id,
            metric_name="accuracy",
            value=0.85,
            epoch=5,
            metric_type="training"
        )
        
        assert metric.job_id == test_job_with_user.job_id
        assert metric.metric_name == "accuracy"
        assert metric.metric_value == 0.85
        assert metric.epoch == 5

    async def test_get_job_metrics(self, metric_repo, test_job_with_user):
        """Test retrieving job metrics."""
        # Add multiple metrics
        metrics_data = [
            ("loss", 0.456, 1),
            ("accuracy", 0.678, 1),
            ("loss", 0.234, 2),
            ("accuracy", 0.834, 2)
        ]
        
        for name, value, epoch in metrics_data:
            await metric_repo.add_metric(
                test_job_with_user.job_id, name, value, epoch=epoch
            )
        
        # Get all metrics
        all_metrics = await metric_repo.get_job_metrics(test_job_with_user.job_id)
        assert len(all_metrics) == 4
        
        # Get specific metric
        loss_metrics = await metric_repo.get_job_metrics(
            test_job_with_user.job_id, "loss"
        )
        assert len(loss_metrics) == 2
        assert all(m.metric_name == "loss" for m in loss_metrics)

    async def test_get_metric_history(self, metric_repo, test_job_with_user):
        """Test retrieving metric history."""
        # Add metric history
        for epoch in range(1, 6):
            loss_value = 1.0 - (epoch * 0.15)  # Decreasing loss
            await metric_repo.add_metric(
                test_job_with_user.job_id, "loss", loss_value, epoch=epoch
            )
        
        history = await metric_repo.get_metric_history(test_job_with_user.job_id, "loss")
        
        assert len(history) == 5
        assert history[0]["epoch"] == 1
        assert history[-1]["epoch"] == 5
        # Loss should be decreasing
        assert history[0]["value"] > history[-1]["value"]

    async def test_delete_job_metrics(self, metric_repo, test_job_with_user):
        """Test deleting all metrics for a job."""
        # Add metrics
        for i in range(3):
            await metric_repo.add_metric(
                test_job_with_user.job_id, f"metric_{i}", float(i), epoch=i+1
            )
        
        # Verify metrics exist
        metrics = await metric_repo.get_job_metrics(test_job_with_user.job_id)
        assert len(metrics) == 3
        
        # Delete metrics
        success = await metric_repo.delete_job_metrics(test_job_with_user.job_id)
        assert success is True
        
        # Verify deletion
        deleted_metrics = await metric_repo.get_job_metrics(test_job_with_user.job_id)
        assert len(deleted_metrics) == 0


class TestUserRepository:
    """Test user repository operations."""
    
    @pytest.fixture
    def user_repo(self, test_session: AsyncSession):
        """User repository instance."""
        return UserRepository(test_session)

    async def test_create_user(self, user_repo):
        """Test creating a new user."""
        user_data = TEST_USERS[0].copy()
        
        user = await user_repo.create_user(user_data)
        
        assert user.username == user_data["username"]
        assert user.email == user_data["email"]
        assert user.is_active is True
        assert user.created_at is not None

    async def test_get_user(self, user_repo):
        """Test retrieving user by ID."""
        # Create user first
        user_data = TEST_USERS[0].copy()
        created_user = await user_repo.create_user(user_data)
        
        # Retrieve user
        retrieved_user = await user_repo.get_user(created_user.id)
        
        assert retrieved_user is not None
        assert retrieved_user.id == created_user.id
        assert retrieved_user.username == user_data["username"]

    async def test_get_user_by_username(self, user_repo):
        """Test retrieving user by username."""
        user_data = TEST_USERS[0].copy()
        created_user = await user_repo.create_user(user_data)
        
        retrieved_user = await user_repo.get_user_by_username(user_data["username"])
        
        assert retrieved_user is not None
        assert retrieved_user.id == created_user.id
        assert retrieved_user.username == user_data["username"]

    async def test_update_user(self, user_repo):
        """Test updating user information."""
        # Create user
        user_data = TEST_USERS[0].copy()
        user = await user_repo.create_user(user_data)
        
        # Update user
        update_data = {
            "full_name": "Updated Full Name",
            "is_active": False
        }
        
        success = await user_repo.update_user(user.id, update_data)
        assert success is True
        
        # Verify update
        updated_user = await user_repo.get_user(user.id)
        assert updated_user.full_name == "Updated Full Name"
        assert updated_user.is_active is False
        assert updated_user.updated_at > updated_user.created_at


class TestClusterConfigRepository:
    """Test cluster configuration repository operations."""
    
    @pytest.fixture
    def config_repo(self, test_session: AsyncSession):
        """Cluster config repository instance."""
        return ClusterConfigRepository(test_session)

    async def test_create_config(self, config_repo):
        """Test creating cluster configuration."""
        config_data = {
            "cluster_name": "test-cluster",
            "cluster_host": "test-cluster.local",
            "username": "testuser",
            "base_path": "/tmp/gpu_jobs",
            "max_concurrent_jobs": 10,
            "max_gpu_per_job": 8,
            "default_timeout_hours": 24
        }
        
        config = await config_repo.create_config(config_data)
        
        assert config.cluster_name == "test-cluster"
        assert config.cluster_host == "test-cluster.local"
        assert config.is_active is True

    async def test_get_active_config(self, config_repo):
        """Test retrieving active cluster configuration."""
        # Create multiple configs
        config1_data = {
            "cluster_name": "inactive-cluster",
            "cluster_host": "inactive.local",
            "username": "testuser",
            "base_path": "/tmp/gpu_jobs",
            "is_active": False
        }
        
        config2_data = {
            "cluster_name": "active-cluster",
            "cluster_host": "active.local", 
            "username": "testuser",
            "base_path": "/tmp/gpu_jobs",
            "is_active": True
        }
        
        await config_repo.create_config(config1_data)
        active_config = await config_repo.create_config(config2_data)
        
        # Get active config
        retrieved = await config_repo.get_active_config()
        
        assert retrieved is not None
        assert retrieved.cluster_name == "active-cluster"
        assert retrieved.is_active is True

    async def test_update_health_status(self, config_repo):
        """Test updating cluster health status."""
        # Create config
        config_data = {
            "cluster_name": "health-test-cluster",
            "cluster_host": "test.local",
            "username": "testuser",
            "base_path": "/tmp/gpu_jobs"
        }
        
        config = await config_repo.create_config(config_data)
        
        # Update health status
        success = await config_repo.update_health_status("health-test-cluster", "healthy")
        assert success is True
        
        # Verify update
        updated_config = await config_repo.get_active_config()
        assert updated_config.health_status == "healthy"
        assert updated_config.last_health_check is not None


class TestDatabaseIntegrity:
    """Test database integrity and constraints."""
    
    async def test_unique_constraints(self, test_session: AsyncSession):
        """Test database unique constraints."""
        user_repo = UserRepository(test_session)
        
        # Create user
        user_data = TEST_USERS[0].copy()
        user1 = await user_repo.create_user(user_data)
        
        # Try to create duplicate username
        duplicate_data = user_data.copy()
        duplicate_data["email"] = "different@example.com"
        
        with pytest.raises(Exception):  # Should raise integrity error
            await user_repo.create_user(duplicate_data)

    async def test_foreign_key_constraints(self, test_session: AsyncSession):
        """Test foreign key constraint enforcement."""
        job_repo = GPUJobRepository(test_session)
        
        job = GPUJobFactory()
        
        # Try to create job with non-existent user
        with pytest.raises(Exception):  # Should raise foreign key error
            await job_repo.create_job(job, "nonexistent_user_id")

    async def test_check_constraints(self, test_session: AsyncSession):
        """Test database check constraints."""
        user_repo = UserRepository(test_session)
        job_repo = GPUJobRepository(test_session)
        
        # Create user
        user = await user_repo.create_user(TEST_USERS[0])
        
        # Try to create job with invalid GPU count
        invalid_job = GPUJobFactory(gpu_count=0)  # Violates check constraint
        
        with pytest.raises(Exception):  # Should raise check constraint error
            await job_repo.create_job(invalid_job, user.id)

    async def test_cascade_deletions(self, test_session: AsyncSession):
        """Test cascade deletion behavior."""
        user_repo = UserRepository(test_session)
        job_repo = GPUJobRepository(test_session)
        result_repo = JobResultRepository(test_session)
        
        # Create user, job, and result
        user = await user_repo.create_user(TEST_USERS[0])
        job = GPUJobFactory()
        db_job = await job_repo.create_job(job, user.id)
        results = JobResultsFactory(job_id=job.job_id)
        await result_repo.create_result(results)
        
        # Delete job (should cascade to results)
        await job_repo.delete_job(job.job_id)
        
        # Result should be deleted due to cascade
        deleted_result = await result_repo.get_result(job.job_id)
        assert deleted_result is None


class TestDatabasePerformance:
    """Test database operation performance."""
    
    async def test_bulk_operations_performance(self, test_session: AsyncSession):
        """Test performance of bulk database operations."""
        user_repo = UserRepository(test_session)
        job_repo = GPUJobRepository(test_session)
        
        # Create user
        user = await user_repo.create_user(TEST_USERS[0])
        
        import time
        start_time = time.time()
        
        # Create many jobs
        jobs = []
        for i in range(50):
            job = GPUJobFactory(job_name=f"bulk_job_{i}")
            await job_repo.create_job(job, user.id)
            jobs.append(job)
        
        elapsed = time.time() - start_time
        
        # Should complete bulk operations reasonably quickly
        assert elapsed < 10.0  # Less than 10 seconds for 50 jobs
        
        # Verify all jobs were created
        all_jobs = await job_repo.list_jobs(limit=100)
        assert len(all_jobs) == 50

    async def test_query_performance_with_indexes(self, test_session: AsyncSession):
        """Test query performance with database indexes."""
        user_repo = UserRepository(test_session)
        job_repo = GPUJobRepository(test_session)
        
        # Create user and many jobs
        user = await user_repo.create_user(TEST_USERS[0])
        
        for i in range(100):
            job = GPUJobFactory(
                status=JobStatus.RUNNING if i % 2 == 0 else JobStatus.COMPLETED
            )
            await job_repo.create_job(job, user.id)
        
        import time
        
        # Test indexed queries
        start_time = time.time()
        
        # Query by status (indexed)
        running_jobs = await job_repo.list_jobs(status=JobStatus.RUNNING)
        
        # Query by user (indexed)
        user_jobs = await job_repo.list_jobs(user_id=user.id)
        
        elapsed = time.time() - start_time
        
        # Indexed queries should be fast
        assert elapsed < 1.0  # Less than 1 second
        assert len(running_jobs) == 50
        assert len(user_jobs) == 100