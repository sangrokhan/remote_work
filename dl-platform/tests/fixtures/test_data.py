"""
Test Data Factories

Factory functions and sample data for creating consistent test objects
across the GPU cluster test suite.
"""

from datetime import datetime, timedelta
from typing import Dict, List
import factory
from factory import Factory, Faker, LazyAttribute, SubFactory, fuzzy

from src.ml.cluster.models import (
    GPUJobRequest, GPUJob, JobStatus, CodeSource, JobProgress, JobResults
)


class GPUJobRequestFactory(Factory):
    """Factory for creating GPUJobRequest test objects."""
    
    class Meta:
        model = GPUJobRequest
    
    job_name = Faker('word')
    gpu_count = fuzzy.FuzzyInteger(1, 4)
    code_source = fuzzy.FuzzyChoice([CodeSource.GIT, CodeSource.MANUAL])
    entry_script = "train.py"
    
    # Code path based on source
    code_path = factory.LazyAttribute(lambda obj: "https://github.com/test/repo.git" if obj.code_source == CodeSource.GIT else None)
    
    # Optional fields
    script_args = factory.Dict({"epochs": "100", "batch_size": "32"})
    environment_vars = factory.Dict({"CUDA_VISIBLE_DEVICES": "0,1"})
    python_packages = ["torch", "numpy", "pandas"]


class GPUJobFactory(Factory):
    """Factory for creating GPUJob test objects."""
    
    class Meta:
        model = GPUJob
    
    job_id = factory.LazyAttribute(lambda _: f"test_job_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    job_name = Faker('word')
    gpu_count = fuzzy.FuzzyInteger(1, 4)
    code_source = fuzzy.FuzzyChoice([CodeSource.GIT, CodeSource.MANUAL])
    entry_script = "train.py"
    
    git_url = "https://github.com/test/repo.git"
    git_branch = "main"
    
    model_files = factory.Dict({"pretrained": "models/pretrained.pth"})
    data_files = factory.Dict({"train": "data/train.csv"})
    environment_vars = factory.Dict({"CUDA_VISIBLE_DEVICES": "0,1"})
    python_packages = ["torch", "numpy"]
    
    status = JobStatus.PENDING
    cluster_path = factory.LazyAttribute(lambda obj: f"/tmp/test_jobs/{obj.job_id}")
    created_at = factory.LazyFunction(datetime.now)


class RunningJobFactory(GPUJobFactory):
    """Factory for running GPU jobs."""
    
    status = JobStatus.RUNNING
    cluster_job_id = "12345"
    submitted_at = factory.LazyFunction(lambda: datetime.now() - timedelta(minutes=30))
    started_at = factory.LazyFunction(lambda: datetime.now() - timedelta(minutes=25))
    current_epoch = 5
    total_epochs = 10
    current_loss = 0.234
    current_accuracy = 0.834


class CompletedJobFactory(GPUJobFactory):
    """Factory for completed GPU jobs."""
    
    status = JobStatus.COMPLETED
    cluster_job_id = "12345"
    submitted_at = factory.LazyFunction(lambda: datetime.now() - timedelta(hours=2))
    started_at = factory.LazyFunction(lambda: datetime.now() - timedelta(hours=2) + timedelta(minutes=5))
    completed_at = factory.LazyFunction(lambda: datetime.now() - timedelta(minutes=10))
    current_epoch = 10
    total_epochs = 10
    current_loss = 0.105
    current_accuracy = 0.925
    final_metrics = factory.Dict({
        "final_loss": 0.105,
        "final_accuracy": 0.925,
        "training_time_hours": 1.83
    })


class FailedJobFactory(GPUJobFactory):
    """Factory for failed GPU jobs."""
    
    status = JobStatus.FAILED
    cluster_job_id = "99999"
    submitted_at = factory.LazyFunction(lambda: datetime.now() - timedelta(hours=1))
    started_at = factory.LazyFunction(lambda: datetime.now() - timedelta(hours=1) + timedelta(minutes=5))
    completed_at = factory.LazyFunction(lambda: datetime.now() - timedelta(minutes=30))
    error_message = "CUDA out of memory"
    exit_code = 1


class JobProgressFactory(Factory):
    """Factory for job progress objects."""
    
    class Meta:
        model = JobProgress
    
    job_id = factory.LazyFunction(lambda: f"test_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    current_epoch = fuzzy.FuzzyInteger(1, 10)
    total_epochs = 10
    progress_percent = factory.LazyAttribute(lambda obj: (obj.current_epoch / obj.total_epochs) * 100)
    current_loss = fuzzy.FuzzyFloat(0.1, 1.0)
    current_accuracy = fuzzy.FuzzyFloat(0.5, 0.99)
    estimated_time_remaining_minutes = fuzzy.FuzzyInteger(10, 120)
    memory_usage_mb = fuzzy.FuzzyFloat(1024, 8192)
    gpu_usage_percent = fuzzy.FuzzyFloat(80, 100)


class JobResultsFactory(Factory):
    """Factory for job results objects."""
    
    class Meta:
        model = JobResults
    
    job_id = factory.LazyFunction(lambda: f"test_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    model_files = ["/app/results/test_job/models/final_model.pth"]
    log_files = ["/app/results/test_job/logs/training.log"]
    output_files = ["/app/results/test_job/outputs/metrics.json"]
    final_metrics = factory.Dict({
        "final_loss": 0.105,
        "final_accuracy": 0.925
    })
    training_history = factory.Dict({
        "loss": [0.856, 0.654, 0.432, 0.234, 0.105],
        "accuracy": [0.234, 0.456, 0.678, 0.834, 0.925]
    })
    total_size_mb = fuzzy.FuzzyFloat(100, 2000)
    collected_at = factory.LazyFunction(datetime.now)


# Sample Data Sets

SAMPLE_JOB_REQUESTS = [
    {
        "job_name": "image_classification_training",
        "gpu_count": 2,
        "code_source": "git",
        "code_path": "https://github.com/example/image-classifier.git",
        "entry_script": "train.py",
        "script_args": {"epochs": "50", "batch_size": "32"},
        "environment_vars": {"CUDA_VISIBLE_DEVICES": "0,1", "BATCH_SIZE": "32"},
        "python_packages": ["torch", "torchvision", "transformers"],
        "docker_image": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
    },
    {
        "job_name": "nlp_fine_tuning",
        "gpu_count": 1,
        "code_source": "manual",
        "entry_script": "finetune.py",
        "script_args": {"learning_rate": "2e-5", "batch_size": "16"},
        "environment_vars": {"TOKENIZERS_PARALLELISM": "false"},
        "python_packages": ["transformers", "datasets", "accelerate"]
    },
    {
        "job_name": "reinforcement_learning",
        "gpu_count": 4,
        "code_source": "git",
        "git_url": "https://github.com/example/rl-training.git",
        "git_branch": "experiment/ppo",
        "entry_script": "train_ppo.py",
        "environment_vars": {
            "NUM_ENVS": "128",
            "TOTAL_TIMESTEPS": "10000000"
        },
        "python_packages": ["torch", "stable-baselines3", "gymnasium"]
    }
]

SAMPLE_TRAINING_LOGS = [
    "2024-08-25 10:00:01 - INFO - Training started with config: {model: 'resnet50', epochs: 10}",
    "2024-08-25 10:05:00 - INFO - Epoch 1/10: loss=0.856, acc=0.234, val_loss=0.901, val_acc=0.198",
    "2024-08-25 10:10:00 - INFO - Epoch 2/10: loss=0.654, acc=0.456, val_loss=0.698, val_acc=0.412",
    "2024-08-25 10:15:00 - INFO - Epoch 3/10: loss=0.432, acc=0.678, val_loss=0.487, val_acc=0.634",
    "2024-08-25 10:20:00 - INFO - Epoch 4/10: loss=0.321, acc=0.789, val_loss=0.356, val_acc=0.745",
    "2024-08-25 10:25:00 - INFO - Epoch 5/10: loss=0.234, acc=0.834, val_loss=0.278, val_acc=0.812",
    "2024-08-25 10:50:01 - INFO - Training completed successfully",
    "2024-08-25 10:50:02 - INFO - Final results: loss=0.105, accuracy=0.925"
]

# Test User Data
TEST_USERS = [
    {
        "username": "testuser1",
        "email": "testuser1@example.com",
        "full_name": "Test User One",
        "is_active": True
    },
    {
        "username": "testuser2", 
        "email": "testuser2@example.com",
        "full_name": "Test User Two",
        "is_active": True
    },
    {
        "username": "inactive_user",
        "email": "inactive@example.com",
        "full_name": "Inactive User",
        "is_active": False
    }
]

# Mock Cluster Responses
CLUSTER_INFO_RESPONSE = {
    "cluster_name": "test-gpu-cluster",
    "total_nodes": 4,
    "active_nodes": 4,
    "total_gpus": 16,
    "available_gpus": 12,
    "queue_length": 3,
    "avg_wait_time_minutes": 15,
    "system_load": 0.75
}