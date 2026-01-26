"""
Mock Response Data

Mock SSH command responses, file transfer results, and cluster
interactions for consistent testing.
"""

from datetime import datetime
from typing import Dict, List, Tuple

# SSH Command Response Mapping
SSH_COMMAND_RESPONSES: Dict[str, Tuple[str, str, int]] = {
    # Basic system commands
    "whoami": ("testuser", "", 0),
    "pwd": ("/home/testuser", "", 0),
    "echo 'test'": ("test", "", 0),
    
    # Python and environment
    "python3 --version": ("Python 3.9.12", "", 0),
    "which python3": ("/usr/bin/python3", "", 0),
    "pip --version": ("pip 23.0.1", "", 0),
    
    # Directory operations
    "mkdir -p /tmp/test_jobs/test_job_12345": ("", "", 0),
    "ls -la /tmp/test_jobs": ("drwxr-xr-x 2 testuser testuser 4096 Aug 25 10:00 test_job_12345", "", 0),
    "rm -rf /tmp/test_jobs/test_job_12345": ("", "", 0),
    
    # File operations
    "test -f /tmp/test_jobs/test_job_12345/train.py": ("", "", 0),
    "test -d /tmp/test_jobs/test_job_12345/logs": ("", "", 0),
    "du -sh /tmp/test_jobs/test_job_12345": ("1.2G\t/tmp/test_jobs/test_job_12345", "", 0),
    
    # Git operations
    "git clone https://github.com/test/repo.git /tmp/test_jobs/test_job_12345/code": ("Cloning into '/tmp/test_jobs/test_job_12345/code'...", "", 0),
    "cd /tmp/test_jobs/test_job_12345/code && git checkout main": ("Switched to branch 'main'", "", 0),
    
    # Process management
    "nohup python3 train.py > /tmp/test_jobs/test_job_12345/logs/training.log 2>&1 & echo $!": ("12345", "", 0),
    "ps aux | grep -v grep | grep 12345": ("testuser 12345  2.3  1.1 123456  7890 ?  R  10:00   0:05 python3 train.py", "", 0),
    "kill -0 12345": ("", "", 0),  # Process exists
    "kill -9 12345": ("", "", 0),
    
    # File finding and collection
    "find /tmp/test_jobs/test_job_12345/outputs -name '*.pth' -type f": (
        "/tmp/test_jobs/test_job_12345/outputs/model.pth\n"
        "/tmp/test_jobs/test_job_12345/outputs/best_model.pth", "", 0
    ),
    "find /tmp/test_jobs/test_job_12345/outputs -type f -not -name '*.pth' -not -name '*.pt'": (
        "/tmp/test_jobs/test_job_12345/outputs/metrics.json\n"
        "/tmp/test_jobs/test_job_12345/outputs/config.yaml", "", 0
    ),
    
    # Log operations
    "tail -100 /tmp/test_jobs/test_job_12345/logs/training.log": (
        "Epoch 1: loss=0.856, acc=0.234\n"
        "Epoch 2: loss=0.654, acc=0.456\n"
        "Epoch 3: loss=0.432, acc=0.678\n"
        "Final results: loss=0.123, accuracy=0.894", "", 0
    ),
    "cd /tmp/test_jobs/test_job_12345 && tar -czf logs_archive.tar.gz logs/": ("", "", 0),
    
    # GPU and system monitoring
    "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits": ("1024, 8192", "", 0),
    "free -m": ("Mem: 16384 8192 8192", "", 0),
    
    # Error cases
    "nonexistent_command": ("", "command not found: nonexistent_command", 127),
    "test -f /nonexistent/file": ("", "", 1),  # File not found
}

# File Content Responses
MOCK_FILE_CONTENTS: Dict[str, str] = {
    "/tmp/test_jobs/test_job_12345/logs/training.log": """
2024-08-25 10:00:01 - INFO - Training started
2024-08-25 10:00:02 - INFO - Loading dataset...
2024-08-25 10:05:00 - INFO - Epoch 1/10: loss=0.856, acc=0.234, val_loss=0.901, val_acc=0.198
2024-08-25 10:10:00 - INFO - Epoch 2/10: loss=0.654, acc=0.456, val_loss=0.698, val_acc=0.412
2024-08-25 10:15:00 - INFO - Epoch 3/10: loss=0.432, acc=0.678, val_loss=0.487, val_acc=0.634
2024-08-25 10:20:00 - INFO - Epoch 4/10: loss=0.321, acc=0.789, val_loss=0.356, val_acc=0.745
2024-08-25 10:25:00 - INFO - Epoch 5/10: loss=0.234, acc=0.834, val_loss=0.278, val_acc=0.812
2024-08-25 10:30:00 - INFO - Epoch 6/10: loss=0.189, acc=0.867, val_loss=0.234, val_acc=0.845
2024-08-25 10:35:00 - INFO - Epoch 7/10: loss=0.156, acc=0.889, val_loss=0.198, val_acc=0.867
2024-08-25 10:40:00 - INFO - Epoch 8/10: loss=0.134, acc=0.902, val_loss=0.176, val_acc=0.878
2024-08-25 10:45:00 - INFO - Epoch 9/10: loss=0.118, acc=0.915, val_loss=0.162, val_acc=0.889
2024-08-25 10:50:00 - INFO - Epoch 10/10: loss=0.105, acc=0.925, val_loss=0.151, val_acc=0.898
2024-08-25 10:50:01 - INFO - Training completed
2024-08-25 10:50:02 - INFO - Final results: loss=0.105, accuracy=0.925
2024-08-25 10:50:03 - INFO - Model saved to outputs/final_model.pth
""".strip(),
    
    "/tmp/test_jobs/test_job_12345/outputs/metrics.json": """
{
    "final_loss": 0.105,
    "final_accuracy": 0.925,
    "final_val_loss": 0.151,
    "final_val_accuracy": 0.898,
    "training_time_seconds": 3000,
    "total_epochs": 10,
    "best_epoch": 10,
    "parameters_count": 12534567
}
""".strip(),
    
    "/tmp/test_jobs/test_job_12345/outputs/training_history.json": """
{
    "loss": [0.856, 0.654, 0.432, 0.321, 0.234, 0.189, 0.156, 0.134, 0.118, 0.105],
    "accuracy": [0.234, 0.456, 0.678, 0.789, 0.834, 0.867, 0.889, 0.902, 0.915, 0.925],
    "val_loss": [0.901, 0.698, 0.487, 0.356, 0.278, 0.234, 0.198, 0.176, 0.162, 0.151],
    "val_accuracy": [0.198, 0.412, 0.634, 0.745, 0.812, 0.845, 0.867, 0.878, 0.889, 0.898]
}
""".strip(),

    "/tmp/test_jobs/test_job_12345/run_job.sh": """
#!/bin/bash
cd /tmp/test_jobs/test_job_12345/code
export CUDA_VISIBLE_DEVICES=0,1
python3 train.py
""".strip()
}

# Mock Process States
MOCK_PROCESS_STATES: Dict[str, Dict] = {
    "12345": {
        "exists": True,
        "status": "running",
        "cpu_percent": 85.2,
        "memory_mb": 2048,
        "runtime_seconds": 1800
    },
    "54321": {
        "exists": False,
        "status": "completed",
        "exit_code": 0
    },
    "99999": {
        "exists": True,
        "status": "failed",
        "exit_code": 1,
        "error": "CUDA out of memory"
    }
}

# Mock GPU Information
MOCK_GPU_INFO = {
    "gpu_count": 4,
    "gpus": [
        {"id": 0, "name": "Tesla V100", "memory_total": 32768, "memory_used": 1024},
        {"id": 1, "name": "Tesla V100", "memory_total": 32768, "memory_used": 2048},
        {"id": 2, "name": "Tesla V100", "memory_total": 32768, "memory_used": 0},
        {"id": 3, "name": "Tesla V100", "memory_total": 32768, "memory_used": 512}
    ],
    "total_memory_mb": 131072,
    "used_memory_mb": 3584,
    "available_memory_mb": 127488
}

# Mock Training Metrics
MOCK_TRAINING_METRICS = {
    "epoch_1": {"loss": 0.856, "accuracy": 0.234, "val_loss": 0.901, "val_accuracy": 0.198},
    "epoch_5": {"loss": 0.234, "accuracy": 0.834, "val_loss": 0.278, "val_accuracy": 0.812},
    "epoch_10": {"loss": 0.105, "accuracy": 0.925, "val_loss": 0.151, "val_accuracy": 0.898},
    "final": {"loss": 0.105, "accuracy": 0.925, "training_time_hours": 2.5}
}

# Error Response Templates
ERROR_RESPONSES: Dict[str, Tuple[str, str, int]] = {
    "connection_refused": ("", "ssh: connect to host test-cluster.local port 22: Connection refused", 255),
    "permission_denied": ("", "Permission denied (publickey)", 255),
    "command_not_found": ("", "bash: nonexistent_command: command not found", 127),
    "disk_full": ("", "No space left on device", 1),
    "gpu_memory_error": ("", "RuntimeError: CUDA out of memory", 1),
    "timeout": ("", "Operation timed out", 124)
}

# Helper Functions for Test Data

def get_mock_command_response(command: str) -> Tuple[str, str, int]:
    """Get mock response for SSH command."""
    # Direct mapping
    if command in SSH_COMMAND_RESPONSES:
        return SSH_COMMAND_RESPONSES[command]
    
    # Pattern matching for dynamic commands
    if "mkdir -p" in command:
        return ("", "", 0)
    elif "test -f" in command:
        return ("", "", 0)  # Assume file exists
    elif "test -d" in command:
        return ("", "", 0)  # Assume directory exists
    elif "ps aux" in command and "grep" in command:
        return ("testuser 12345 python3 train.py", "", 0)
    elif "find" in command and "*.pth" in command:
        return ("/tmp/test/model.pth", "", 0)
    elif "tail" in command and ".log" in command:
        return ("Final results: loss=0.123, accuracy=0.894", "", 0)
    else:
        return ("", "", 0)  # Default success


def create_mock_file_content(file_path: str) -> str:
    """Create mock file content based on file path."""
    if file_path.endswith(".log"):
        return MOCK_FILE_CONTENTS.get(file_path, "Mock log content")
    elif file_path.endswith(".json"):
        return MOCK_FILE_CONTENTS.get(file_path, '{"mock": "data"}')
    elif file_path.endswith(".py"):
        return "# Mock Python script\nprint('Hello, World!')"
    elif file_path.endswith(".sh"):
        return "#!/bin/bash\necho 'Mock script'"
    else:
        return "Mock file content"


def get_mock_process_info(process_id: str) -> Dict:
    """Get mock process information."""
    return MOCK_PROCESS_STATES.get(process_id, {
        "exists": False,
        "status": "not_found"
    })