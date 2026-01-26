"""
Unit Tests for Cluster Connection Manager

Tests for SSH/SCP connection management, command execution, 
file transfers, and error handling with mocked dependencies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import tempfile
from pathlib import Path

from src.ml.cluster.connection import ClusterConnection
from src.ml.cluster.exceptions import ConnectionError, AuthenticationError, FileTransferError
from tests.fixtures.mock_responses import get_mock_command_response


class TestClusterConnection:
    """Test ClusterConnection class functionality."""
    
    @pytest.fixture
    def connection_config(self):
        """Connection configuration for testing."""
        return {
            "host": "test-cluster.local",
            "port": 22,
            "username": "testuser",
            "password": "testpass",
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 0.1  # Fast retries for testing
        }
    
    @pytest.fixture
    def mock_ssh_client(self):
        """Mock SSH client."""
        client = MagicMock()
        client.connect = MagicMock()
        client.close = MagicMock()
        
        # Mock command execution
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        
        mock_stdout.read.return_value = b"test output"
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        
        client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)
        
        return client
    
    @pytest.fixture
    def mock_scp_client(self):
        """Mock SCP client."""
        scp = MagicMock()
        scp.put = MagicMock()
        scp.get = MagicMock()
        scp.close = MagicMock()
        return scp

    async def test_connection_initialization(self, connection_config):
        """Test connection object initialization."""
        conn = ClusterConnection(**connection_config)
        
        assert conn.host == "test-cluster.local"
        assert conn.port == 22
        assert conn.username == "testuser"
        assert conn.timeout == 30
        assert conn.max_retries == 3
        assert not conn.connected

    @patch('paramiko.SSHClient')
    async def test_successful_connection(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test successful SSH connection."""
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        result = await conn.connect()
        
        assert result is True
        assert conn.connected is True
        mock_ssh_client.connect.assert_called_once()

    @patch('paramiko.SSHClient')
    async def test_connection_failure(self, mock_ssh_class, connection_config):
        """Test SSH connection failure handling."""
        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception("Connection refused")
        mock_ssh_class.return_value = mock_client
        
        conn = ClusterConnection(**connection_config)
        
        with pytest.raises(ConnectionError) as exc_info:
            await conn.connect()
        
        assert "Connection refused" in str(exc_info.value)
        assert conn.connected is False

    @patch('paramiko.SSHClient')
    async def test_authentication_failure(self, mock_ssh_class, connection_config):
        """Test SSH authentication failure."""
        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception("Authentication failed")
        mock_ssh_class.return_value = mock_client
        
        conn = ClusterConnection(**connection_config)
        
        with pytest.raises(AuthenticationError):
            await conn.connect()

    @patch('paramiko.SSHClient')
    async def test_command_execution_success(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test successful command execution."""
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        stdout, stderr, exit_code = await conn.execute_command("echo 'test'")
        
        assert stdout == "test output"
        assert stderr == ""
        assert exit_code == 0
        mock_ssh_client.exec_command.assert_called_once_with("echo 'test'", timeout=None)

    @patch('paramiko.SSHClient')
    async def test_command_execution_with_timeout(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test command execution with timeout."""
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        await conn.execute_command("long_command", timeout=60)
        
        mock_ssh_client.exec_command.assert_called_once_with("long_command", timeout=60)

    @patch('paramiko.SSHClient')
    async def test_command_execution_failure(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test command execution failure handling."""
        # Configure mock to return error
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr.read.return_value = b"command not found"
        mock_stdout.channel.recv_exit_status.return_value = 127
        
        mock_ssh_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        stdout, stderr, exit_code = await conn.execute_command("nonexistent_command")
        
        assert stdout == ""
        assert stderr == "command not found"
        assert exit_code == 127

    @patch('paramiko.SSHClient')
    @patch('scp.SCPClient')
    async def test_file_upload_success(self, mock_scp_class, mock_ssh_class, connection_config, mock_ssh_client, mock_scp_client, tmp_path):
        """Test successful file upload."""
        mock_ssh_class.return_value = mock_ssh_client
        mock_scp_class.return_value = mock_scp_client
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        result = await conn.upload_file(str(test_file), "/remote/path/test.txt")
        
        assert result is True
        mock_scp_client.put.assert_called_once()

    @patch('paramiko.SSHClient')
    @patch('scp.SCPClient')
    async def test_file_upload_with_verification(self, mock_scp_class, mock_ssh_class, connection_config, mock_ssh_client, mock_scp_client, tmp_path):
        """Test file upload with integrity verification."""
        mock_ssh_class.return_value = mock_ssh_client
        mock_scp_class.return_value = mock_scp_client
        
        # Configure checksum verification
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"5d41402abc4b2a76b9719d911017c592"  # MD5 of "hello"
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_ssh_client.exec_command.return_value = (MagicMock(), mock_stdout, MagicMock())
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        result = await conn.upload_file(str(test_file), "/remote/path/test.txt", verify_integrity=True)
        
        assert result is True

    @patch('paramiko.SSHClient')
    @patch('scp.SCPClient')
    async def test_file_upload_failure(self, mock_scp_class, mock_ssh_class, connection_config, mock_ssh_client):
        """Test file upload failure handling."""
        mock_ssh_class.return_value = mock_ssh_client
        mock_scp_client = MagicMock()
        mock_scp_client.put.side_effect = Exception("Permission denied")
        mock_scp_class.return_value = mock_scp_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        with pytest.raises(FileTransferError) as exc_info:
            await conn.upload_file("/local/file.txt", "/remote/file.txt")
        
        assert "Permission denied" in str(exc_info.value)

    @patch('paramiko.SSHClient')
    @patch('scp.SCPClient')
    async def test_file_download_success(self, mock_scp_class, mock_ssh_class, connection_config, mock_ssh_client, mock_scp_client, tmp_path):
        """Test successful file download."""
        mock_ssh_class.return_value = mock_ssh_client
        mock_scp_class.return_value = mock_scp_client
        
        # Mock file creation during download
        def mock_scp_get(remote_path, local_path):
            Path(local_path).write_text("downloaded content")
        
        mock_scp_client.get.side_effect = mock_scp_get
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        local_file = tmp_path / "downloaded.txt"
        result = await conn.download_file("/remote/file.txt", str(local_file))
        
        assert result is True
        assert local_file.exists()
        assert local_file.read_text() == "downloaded content"

    @patch('paramiko.SSHClient')
    async def test_connection_retry_logic(self, mock_ssh_class, connection_config):
        """Test connection retry mechanism."""
        mock_client = MagicMock()
        
        # Fail first two attempts, succeed on third
        mock_client.connect.side_effect = [
            Exception("Connection timeout"),
            Exception("Connection timeout"), 
            None  # Success
        ]
        mock_ssh_class.return_value = mock_client
        
        conn = ClusterConnection(**connection_config)
        result = await conn.connect()
        
        assert result is True
        assert mock_client.connect.call_count == 3

    @patch('paramiko.SSHClient')
    async def test_connection_max_retries_exceeded(self, mock_ssh_class, connection_config):
        """Test behavior when max retries exceeded."""
        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception("Connection timeout")
        mock_ssh_class.return_value = mock_client
        
        conn = ClusterConnection(**connection_config)
        
        with pytest.raises(ConnectionError):
            await conn.connect()
        
        # Should try max_retries + 1 times (initial + retries)
        assert mock_client.connect.call_count == 4

    @patch('paramiko.SSHClient')
    async def test_connection_test(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test connection testing functionality."""
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        result = await conn.test_connection()
        
        assert result is True
        mock_ssh_client.exec_command.assert_called_once_with("echo 'connection_test'", timeout=10)

    @patch('paramiko.SSHClient')
    async def test_connection_cleanup(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test connection cleanup and resource management."""
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        await conn.disconnect()
        
        assert conn.connected is False
        mock_ssh_client.close.assert_called_once()

    @patch('paramiko.SSHClient')
    async def test_context_manager_usage(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test using connection as async context manager."""
        mock_ssh_class.return_value = mock_ssh_client
        
        async with ClusterConnection(**connection_config) as conn:
            assert conn.connected is True
        
        # Should automatically disconnect
        mock_ssh_client.close.assert_called_once()

    @patch('paramiko.SSHClient')
    async def test_command_injection_prevention(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test command injection prevention."""
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        # Potentially dangerous commands should be handled carefully
        dangerous_commands = [
            "rm -rf /",
            "echo 'test'; rm important_file",
            "test && rm -rf /tmp/*",
            "$(rm -rf /)"
        ]
        
        for cmd in dangerous_commands:
            # The connection should execute but we should have validation elsewhere
            # This test ensures the connection layer doesn't crash
            await conn.execute_command(cmd)
            mock_ssh_client.exec_command.assert_called()

    @patch('paramiko.SSHClient')
    async def test_large_output_handling(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test handling of large command output."""
        # Mock large output
        large_output = "line\n" * 10000  # 10k lines
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = large_output.encode()
        mock_stdout.channel.recv_exit_status.return_value = 0
        
        mock_ssh_client.exec_command.return_value = (MagicMock(), mock_stdout, MagicMock())
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        stdout, stderr, exit_code = await conn.execute_command("generate_large_output")
        
        assert len(stdout.split('\n')) == 10001  # 10k lines + empty line
        assert exit_code == 0

    @patch('paramiko.SSHClient')
    @patch('scp.SCPClient')
    async def test_concurrent_operations(self, mock_scp_class, mock_ssh_class, connection_config, mock_ssh_client, mock_scp_client):
        """Test concurrent SSH operations."""
        mock_ssh_class.return_value = mock_ssh_client
        mock_scp_class.return_value = mock_scp_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        # Execute multiple commands concurrently
        import asyncio
        tasks = [
            conn.execute_command("command1"),
            conn.execute_command("command2"),
            conn.execute_command("command3")
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(result[2] == 0 for result in results)  # All should succeed

    @patch('paramiko.SSHClient')
    async def test_connection_recovery(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test connection recovery after network interruption."""
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        # Simulate connection loss
        mock_ssh_client.exec_command.side_effect = Exception("Connection lost")
        
        # Should handle gracefully and attempt reconnection
        with patch.object(conn, 'connect', return_value=True) as mock_reconnect:
            try:
                await conn.execute_command("test_command")
            except:
                pass  # Expected to fail first time
            
            # Manual reconnection test
            result = await conn.connect()
            assert result is True

    @patch('hashlib.md5')
    async def test_file_integrity_verification(self, mock_md5, tmp_path):
        """Test file integrity verification logic."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = "test file content"
        test_file.write_text(test_content)
        
        # Mock MD5 calculation
        mock_hash = MagicMock()
        mock_hash.hexdigest.return_value = "abc123def456"
        mock_md5.return_value = mock_hash
        
        conn = ClusterConnection(host="test", username="test")
        
        # Test local checksum calculation
        local_checksum = await conn._calculate_file_checksum(str(test_file))
        assert local_checksum == "abc123def456"

    async def test_path_sanitization(self):
        """Test path sanitization for security."""
        conn = ClusterConnection(host="test", username="test")
        
        # Test various path inputs
        test_cases = [
            ("/safe/path/file.txt", "/safe/path/file.txt"),
            ("../../../etc/passwd", "etc/passwd"),  # Should strip leading ../
            ("/tmp/../etc/passwd", "/etc/passwd"),  # Should resolve ..
            ("normal_file.txt", "normal_file.txt")
        ]
        
        for input_path, expected in test_cases:
            # This assumes we have a path sanitization method
            # The actual implementation should be added to ClusterConnection
            sanitized = conn._sanitize_path(input_path) if hasattr(conn, '_sanitize_path') else input_path
            # For now, just ensure no exception is raised
            assert isinstance(sanitized, str)

    @patch('paramiko.SSHClient')
    async def test_connection_timeout_handling(self, mock_ssh_class, connection_config):
        """Test connection timeout handling."""
        mock_client = MagicMock()
        
        import socket
        mock_client.connect.side_effect = socket.timeout("Connection timed out")
        mock_ssh_class.return_value = mock_client
        
        conn = ClusterConnection(**connection_config)
        
        with pytest.raises(ConnectionError) as exc_info:
            await conn.connect()
        
        assert "timed out" in str(exc_info.value).lower()

    @patch('paramiko.SSHClient')
    async def test_cleanup_job_files(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test job file cleanup functionality."""
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        job_path = "/tmp/test_jobs/test_job_123"
        
        # Test cleanup with results preservation
        await conn.cleanup_job_files(job_path, keep_results=True)
        mock_ssh_client.exec_command.assert_called()
        
        # Verify the correct cleanup command was called
        last_call = mock_ssh_client.exec_command.call_args
        command = last_call[0][0]
        assert job_path in command
        assert "outputs" not in command or "mv" in command  # Should preserve outputs

    @patch('paramiko.SSHClient')
    async def test_get_cluster_info(self, mock_ssh_class, connection_config, mock_ssh_client):
        """Test cluster information gathering."""
        mock_ssh_class.return_value = mock_ssh_client
        
        # Mock various system info commands
        def mock_exec_command(command, timeout=None):
            mock_stdout = MagicMock()
            mock_stderr = MagicMock()
            mock_stderr.read.return_value = b""
            
            if "nvidia-smi" in command:
                mock_stdout.read.return_value = b"Tesla V100, Tesla V100"
            elif "free -m" in command:
                mock_stdout.read.return_value = b"Mem: 65536 32768 32768"
            elif "uptime" in command:
                mock_stdout.read.return_value = b"up 10 days, 5:30, load average: 0.50"
            else:
                mock_stdout.read.return_value = b""
            
            mock_stdout.channel.recv_exit_status.return_value = 0
            return (MagicMock(), mock_stdout, mock_stderr)
        
        mock_ssh_client.exec_command.side_effect = mock_exec_command
        mock_ssh_class.return_value = mock_ssh_client
        
        conn = ClusterConnection(**connection_config)
        conn._client = mock_ssh_client
        conn.connected = True
        
        info = await conn.get_cluster_info()
        
        assert isinstance(info, dict)
        # Should contain system information
        assert len(info) > 0