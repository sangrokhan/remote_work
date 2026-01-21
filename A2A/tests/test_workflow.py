import unittest
import os
import sys
import shutil
import asyncio
from unittest.mock import MagicMock, patch

# Ensure we can import the A2A modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base import A2AMessage
from agents.summarizer import SummarizerAgent
from agents.email_sender import EmailAgent
from main import ManagerAgent

class TestA2AWorkflow(unittest.TestCase):
    def setUp(self):
        # Setup paths
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.mcp_server_path = os.path.join(self.base_dir, "mcp", "server.py")
        self.output_dir = os.path.join(self.base_dir, "output")

        # Clean output dir
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        # Optional: Clean up output dir
        # shutil.rmtree(self.output_dir)
        pass

    @patch('ollama.chat')
    def test_workflow_execution(self, mock_ollama):
        """
        Tests the full workflow:
        Manager -> Request Summary -> Summarizer (Mock LLM) -> MCP Read -> Summarizer ->
        Response -> Manager -> Request Email -> Emailer -> MCP Write -> Response -> Manager
        """

        # Mock LLM response
        mock_ollama.return_value = {
            'message': {
                'content': "Unit Test Summary: Projects are going well."
            }
        }

        # Initialize Agents
        # Note: We rely on the real MCP server script being present and python being in path
        summarizer = SummarizerAgent("Summarizer", self.mcp_server_path)
        emailer = EmailAgent("EmailSender", self.mcp_server_path)
        manager = ManagerAgent("Manager", summarizer, emailer)

        # Run the routine
        # Since the agents use asyncio.run() internally for their tasks,
        # and this test is synchronous, it should work fine without an outer event loop.
        print("\n>>> Starting Test Workflow...")
        manager.start_daily_routine()

        # Verify Email was "Sent" (File created)
        files = os.listdir(self.output_dir)
        email_files = [f for f in files if f.startswith("email_")]

        self.assertTrue(len(email_files) > 0, "No email file was created.")

        # Read the email file content
        with open(os.path.join(self.output_dir, email_files[0]), 'r') as f:
            content = f.read()
            print(f">>> Generated Email Content:\n{content}")
            self.assertIn("Unit Test Summary", content)
            self.assertIn("SUBJECT: Daily Project Update", content)

if __name__ == '__main__':
    unittest.main()
