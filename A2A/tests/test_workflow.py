import unittest
import os
import sys
import shutil
import asyncio
import json
from unittest.mock import MagicMock, patch

# Ensure we can import the A2A modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.executors import SummarizerExecutor, EmailExecutor
from python_a2a.utils.conversion import create_text_message
from python_a2a.models.message import MessageRole

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
        # Optional: Clean up
        pass

    @patch('agents.executors.pipeline') 
    @patch('agents.executors.AutoTokenizer.from_pretrained')
    @patch('agents.executors.AutoModelForCausalLM.from_pretrained')
    def test_workflow_execution(self, mock_model, mock_tokenizer, mock_pipeline):
        """
        Tests the full workflow using Executors
        """
        
        # Mock Transformer Pipeline Output
        mock_generator = MagicMock()
        mock_generator.return_value = [{'generated_text': "Unit Test Summary: Projects are going well."}]
        mock_pipeline.return_value = mock_generator

        async def run_test():
            # Initialize Executors
            summarizer = SummarizerExecutor(self.mcp_server_path)
            emailer = EmailExecutor(self.mcp_server_path)

            # 1. Test Summarizer
            files = ["projects.csv", "updates.txt"]
            summarize_payload = json.dumps({"files": files})
            input_msg = create_text_message(summarize_payload, role=MessageRole.USER)

            response_msg = await summarizer.execute_task(input_msg)
            
            # Verify Response
            summary_text = ""
            if hasattr(response_msg.content, 'text'):
                summary_text = response_msg.content.text
            else:
                summary_text = str(response_msg.content)
            
            print(f"Test Summary: {summary_text}")
            self.assertIn("Unit Test Summary", summary_text)

            # 2. Test Emailer
            email_payload = json.dumps({
                "email_data": {
                    "from": "manager@company.com",
                    "to": "stakeholders@company.com",
                    "subject": "Daily Project Update",
                    "body": summary_text
                }
            })
            email_msg = create_text_message(email_payload, role=MessageRole.USER)

            response_msg = await emailer.execute_task(email_msg)

            # Verify Email Response
            resp_text = ""
            if hasattr(response_msg.content, 'text'):
                resp_text = response_msg.content.text
            else:
                resp_text = str(response_msg.content)

            print(f"Test Email Event: {resp_text}")
            self.assertIn("Email sent successfully", resp_text)

            # Verify File Creation
            files = os.listdir(self.output_dir)
            email_files = [f for f in files if f.startswith("email_")]
            self.assertTrue(len(email_files) > 0, "No email file was created.")

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
