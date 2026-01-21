import unittest
from unittest.mock import MagicMock
from main import ManagerAgent
from agents.base import A2AMessage

class TestManagerAgent(unittest.TestCase):
    def setUp(self):
        self.mock_summarizer = MagicMock()
        self.mock_summarizer.name = "MockSummarizer"
        
        self.mock_emailer = MagicMock()
        self.mock_emailer.name = "MockEmailer"
        
        self.manager = ManagerAgent("TestManager", self.mock_summarizer, self.mock_emailer)

    def test_start_daily_routine_sends_request(self):
        self.manager.start_daily_routine()
        
        # Verify message sent to summarizer
        self.mock_summarizer.receive_message.assert_called_once()
        call_args = self.mock_summarizer.receive_message.call_args[0][0]
        
        self.assertIsInstance(call_args, A2AMessage)
        self.assertEqual(call_args.recipient, "MockSummarizer")
        self.assertEqual(call_args.message_type, "REQUEST")
        self.assertEqual(call_args.payload["task"], "summarize")
        
        # Verify state
        self.assertIn(call_args.conversation_id, self.manager.pending_tasks)
        self.assertEqual(self.manager.pending_tasks[call_args.conversation_id], "waiting_for_summary")

    def test_process_summary_response_sends_email(self):
        # Setup initial state
        conv_id = "test_conv_id"
        self.manager.pending_tasks[conv_id] = "waiting_for_summary"
        
        # Simulate Incoming Response
        summary_payload = {"summary": "Everything is good."}
        response_msg = A2AMessage("MockSummarizer", "TestManager", "RESPONSE", summary_payload, conversation_id=conv_id)
        
        self.manager.process_message(response_msg)
        
        # Verify message sent to emailer
        self.mock_emailer.receive_message.assert_called_once()
        call_args = self.mock_emailer.receive_message.call_args[0][0]
        
        self.assertEqual(call_args.recipient, "MockEmailer")
        self.assertEqual(call_args.message_type, "REQUEST")
        self.assertEqual(call_args.payload["task"], "send_email")
        self.assertEqual(call_args.payload["email_data"]["body"], "Everything is good.")
        
        # Verify state transition
        self.assertEqual(self.manager.pending_tasks[conv_id], "waiting_for_email_confirmation")

    def test_process_email_confirmation_completes_task(self):
        # Setup state
        conv_id = "test_conv_id"
        self.manager.pending_tasks[conv_id] = "waiting_for_email_confirmation"
        
        # Simulate Confirmation
        confirm_payload = {"status": "sent"}
        response_msg = A2AMessage("MockEmailer", "TestManager", "RESPONSE", confirm_payload, conversation_id=conv_id)
        
        self.manager.process_message(response_msg)
        
        # Verify task removed
        self.assertNotIn(conv_id, self.manager.pending_tasks)

if __name__ == '__main__':
    unittest.main()
