import unittest
import json
from agents.base import BaseAgent, A2AMessage

class TestA2AMessage(unittest.TestCase):
    def test_message_creation(self):
        payload = {"key": "value"}
        msg = A2AMessage("sender", "recipient", "TEST", payload)
        
        self.assertEqual(msg.sender, "sender")
        self.assertEqual(msg.recipient, "recipient")
        self.assertEqual(msg.message_type, "TEST")
        self.assertEqual(msg.payload, payload)
        self.assertIsNotNone(msg.id)
        self.assertIsNotNone(msg.timestamp)
        self.assertIsNotNone(msg.conversation_id)

    def test_json_serialization(self):
        payload = {"data": 123}
        msg = A2AMessage("A", "B", "INFO", payload)
        json_str = msg.to_json()
        
        msg_loaded = A2AMessage.from_json(json_str)
        
        self.assertEqual(msg.id, msg_loaded.id)
        self.assertEqual(msg.sender, msg_loaded.sender)
        self.assertEqual(msg.payload, msg_loaded.payload)
        self.assertEqual(msg.conversation_id, msg_loaded.conversation_id)

class TestBaseAgent(unittest.TestCase):
    def test_send_receive(self):
        agent1 = BaseAgent("Agent1")
        agent2 = BaseAgent("Agent2")
        
        payload = {"foo": "bar"}
        conversation_id = agent1.send_message(agent2, "PING", payload)
        
        self.assertEqual(len(agent2.inbox), 1)
        received_msg = agent2.inbox[0]
        
        self.assertEqual(received_msg.sender, "Agent1")
        self.assertEqual(received_msg.recipient, "Agent2")
        self.assertEqual(received_msg.message_type, "PING")
        self.assertEqual(received_msg.conversation_id, conversation_id)

if __name__ == '__main__':
    unittest.main()
