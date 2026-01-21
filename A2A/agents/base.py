import uuid
import json
from datetime import datetime
from typing import Dict, Any

class A2AMessage:
    def __init__(self, sender: str, recipient: str, message_type: str, payload: Dict[str, Any], conversation_id: str = None):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type # e.g., "REQUEST", "RESPONSE", "INFO"
        self.payload = payload
        self.conversation_id = conversation_id or str(uuid.uuid4())

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        msg = cls(
            sender=data['sender'],
            recipient=data['recipient'],
            message_type=data['message_type'],
            payload=data['payload'],
            conversation_id=data.get('conversation_id')
        )
        msg.id = data['id']
        msg.timestamp = data['timestamp']
        return msg

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.inbox = []

    def send_message(self, recipient_agent: 'BaseAgent', message_type: str, payload: Dict[str, Any], conversation_id: str = None):
        msg = A2AMessage(self.name, recipient_agent.name, message_type, payload, conversation_id)
        print(f"[{self.name} -> {recipient_agent.name}] Sending {message_type}...")
        recipient_agent.receive_message(msg)
        return msg.conversation_id

    def receive_message(self, message: A2AMessage):
        self.inbox.append(message)
        self.process_message(message)

    def process_message(self, message: A2AMessage):
        """To be overridden by subclasses"""
        pass
