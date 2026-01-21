import time
import os
import sys
import uuid

# If running as a script, ensure local imports work
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))

from agents.base import BaseAgent, A2AMessage
from agents.summarizer import SummarizerAgent
from agents.email_sender import EmailAgent

class ManagerAgent(BaseAgent):
    def __init__(self, name: str, summarizer: SummarizerAgent, emailer: EmailAgent):
        super().__init__(name)
        self.summarizer = summarizer
        self.emailer = emailer
        self.pending_tasks = {}

    def start_daily_routine(self):
        print(f"[{self.name}] Starting daily routine...")

        # Pre-generate conversation ID to handle synchronous/local execution flow
        conversation_id = str(uuid.uuid4())
        self.pending_tasks[conversation_id] = "waiting_for_summary"

        # Step 1: Request Summary
        files = ["projects.csv", "updates.txt"]
        payload = {
            "task": "summarize",
            "files": files,
            "callback_agent": self
        }
        self.send_message(self.summarizer, "REQUEST", payload, conversation_id=conversation_id)

    def process_message(self, message: A2AMessage):
        # Handle incoming messages
        state = self.pending_tasks.get(message.conversation_id)

        if state == "waiting_for_summary" and message.message_type == "RESPONSE":
            summary = message.payload.get("summary")
            print(f"[{self.name}] Received summary. Preparing email...")

            # Step 2: Send Email
            email_payload = {
                "task": "send_email",
                "email_data": {
                    "from": "manager@company.com",
                    "to": "stakeholders@company.com",
                    "subject": "Daily Project Update",
                    "body": summary
                },
                "callback_agent": self
            }
            # Update state for next step
            self.pending_tasks[message.conversation_id] = "waiting_for_email_confirmation"
            self.send_message(self.emailer, "REQUEST", email_payload, message.conversation_id)

        elif state == "waiting_for_email_confirmation" and message.message_type == "RESPONSE":
            print(f"[{self.name}] Email sending confirmed: {message.payload}")
            print(f"[{self.name}] Daily routine complete.")
            self.pending_tasks.pop(message.conversation_id)

        elif message.message_type == "ERROR":
            print(f"[{self.name}] Received Error: {message.payload}")

if __name__ == "__main__":
    # Path to the MCP server script relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(base_dir, "mcp", "server.py")

    # Initialize Agents
    summarizer = SummarizerAgent("Summarizer", mcp_server_path)
    emailer = EmailAgent("EmailSender", mcp_server_path)
    manager = ManagerAgent("Manager", summarizer, emailer)

    # Run
    manager.start_daily_routine()
