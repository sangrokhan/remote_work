import asyncio
import os
from .base import BaseAgent, A2AMessage
from utils.mcp_client_helper import SimpleMCPClient

class EmailAgent(BaseAgent):
    def __init__(self, name: str, mcp_server_path: str):
        super().__init__(name)
        self.mcp_client = SimpleMCPClient(mcp_server_path)

    def process_message(self, message: A2AMessage):
        if message.message_type == "REQUEST" and message.payload.get("task") == "send_email":
            # Check if we are already in an event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                loop.create_task(self.send_email(message))
            else:
                asyncio.run(self.send_email(message))

    async def send_email(self, original_message: A2AMessage):
        data = original_message.payload.get("email_data", {})
        sender = data.get("from")
        receiver = data.get("to")
        subject = data.get("subject")
        body = data.get("body")

        email_content = f"""
FROM: {sender}
TO: {receiver}
SUBJECT: {subject}
DATE: {original_message.timestamp}

----------------------------------------------------------------------
{body}
----------------------------------------------------------------------
        """

        filename = f"email_{original_message.id}.txt"

        try:
            await self.mcp_client.connect()
            print(f"[{self.name}] Writing email to file system via MCP...")
            result = await self.mcp_client.call_tool("write_email_file", {"filename": filename, "content": email_content})
            await self.mcp_client.close()

            # Reply success
            callback = original_message.payload.get("callback_agent")
            if callback:
                self.send_message(callback, "RESPONSE", {"status": "sent", "file": filename}, original_message.conversation_id)

        except Exception as e:
            print(f"[{self.name}] Error writing email: {e}")
