import asyncio
import os
import ollama
from .base import BaseAgent, A2AMessage
from utils.mcp_client_helper import SimpleMCPClient

class SummarizerAgent(BaseAgent):
    def __init__(self, name: str, mcp_server_path: str):
        super().__init__(name)
        self.mcp_client = SimpleMCPClient(mcp_server_path)

    def process_message(self, message: A2AMessage):
        if message.message_type == "REQUEST" and message.payload.get("task") == "summarize":
            # Check if we are already in an event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We are in an event loop, create a task
                loop.create_task(self.perform_summarization(message))
            else:
                # No event loop, run synchronously
                asyncio.run(self.perform_summarization(message))

    async def perform_summarization(self, original_message: A2AMessage):
        files_to_read = original_message.payload.get("files", [])
        combined_text = ""

        # 1. Read files using MCP
        try:
            await self.mcp_client.connect()

            for file in files_to_read:
                print(f"[{self.name}] Reading {file} via MCP...")
                if file.endswith(".csv"):
                    result = await self.mcp_client.call_tool("read_csv_file", {"filename": file})
                else:
                    result = await self.mcp_client.call_tool("read_text_file", {"filename": file})

                # MCP returns a list of content blocks, usually text
                if result.content:
                    combined_text += f"\n--- Content of {file} ---\n"
                    combined_text += result.content[0].text
                else:
                    combined_text += f"\n--- Failed to read {file} ---\n"

            await self.mcp_client.close()

        except Exception as e:
            print(f"[{self.name}] MCP Error: {e}")
            self.reply(original_message, "ERROR", {"error": str(e)})
            return

        # 2. Summarize using Gemma via Ollama
        try:
            print(f"[{self.name}] Sending data to Gemma 4B for summarization...")
            # Ensure OLLAMA_HOST is picked up from env if set
            prompt = f"Please summarize the following project updates and status information into a concise daily briefing:\n\n{combined_text}"

            response = ollama.chat(model='gemma:4b', messages=[
                {'role': 'user', 'content': prompt},
            ])

            summary = response['message']['content']

            # 3. Send back response
            self.reply(original_message, "RESPONSE", {"summary": summary})

            # CRITICAL: Since this agent essentially owns the event loop (if started via asyncio.run),
            # and the reply might trigger other async agents (like Emailer) that schedule tasks on this loop,
            # we must wait a bit to allow those tasks to complete before this function returns and the loop closes.
            # In a real distributed A2A system, this wouldn't be necessary as agents run independently.
            print(f"[{self.name}] Waiting for downstream tasks to complete...")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"[{self.name}] LLM Error: {e}")
            self.reply(original_message, "ERROR", {"error": str(e)})

    def reply(self, original_message, status, payload):
        callback = original_message.payload.get("callback_agent")
        if callback:
            self.send_message(callback, status, payload, original_message.conversation_id)
