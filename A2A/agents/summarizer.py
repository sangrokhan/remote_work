import asyncio
import os

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

        # 2. Summarize using Gemma via Transformers
        try:
            print(f"[{self.name}] Loading local model for summarization...")
            model_path = os.environ.get("LOCAL_MODEL_PATH", "./models/gemma-2b-it")
            
            # Lazy import to avoid import errors if not installed yet or heavy load on init
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            # TODO: Optimization - persist the model in memory (`self.model`) to avoid reloading.
            # But for now, we load on demand as requested.
            print(f"[{self.name}] Loading model from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            prompt = f"Please summarize the following project updates and status information into a concise daily briefing:\n\n{combined_text}"
            
            # Generate
            results = generator(prompt, max_length=500, do_sample=True, temperature=0.7)
            summary = results[0]['generated_text']
            # Basic cleanup if prompt is included
            if summary.startswith(prompt):
                summary = summary[len(prompt):].strip()

            # 3. Send back response
            self.reply(original_message, "RESPONSE", {"summary": summary})
            
            print(f"[{self.name}] Waiting for downstream tasks to complete...")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"[{self.name}] LLM Error: {e}")
            self.reply(original_message, "ERROR", {"error": str(e)})

    def reply(self, original_message, status, payload):
        callback = original_message.payload.get("callback_agent")
        if callback:
            self.send_message(callback, status, payload, original_message.conversation_id)
