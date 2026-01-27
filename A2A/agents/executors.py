import asyncio
import os
import json
import uuid
from typing import Dict, Any

from python_a2a.server.base import BaseA2AServer
from python_a2a.models.message import Message, MessageRole
from python_a2a.utils.conversion import create_text_message

# Move transformers imports to top level for better testing/mocking
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    pass

# Reuse existing utils
from utils.mcp_client_helper import SimpleMCPClient

class SummarizerExecutor(BaseA2AServer):
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.mcp_client = SimpleMCPClient(mcp_server_path)

    def handle_message(self, message: Message) -> Message:
        return asyncio.run(self.execute_task(message))

    async def execute_task(self, message: Message) -> Message:
        # Extract arguments
        content_obj = message.content
        args = {}
        # If accessing content object directly 
        if hasattr(content_obj, 'text'): 
             try:
                args = json.loads(content_obj.text)
             except:
                pass
        elif isinstance(content_obj, dict): # Should not happen if strictly typed Message
             pass # Already dict?
        
        # If type is dict (e.g. from legacy dict conversion), handle it
        # But Message.content must be a Content object.

        files_to_read = args.get("files", [])
        combined_text = ""

        # 1. Read files
        try:
            await self.mcp_client.connect()
            for file in files_to_read:
                print(f"[Summarizer] Reading {file} via MCP...")
                if file.endswith(".csv"):
                    result = await self.mcp_client.call_tool("read_csv_file", {"filename": file})
                else:
                    result = await self.mcp_client.call_tool("read_text_file", {"filename": file})

                if result.content:
                    combined_text += f"\n--- Content of {file} ---\n"
                    combined_text += result.content[0].text
                else:
                    combined_text += f"\n--- Failed to read {file} ---\n"
            await self.mcp_client.close()
        except Exception as e:
            return create_text_message(f"Error reading files: {e}", role=MessageRole.SYSTEM)

        # 2. Summarize
        try:
            print("[Summarizer] Sending text to LLM service...")
            llm_service_url = os.environ.get("LLM_SERVICE_URL", "http://llm_service:8000")
            
            import httpx
            
            prompt = f"Summarize:\n\n{combined_text}"
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{llm_service_url}/generate",
                    json={"prompt": prompt, "max_length": 500}
                )
                response.raise_for_status()
                result = response.json()
                summary = result.get("text", "")
            
            return create_text_message(summary, role=MessageRole.AGENT)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return create_text_message(f"Error summarizing: {e}", role=MessageRole.SYSTEM)


class EmailExecutor(BaseA2AServer):
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.mcp_client = SimpleMCPClient(mcp_server_path)

    async def execute_task(self, message: Message) -> Message:
        # Parse args
        content_obj = message.content
        args = {}
        if hasattr(content_obj, 'text'):
            try: 
                args = json.loads(content_obj.text)
            except: pass
        
        email_data = args.get("email_data", {})
        sender = email_data.get("from")
        receiver = email_data.get("to")
        subject = email_data.get("subject")
        body = email_data.get("body")
        
        msg_id = str(uuid.uuid4())
        filename = f"email_{msg_id}.txt"
        
        email_content = f"FROM: {sender}\nTO: {receiver}\nSUBJECT: {subject}\n\n{body}"

        try:
            await self.mcp_client.connect()
            print(f"[Emailer] Writing email via MCP...")
            await self.mcp_client.call_tool("write_email_file", {"filename": filename, "content": email_content})
            await self.mcp_client.close()

            return create_text_message(f"Email sent successfully. File: {filename}", role=MessageRole.AGENT)

        except Exception as e:
            return create_text_message(f"Error sending email: {e}", role=MessageRole.SYSTEM)

    def handle_message(self, message: Message) -> Message:
        return asyncio.run(self.execute_task(message))


class ParquetAnalyzerExecutor(BaseA2AServer):
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.mcp_client = SimpleMCPClient(mcp_server_path)

    def handle_message(self, message: Message) -> Message:
        return asyncio.run(self.execute_task(message))

    async def execute_task(self, message: Message) -> Message:
        import pandas as pd
        import io

        # Parse args
        content_obj = message.content
        args = {}
        if hasattr(content_obj, 'text'):
            try:
                args = json.loads(content_obj.text)
            except: pass
        
        files_to_read = args.get("files", [])
        combined_analysis = ""

        try:
            await self.mcp_client.connect()
            for file in files_to_read:
                print(f"[ParquetAnalyzer] Reading {file} via MCP...")
                # We need to read the file as binary or ensure we can access it. 
                # Since the current MCP 'read_file' tools might return text, 
                # we technically need a way to read binary or load it directly if it's a local file path.
                # Assuming the agent has access to the filesystem for now as a simplification,
                # or we use a tool that supports binary. 
                # However, typically MCP reads might be text-based.
                # Let's check if we can read it directly from disk if we share the volume.
                # If not, we might need a read_binary_file tool.
                # For this implementation, I will assume we can access the file path directly 
                # if the path is absolute or relative to the workspace.
                
                # Check if file exists locally (assuming shared FS for this agent)
                if os.path.exists(file):
                    df = pd.read_parquet(file)
                    
                    # Analyze
                    analysis = []
                    analysis.append(f"--- Analysis of {file} ---")
                    analysis.append(f"Shape: {df.shape}")
                    analysis.append(f"Columns: {', '.join(df.columns)}")
                    
                    # Data Types
                    analysis.append("Data Types:")
                    for col, dtype in df.dtypes.items():
                        analysis.append(f"  - {col}: {dtype}")
                        
                    # Missing Values
                    missing = df.isnull().sum()
                    if missing.any():
                        analysis.append("Missing Values:")
                        for col, count in missing[missing > 0].items():
                            analysis.append(f"  - {col}: {count}")
                    else:
                        analysis.append("No missing values.")
                        
                    # Basic Stats (numeric)
                    analysis.append("Basic Statistics (Numeric):")
                    desc = df.describe().to_string()
                    analysis.append(desc)
                    
                    combined_analysis += "\n".join(analysis) + "\n\n"
                else:
                    combined_analysis += f"File not found or inaccessible: {file}\n"

            await self.mcp_client.close()
            
            if not combined_analysis:
                return create_text_message("No analysis generated. Check file paths.", role=MessageRole.AGENCY)
            
            return create_text_message(combined_analysis, role=MessageRole.AGENT)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return create_text_message(f"Error analyzing parquet: {e}", role=MessageRole.SYSTEM)
