import asyncio
import os
import json
import uuid
from typing import Dict, Any

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import Message, Role
from utils.a2a_bridge import create_text_message

# Move transformers imports to top level for better testing/mocking
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    pass

# Reuse existing utils
from utils.mcp_client_helper import SimpleMCPClient

class SummarizerExecutor(AgentExecutor):
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.mcp_client = SimpleMCPClient(mcp_server_path)

    def cancel(self) -> None:
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        # Extract arguments
        content_parts = message.parts
        content_text = content_parts[0].root.text if content_parts else ""
        args = {}
        # If accessing content object directly 
        if content_text: 
             try:
                args = json.loads(content_text)
             except:
                pass
        elif isinstance(content_obj, dict): # Should not happen if strictly typed Message
             pass # Already dict?
        
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
                    combined_text += result.content[0].root.text
                else:
                    combined_text += f"\n--- Failed to read {file} ---\n"
            await self.mcp_client.close()
        except Exception as e:
            await event_queue.enqueue_event(create_text_message(f"Error reading files: {e}", role=Role.agent))
            return

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
            
            await event_queue.enqueue_event(create_text_message(summary, role=Role.agent))

        except Exception as e:
            import traceback
            traceback.print_exc()
            await event_queue.enqueue_event(create_text_message(f"Error summarizing: {e}", role=Role.agent))


class EmailExecutor(AgentExecutor):
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.mcp_client = SimpleMCPClient(mcp_server_path)

    def cancel(self) -> None:
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        # Parse args
        content_parts = message.parts
        content_text = content_parts[0].root.text if content_parts else ""
        args = {}
        if content_text:
            try: 
                args = json.loads(content_text)
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

            await event_queue.enqueue_event(create_text_message(f"Email sent successfully. File: {filename}", role=Role.agent))

        except Exception as e:
            await event_queue.enqueue_event(create_text_message(f"Error sending email: {e}", role=Role.agent))


class ParquetAnalyzerExecutor(AgentExecutor):
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.mcp_client = SimpleMCPClient(mcp_server_path)

    def cancel(self) -> None:
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        import pandas as pd
        from utils.analysis_utils import (
            analyze_distribution, 
            analyze_correlation, 
            detect_outliers, 
            generate_llm_summary,
            analyze_dataset_size,
            analyze_text_metrics
        )

        # Parse args
        content_parts = message.parts
        content_text = content_parts[0].root.text if content_parts else ""
        args = {}
        if content_text:
            try:
                args = json.loads(content_text)
            except: pass
        
        files_to_read = args.get("files", [])
        combined_analysis = ""

        try:
            await self.mcp_client.connect()
            for file in files_to_read:
                print(f"[ParquetAnalyzer] Reading {file} via MCP...")
                
                # Check if file exists locally (assuming shared FS for this agent)
                if os.path.exists(file):
                    df = pd.read_parquet(file)
                    
                    # 1. Perform Analysis
                    dist_results = analyze_distribution(df)
                    corr_results = analyze_correlation(df)
                    outlier_results = detect_outliers(df)
                    
                    size_results = analyze_dataset_size(df)
                    text_results = analyze_text_metrics(df)
                    
                    full_results = {
                        "distribution": dist_results,
                        "correlation": corr_results,
                        "outliers": outlier_results,
                        "size_info": size_results,
                        "text_info": text_results
                    }
                    
                    # 2. Generate Summary
                    report = generate_llm_summary(full_results, filename=file)
                    combined_analysis += report + "\n\n"

                else:
                    combined_analysis += f"File not found or inaccessible: {file}\n"

            await self.mcp_client.close()
            
            if not combined_analysis:
                await event_queue.enqueue_event(create_text_message("No analysis generated. Check file paths.", role=Role.agent))
                return
            
            await event_queue.enqueue_event(create_text_message(combined_analysis, role=Role.agent))

        except Exception as e:
            import traceback
            traceback.print_exc()
            await event_queue.enqueue_event(create_text_message(f"Error analyzing parquet: {e}", role=Role.agent))
