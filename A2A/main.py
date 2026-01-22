import asyncio
import os
import sys
import json
import uuid

# If running as a script, ensure local imports work
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))

from agents.executors import SummarizerExecutor, EmailExecutor
from python_a2a.utils.conversion import create_text_message
from python_a2a.models.message import MessageRole

async def run_daily_routine():
    print("[Main] Starting daily routine using A2A Executors (BaseA2AServer)...")

    # Path to the MCP server script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(base_dir, "mcp", "server.py")

    # Initialize Executors
    summarizer = SummarizerExecutor(mcp_server_path)
    emailer = EmailExecutor(mcp_server_path)

    # --- Step 1: Request Summary ---
    print("\n[Main] Step 1: Requesting Summary...")
    files = ["projects.csv", "updates.txt"]
    summarize_payload = json.dumps({"files": files})
    
    # Create input message
    input_msg = create_text_message(summarize_payload, role=MessageRole.USER)
    
    # Execute (directly calling async method to avoid blocking loop issues)
    response_msg = await summarizer.execute_task(input_msg)

    # Parse response
    summary_text = ""
    if hasattr(response_msg.content, 'text'):
        summary_text = response_msg.content.text
    else:
        summary_text = str(response_msg.content)

    print(f"[Main] Received Summary: {summary_text[:50]}...")

    if "Error" in summary_text:
        print("Aborting due to error.")
        return

    # --- Step 2: Send Email ---
    print("\n[Main] Step 2: Sending Email...")
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
    
    # Check response text
    resp_text = ""
    if hasattr(response_msg.content, 'text'):
        resp_text = response_msg.content.text
    else:
        resp_text = str(response_msg.content)
    
    print(f"[Main] Emailer Response: {resp_text}")
    print("[Main] Daily routine complete.")

if __name__ == "__main__":
    asyncio.run(run_daily_routine())
