import asyncio
import os
import sys
import json
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# Ensure local imports work
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))

from agents.executors import SummarizerExecutor, EmailExecutor
from python_a2a.utils.conversion import create_text_message
from python_a2a.models.message import MessageRole

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task store
tasks: Dict[str, Dict[str, Any]] = {}

class CommandRequest(BaseModel):
    command: str

async def process_daily_routine(task_id: str):
    print(f"[Agent Core] Starting daily routine (Task {task_id})...")
    tasks[task_id]["status"] = "running"
    tasks[task_id]["logs"] = []
    
    def log(msg):
        print(f"[Task {task_id}] {msg}")
        tasks[task_id]["logs"].append(msg)

    # Path to the MCP server script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(base_dir, "mcp", "server.py")

    # Initialize Executors
    summarizer = SummarizerExecutor(mcp_server_path)
    emailer = EmailExecutor(mcp_server_path)
    
    try:
        # --- Step 1: Request Summary ---
        log("Step 1: Requesting Summary...")
        files = ["projects.csv", "updates.txt"]
        summarize_payload = json.dumps({"files": files})
        
        input_msg = create_text_message(summarize_payload, role=MessageRole.USER)
        
        response_msg = await summarizer.execute_task(input_msg)
        summary_text = ""
        if hasattr(response_msg.content, 'text'):
            summary_text = response_msg.content.text
        else:
            summary_text = str(response_msg.content)
        
        log(f"Received Summary: {summary_text[:50]}...")

        if "Error" in summary_text:
            log("Aborting due to error.")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["result"] = summary_text
            return

        # --- Step 2: Send Email ---
        log("Step 2: Sending Email...")
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
        
        resp_text = ""
        if hasattr(response_msg.content, 'text'):
            resp_text = response_msg.content.text
        else:
            resp_text = str(response_msg.content)
        
        log(f"Emailer Response: {resp_text}")
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = "Daily routine completed successfully."
        
    except Exception as e:
        log(f"Error during routine: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

    print(f"[Agent Core] Task {task_id} finished.")

@app.post("/api/run-daily-routine")
async def run_routine(background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "logs": []}
    background_tasks.add_task(process_daily_routine, task_id)
    return {"status": "started", "task_id": task_id, "message": "Daily routine started in background"}

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.post("/api/command")
async def run_command(request: CommandRequest):
    if "daily" in request.command.lower() or "routine" in request.command.lower():
         return await run_routine(BackgroundTasks())
    
    return {"status": "ignored", "message": f"Command received: {request.command}. (Only 'daily routine' is currently implemented)"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
