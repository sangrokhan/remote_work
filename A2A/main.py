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

from utils.logger import get_logger

logger = get_logger("A2A.Main")

# Ensure local imports work
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))

from agents.executors import SummarizerExecutor, EmailExecutor, ParquetAnalyzerExecutor
from agents.training_executor import TrainingExecutor
from agents.training_planning_agent import TrainingPlanningExecutor
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

class AnalyzeRequest(BaseModel):
    files: list[str]

class AutoTrainRequest(BaseModel):
    files: list[str]
    goal: str = "Train an optimal model based on the data."

async def process_daily_routine(task_id: str):
    logger.info(f"[Agent Core] Starting daily routine (Task {task_id})...")
    tasks[task_id]["status"] = "running"
    tasks[task_id]["logs"] = []
    
    def log(msg):
        logger.info(f"[Task {task_id}] {msg}")
        tasks[task_id]["logs"].append(msg)

    # Path to the MCP server script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(base_dir, "mcp", "server.py")

    # Initialize Executors
    summarizer = SummarizerExecutor(mcp_server_path)
    emailer = EmailExecutor(mcp_server_path)
    trainer = TrainingExecutor(mcp_server_path)
    
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
        logger.error(f"Error during routine: {str(e)}", exc_info=True)
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

    logger.info(f"[Agent Core] Task {task_id} finished.")

@app.post("/api/run-daily-routine")
async def run_routine(background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "logs": []}
    background_tasks.add_task(process_daily_routine, task_id)
    return {"status": "started", "task_id": task_id, "message": "Daily routine started in background"}

@app.post("/api/analyze-parquet")
async def analyze_parquet(request: AnalyzeRequest):
    # Path to the MCP server script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(base_dir, "mcp", "server.py")
    
    analyzer = ParquetAnalyzerExecutor(mcp_server_path)
    
    payload = json.dumps({"files": request.files})
    message = create_text_message(payload, role=MessageRole.USER)
    
    response = await analyzer.execute_task(message)
    
    result_text = ""
    if hasattr(response.content, 'text'):
        result_text = response.content.text
    else:
        result_text = str(response.content)
        
    return {"status": "success", "analysis": result_text}

@app.post("/api/auto-train-pipeline")
async def auto_train_pipeline(request: AutoTrainRequest, background_tasks: BackgroundTasks):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(base_dir, "mcp", "server.py")
    
    # 1. Initialize Agents
    analyzer = ParquetAnalyzerExecutor(mcp_server_path)
    planner = TrainingPlanningExecutor(mcp_server_path)
    trainer = TrainingExecutor(mcp_server_path)
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "starting", "logs": []}

    async def run_pipeline(tid: str, files: list, goal: str):
        def log(msg):
            logger.info(f"[Pipeline {tid}] {msg}")
            tasks[tid]["logs"].append(msg)
            
        try:
            tasks[tid]["status"] = "running"
            
            # --- Phase 1: Data Analysis ---
            log("State: ANALYZING_DATA")
            payload = json.dumps({"files": files})
            msg = create_text_message(payload, role=MessageRole.USER)
            
            resp = await analyzer.execute_task(msg)
            analysis_text = resp.content.text if hasattr(resp.content, 'text') else str(resp.content)
            log(f"Analysis Complete. Report length: {len(analysis_text)}")
            
            # --- Phase 2: Training Planning ---
            log("State: PLANNING_TRAINING")
            plan_payload = json.dumps({
                "analysis_report": analysis_text,
                "user_goal": goal
            })
            plan_msg = create_text_message(plan_payload, role=MessageRole.USER)
            
            resp = await planner.execute_task(plan_msg)
            plan_json_str = resp.content.text if hasattr(resp.content, 'text') else str(resp.content)
            
            try:
                plan_data = json.loads(plan_json_str)
                log(f"Plan Generated: Strategy={plan_data.get('strategy')}, Model={plan_data.get('model_name')}")
            except:
                log(f"Plan generation failed or invalid JSON: {plan_json_str}")
                tasks[tid]["status"] = "failed"
                return

            # Validate/Inject Dataset Path if missing (Planner might not know absolute path)
            # Assuming the first file in list is the primary dataset for training
            if not plan_data.get("dataset_path") or "placeholder" in plan_data.get("dataset_path"):
                 plan_data["dataset_path"] = files[0]
                 log(f"Injected dataset path: {files[0]}")

            # --- Phase 3: Training Execution ---
            log("State: EXECUTING_TRAINING")
            train_payload = json.dumps(plan_data)
            train_msg = create_text_message(train_payload, role=MessageRole.USER)
            
            resp = await trainer.execute_task(train_msg)
            result_text = resp.content.text if hasattr(resp.content, 'text') else str(resp.content)
            
            log(f"Training Result: {result_text}")
            tasks[tid]["status"] = "completed"
            tasks[tid]["result"] = result_text

        except Exception as e:
            logger.error(f"Pipeline Error: {str(e)}", exc_info=True)
            log(f"Pipeline Error: {str(e)}")
            tasks[tid]["status"] = "failed"
            tasks[tid]["error"] = str(e)

    background_tasks.add_task(run_pipeline, task_id, request.files, request.goal)
    
    return {
        "status": "protocol_initiated",
        "task_id": task_id,
        "message": "AutoML Pipeline started. Check status at /api/task/{task_id}"
    }

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
