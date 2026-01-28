import os
import json
import httpx
from typing import Dict, Any

from python_a2a.models.message import Message, MessageRole
from python_a2a.utils.conversion import create_text_message
from python_a2a.server.base import BaseA2AServer

class TrainingPlanningExecutor(BaseA2AServer):
    def __init__(self, mcp_server_path: str = ""):
        self.mcp_server_path = mcp_server_path
        # No MCP client needed strictly for pure reasoning, but consistent init is good

    async def execute_task(self, message: Message) -> Message:
        try:
            content_obj = message.content
            # Input Expected: {"analysis_report": "...", "user_goal": "..."}
            args = {}
            if hasattr(content_obj, 'text'):
                try:
                    args = json.loads(content_obj.text)
                except:
                    # If not JSON, treat whole text as analysis report
                    args = {"analysis_report": content_obj.text}
            
            report = args.get("analysis_report", "")
            goal = args.get("user_goal", "Train a high performing model on this data.")
            
            if not report:
                return create_text_message("Error: No analysis report provided.", role=MessageRole.SYSTEM)

            # 1. Construct Prompt for the Planner LLM
            prompt = f"""
You are an expert Machine Learning Architect. 
Your goal is to design a training strategy based on a Data Analysis Report.

USER GOAL: {goal}

DATA ANALYSIS REPORT:
{report}

Start by analyzing the report briefly. Then, select the best training strategy and hyperparameters.
You MUST output a valid JSON object that matches the following schema exactly (no markdown, just JSON):

{{
    "model_name": "string (e.g. gpt2, bert-base-uncased, distilgpt2)",
    "dataset_path": "string (path to the processed dataset, infer from context or use placeholder)",
    "strategy": "string (one of: full_training, fine_tuning, transfer, lora, adapter, layer_freezing, continual, curriculum)",
    "epochs": float,
    "batch_size": int,
    "learning_rate": float,
    "lora_r": int (optional, default 8),
    "lora_alpha": int (optional, default 32),
    "freeze_layers": int (optional, default 0),
    "reasoning": "string (brief explanation of why this strategy was chosen)"
}}

Heuristics:
- If dataset is small, prefer 'lora' or 'transfer'.
- If computing resources are mentioned as low, use 'lora' or 'layer_freezing'.
- If data is non-normal or complex, adjust epochs/LR accordingly.
- 'model_name' should be a standard HuggingFace model identifier.
"""

            # 2. Call LLM Service
            llm_service_url = os.environ.get("LLM_SERVICE_URL", "http://llm_service:8000")
            print(f"[TrainingPlanner] Sending prompt to LLM service at {llm_service_url}...")
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{llm_service_url}/generate",
                    json={"prompt": prompt, "max_length": 1000, "temperature": 0.2} 
                    # Low temp for deterministic planning
                )
                response.raise_for_status()
                result = response.json()
                generated_text = result.get("text", "")

            # 3. Parse JSON from LLM output
            # Attempt to find JSON block if LLM adds text around it
            try:
                # Naive cleaning: find first { and last }
                start_idx = generated_text.find("{")
                end_idx = generated_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = generated_text[start_idx:end_idx+1]
                    plan_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except Exception as e:
                print(f"[TrainingPlanner] JSON Parsing failed. Raw output: {generated_text}")
                # Fallback or error
                return create_text_message(f"Failed to generate valid plan. Raw: {generated_text}", role=MessageRole.SYSTEM)
            
            # Return the plan as a JSON string
            return create_text_message(json.dumps(plan_data, indent=2), role=MessageRole.AGENT)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return create_text_message(f"Planning failed: {str(e)}", role=MessageRole.SYSTEM)

    def handle_message(self, message: Message) -> Message:
        import asyncio
        return asyncio.run(self.execute_task(message))
