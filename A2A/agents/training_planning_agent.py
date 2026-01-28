import os
import json
import httpx
from typing import Dict, Any

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import Message, Role
from utils.a2a_bridge import create_text_message
from agents.training_executor import TrainingExecutor
from utils.logger import get_logger

logger = get_logger("A2A.TrainingPlanner")

class TrainingPlanningExecutor(AgentExecutor):
    def __init__(self, mcp_server_path: str = ""):
        self.mcp_server_path = mcp_server_path
        # No MCP client needed strictly for pure reasoning, but consistent init is good

    def cancel(self) -> None:
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            message = context.message
            content_parts = message.parts
            content_text = content_parts[0].root.text if content_parts else ""
            # Input Expected: {"analysis_report": "...", "user_goal": "..."}
            
            logger.info(f"[TrainingPlanningAgent] Received task content")
            
            args = {}
            if content_text:
                try:
                    args = json.loads(content_text)
                except:
                    # If not JSON, treat whole text as analysis report
                    logger.warning(f"[TrainingPlanningAgent] Content not JSON, treating as raw report")
                    args = {"analysis_report": content_text}
            
            report = args.get("analysis_report", "")
            goal = args.get("user_goal", "Train a high performing model on this data.")
            
            logger.info(f"[TrainingPlanningAgent] User Goal: {goal}")
            if not report:
                logger.error("[TrainingPlanningAgent] No analysis report provided")
                event_queue.enqueue_event(create_text_message("Error: No analysis report provided.", role=Role.agent))
                return

            # 0. Get Execution Capabilities
            capabilities = TrainingExecutor.get_capabilities()
            logger.info(f"[TrainingPlanningAgent] Fetched Execution Capabilities: {json.dumps(capabilities, indent=2)}")

            # 1. Construct Prompt for the Planner LLM
            prompt = f"""
You are an expert Machine Learning Architect. 
Your goal is to design a training strategy based on a Data Analysis Report and available Execution Capabilities.

USER GOAL: {goal}

DATA ANALYSIS REPORT:
{report}

EXECUTION CAPABILITIES (Strictly follow these):
{json.dumps(capabilities, indent=2)}

You MUST output ONLY a valid JSON object. Do not include any pre-amble, explanation, or markdown formatting outside the JSON itself.
The JSON must follow the schema required by the Executor (see 'samples' in capabilities).

Heuristics:
- If dataset is small, prefer 'lora' or 'transfer'.
- If computing resources are mentioned as low, use 'lora' or 'layer_freezing'.
- If data is non-normal or complex, adjust epochs/LR accordingly.
- 'model_name' should be a standard HuggingFace model identifier.
"""

            # 2. Call LLM Service
            llm_service_url = os.environ.get("LLM_SERVICE_URL", "http://llm_service:8000")
            logger.info(f"[TrainingPlanner] Sending prompt to LLM service at {llm_service_url}...")
            
            async with httpx.AsyncClient(timeout=1200.0) as client:
                response = await client.post(
                    f"{llm_service_url}/generate",
                    json={"prompt": prompt, "max_length": 1000, "temperature": 0.2} 
                    # Low temp for deterministic planning
                )
                response.raise_for_status()
                result = response.json()
                generated_text = result.get("text", "")
            
            logger.info(f"[TrainingPlanner] Received LLM response: {generated_text}")

            # 3. Parse JSON from LLM output
            # Attempt to find JSON block if LLM adds text around it
            try:
                # Remove markdown code blocks if present
                clean_text = generated_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()

                # Naive cleaning: find first { and last }
                start_idx = clean_text.find("{")
                end_idx = clean_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = clean_text[start_idx:end_idx+1]
                    plan_data = json.loads(json_str)
                else:
                    plan_data = json.loads(clean_text)
                
                logger.info(f"[TrainingPlanner] Successfully parsed Plan: {json.dumps(plan_data, indent=2)}")

            except Exception as e:
                logger.error(f"[TrainingPlanner] JSON Parsing failed. Raw output: {generated_text}")
                # Fallback or error
                return create_text_message(f"Failed to generate valid plan. Raw: {generated_text}", role=Role.agent)
            
            # Return the plan as a JSON string
            event_queue.enqueue_event(create_text_message(json.dumps(plan_data, indent=2), role=Role.agent))

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Planning failed: {str(e)}", exc_info=True)
            event_queue.enqueue_event(create_text_message(f"Planning failed: {str(e)}", role=Role.agent))
