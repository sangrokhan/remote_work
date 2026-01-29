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
            
            logger.info(f"[TrainingPlanningAgent] Received task content")
            
            args = {}
            if content_text:
                try:
                    args = json.loads(content_text)
                except:
                    logger.warning(f"[TrainingPlanningAgent] Content not JSON, treating as raw report")
                    args = {"analysis_report": content_text}
            
            report = args.get("analysis_report", "")
            goal = args.get("user_goal", "Train a high performing model on this data.")
            
            if not report:
                logger.error("[TrainingPlanningAgent] No analysis report provided")
                event_queue.enqueue_event(create_text_message("Error: No analysis report provided.", role=Role.agent))
                return

            # 0. Get Execution Capabilities
            capabilities = TrainingExecutor.get_capabilities()
            strategies_info = capabilities.get("strategies", {})
            defaults = capabilities.get("defaults", {})
            
            # Dynamic Strategy List for Prompt
            strategies_text = ""
            for name, info in strategies_info.items():
                strategies_text += f'- "{name}": {info["description"]}\n'

            # 1. Base Prompt (Dynamic Strategy Selection)
            base_prompt = f"""
ROLE: You are a Training Strategy Selector.
TASK: Analyze the report and select the best training strategy.

USER GOAL: {goal}

DATA ANALYSIS REPORT (NARRATIVE):
{report}

AVAILABLE STRATEGIES:
{strategies_text}
INSTRUCTIONS:
1. Select ONE strategy from the list above.
2. Output valid JSON with a SINGLE key "strategy".

OUTPUT FORMAT:
{{
  "strategy": "full_training"
}}
"""
            
            current_prompt = base_prompt
            max_retries = 3
            llm_service_url = os.environ.get("LLM_SERVICE_URL", "http://llm_service:8000")
            
            for attempt in range(max_retries):
                logger.info(f"[TrainingPlanner] Attempt {attempt+1}/{max_retries}...")
                
                # 2. Call LLM Service
                async with httpx.AsyncClient(timeout=1200.0) as client:
                    response = await client.post(
                        f"{llm_service_url}/generate",
                        json={"prompt": current_prompt, "max_length": 100, "temperature": 0.1} 
                    )
                    response.raise_for_status()
                    result = response.json()
                    generated_text = result.get("text", "")
                
                logger.info(f"[TrainingPlanner] Received LLM response: {generated_text[:200]}...")

                # 3. Robust JSON Extraction & Validation
                try:
                    found_plan = None
                    validation_error = None
                    
                    clean_text = generated_text.strip()
                    start_indices = [i for i, char in enumerate(clean_text) if char == '{']
                    
                    for start in start_indices:
                        for end in range(len(clean_text), start, -1):
                            if clean_text[end-1] == '}':
                                candidate = clean_text[start:end]
                                try:
                                    data = json.loads(candidate)
                                    if "strategy" not in data:
                                        continue
                                    if data["strategy"] not in strategies_info:
                                        # Invalid strategy selected
                                        continue
                                    found_plan = data
                                    break
                                except json.JSONDecodeError:
                                    continue
                        if found_plan:
                            break
                    
                    if found_plan:
                        selected_strategy = found_plan["strategy"]
                        strategy_schema = strategies_info[selected_strategy]
                        required_params = strategy_schema.get("required_params", [])
                        
                        logger.info(f"[TrainingPlanner] Selected strategy: {selected_strategy}. Filling required params: {required_params}")

                        # Fill parameters dynamically from defaults
                        final_plan = {"strategy": selected_strategy}
                        for param in required_params:
                            if param in defaults:
                                final_plan[param] = defaults[param]
                            else:
                                # Fallback or leave execution to fail if param missing
                                logger.warning(f"Parameter '{param}' required but no default found.")
                        
                        final_json = json.dumps(final_plan, indent=2)
                        
                        logger.info(f"[TrainingPlanner] Successfully parsed and validated Plan: {final_json}")
                        await event_queue.enqueue_event(create_text_message(final_json, role=Role.agent))
                        return
                    else:
                        validation_error = "Could not find a JSON object with required key 'strategy' or strategy is invalid."
                        
                except Exception as e:
                    validation_error = f"Parsing error: {str(e)}"

                # 4. Handle Failure & Feedback
                logger.warning(f"[TrainingPlanner] Validation failed: {validation_error}")
                
                feedback = f"\n\nERROR: Invalid output. {validation_error}"
                if "def " in generated_text or "import " in generated_text:
                     feedback += "\nSTOP WRITING PYTHON CODE. I need a JSON object, NOT a function definition."
                
                current_prompt += feedback + "\nPlease generate the JSON again."

            # If loop finishes without return
            erro_msg = f"Failed to generate valid plan after {max_retries} attempts."
            logger.error(erro_msg)
            await event_queue.enqueue_event(create_text_message(erro_msg, role=Role.agent))

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Planning failed: {str(e)}", exc_info=True)
            event_queue.enqueue_event(create_text_message(f"Planning failed: {str(e)}", role=Role.agent))
