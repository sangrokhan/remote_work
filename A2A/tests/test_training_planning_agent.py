import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch
from agents.training_planning_agent import TrainingPlanningExecutor
from python_a2a.models.message import Message, MessageRole
from python_a2a.models.content import TextContent

@pytest.mark.asyncio
async def test_training_planning_executor():
    # Mock response from LLM
    mock_plan = {
        "model_name": "gpt2",
        "dataset_path": "placeholder",
        "strategy": "lora",
        "epochs": 3.0,
        "batch_size": 8,
        "learning_rate": 2e-4,
        "reasoning": "Small dataset requires efficient fine-tuning."
    }
    
    # Simulate LLM outputting text with JSON inside
    llm_output_text = f"Here is the plan:\n```json\n{json.dumps(mock_plan)}\n```"

    with patch("httpx.AsyncClient") as MockClient:
        # Setup Mock
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": llm_output_text}
        mock_response.raise_for_status = MagicMock()
        
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = asyncio.Future()
        mock_client_instance.post.return_value.set_result(mock_response)
        
        # Async context manager mock
        MockClient.return_value.__aenter__.return_value = mock_client_instance
        MockClient.return_value.__aexit__.return_value = None

        # Execute
        executor = TrainingPlanningExecutor()
        input_msg = Message(
            role=MessageRole.USER, 
            content=TextContent(text=json.dumps({"analysis_report": "High variance in column X", "user_goal": "Test"}))
        )
        
        response = await executor.execute_task(input_msg)
        
        # Verify
        assert response.role == MessageRole.AGENT
        resp_data = json.loads(response.content.text)
        assert resp_data["model_name"] == "gpt2"
        assert resp_data["strategy"] == "lora"
        
        print("Test Passed: Plan correctly parsed.")

if __name__ == "__main__":
    asyncio.run(test_training_planning_executor())
