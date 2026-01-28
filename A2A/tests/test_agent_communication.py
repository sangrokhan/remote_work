import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from agents.training_executor import TrainingExecutor
from agents.training_planning_agent import TrainingPlanningExecutor
from a2a.types import Message, Role
from utils.a2a_bridge import create_text_message, run_agent_sync

class TestAgentCommunication:
    
    def test_executor_capabilities(self):
        """Verify TrainingExecutor exposes capabilities schema correctly."""
        caps = TrainingExecutor.get_capabilities()
        
        assert "strategies" in caps
        assert "lora" in caps["strategies"]
        assert "argument_ranges" in caps
        assert "samples" in caps
        print("\n[Passed] Executor Capabilities Schema validated.")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_planning_agent_prompt_construction(self, mock_client_cls):
        """Verify Planning Agent constructs prompt with capabilities and logs interaction."""
        
        # Setup Mock LLM Response
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": json.dumps({
                "model_name": "distilgpt2",
                "strategy": "lora",
                "epochs": 3.0,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "dataset_path": "dummy.parquet",
                "reasoning": "Test reasoning"
            })
        }
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        # Setup Agent
        agent = TrainingPlanningExecutor()
        
        # Create Input Message
        input_payload = json.dumps({
            "analysis_report": "Dataset is small (1000 rows). Distribution is normal.",
            "user_goal": "Optimize for size."
        })
        message = create_text_message(input_payload, role=Role.user)

        # Execute using bridge
        result_message = await run_agent_sync(agent, message)
        
        # Verify Result
        assert result_message.role == Role.agent
        
        if result_message.parts:
            # Check result
            pass


        assert result_message.parts is not None
        assert len(result_message.parts) > 0
        response_text = result_message.parts[0].root.text
        
        plan = json.loads(response_text)
        assert plan["strategy"] == "lora"
        
        # Verify Capabilities were fetched and likely used
        call_args = mock_client.post.call_args
        assert call_args is not None
        prompt_sent = call_args[1]['json']['prompt']
        
        assert "EXECUTION CAPABILITIES" in prompt_sent
        assert "lora" in prompt_sent  # from capabilities injection
        print("\n[Passed] Planning Agent Prompt contains Capabilities.")

if __name__ == "__main__":
    # verification script usage without pytest
    t = TestAgentCommunication()
    t.test_executor_capabilities()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(t.test_planning_agent_prompt_construction())
    print("All tests passed!")
