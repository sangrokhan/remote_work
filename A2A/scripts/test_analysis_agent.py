import asyncio
import os
import sys
import pandas as pd
import json
from unittest.mock import MagicMock

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.executors import ParquetAnalyzerExecutor
from python_a2a.models.message import Message, MessageRole
# from python_a2a.models.message import MessageContent # Not available


# Mock MCP Client to avoid needing the actual server
class MockMCPClient:
    def __init__(self, path): pass
    async def connect(self): pass
    async def close(self): pass
    async def call_tool(self, name, args): return None

# Monkey patch the client in agents.executors if needed, 
# or just pass a dummy path since the class instantiates SimpleMCPClient internally.
# But execute_task uses self.mcp_client. 
# We can mock it after instantiation.

async def main():
    # 1. Generate Mock Data
    print("Generating mock data...")
    from scripts.generate_mock_data import generate_data
    output_file = "data/test_mock.parquet"
    generate_data(num_days=1, anomaly_ratio=0.2, output_path=output_file)
    
    # 2. Setup Executor
    executor = ParquetAnalyzerExecutor("dummy/path")
    executor.mcp_client = MockMCPClient("dummy/path")
    
    # 3. Create Message
    payload = json.dumps({"files": [output_file]})
    # Create a dummy message object. 
    # Since Message is a pydantic model, we can instantiate it or better yet, use the create_text_message util if available or just construct it.
    # The code expects message.content to have .text attribute or be content object.
    
    # Let's try to strictly match what execute_task expects
    # It checks `content_obj.text`
    
    class MockContent:
        def __init__(self, text):
            self.text = text
            
    class MockMessage:
        def __init__(self, text):
            self.content = MockContent(text)
            
    msg = MockMessage(payload)
    
    # 4. Run Execution
    print("\nRunning Analysis...")
    response = await executor.execute_task(msg)
    
    # 5. Print Result
    print("\nXXX ANALYSIS RESULT XXX\n")
    if hasattr(response.content, 'text'):
        print(response.content.text)
    else:
        print(response.content)
    print("\nXXX END RESULT XXX\n")
    
    # 6. Check for specific markers
    text = response.content.text if hasattr(response.content, 'text') else str(response.content)
    
    has_json = "```json" in text
    has_skew = "Skew:" in text
    
    if has_json and has_skew:
        print("✅ SUCCESS: Output contains expected Markdown and JSON block.")
    else:
        print("❌ FAILURE: Output missing required format elements.")

if __name__ == "__main__":
    asyncio.run(main())
