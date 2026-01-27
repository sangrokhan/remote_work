import sys
import os
import asyncio
import pandas as pd
import json

# Ensure we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.executors import ParquetAnalyzerExecutor
from python_a2a.models.message import Message, MessageRole
from python_a2a.utils.conversion import create_text_message

async def run_test():
    print("--- Starting Parquet Agent Test ---")
    
    # 1. Create a dummy parquet file
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 40],
        'salary': [50000.0, 60000.0, 75000.0, None], # Test missing value
        'department': ['HR', 'Engineering', 'Marketing', 'Engineering']
    }
    df = pd.DataFrame(data)
    test_file = "test_data.parquet"
    df.to_parquet(test_file)
    print(f"Created dummy file: {test_file}")
    
    try:
        # 2. Initialize Executor
        # Point to the mcp server script relative to this test file or project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mcp_server_path = os.path.join(base_dir, "mcp", "server.py")
        
        executor = ParquetAnalyzerExecutor(mcp_server_path)
        
        # 3. Create a Message invoking the agent
        payload = json.dumps({"files": [test_file]})
        message = create_text_message(payload, role=MessageRole.USER)
        
        # 4. Execute
        print("Executing task...")
        response = await executor.execute_task(message)
        
        # 5. Print Result
        print("--- Agent Response ---")
        if hasattr(response.content, 'text'):
            print(response.content.text)
        else:
            print(response.content)
            
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nRemoved dummy file: {test_file}")

if __name__ == "__main__":
    asyncio.run(run_test())
