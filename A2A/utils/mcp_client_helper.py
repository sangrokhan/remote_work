import asyncio
import os
import sys
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class SimpleMCPClient:
    def __init__(self, server_script_path):
        self.server_script_path = server_script_path
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def connect(self):
        # We assume python is in the path
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path],
            env=os.environ.copy() # Pass current env
        )

        # Initialize connection
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.read, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.session.initialize()

    async def call_tool(self, tool_name, arguments):
        if not self.session:
            await self.connect()

        result = await self.session.call_tool(tool_name, arguments)
        return result

    async def close(self):
        await self.exit_stack.aclose()
