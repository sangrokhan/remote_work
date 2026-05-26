"""End-to-end MCP test: spawns real server subprocess, communicates via mcp client SDK."""
from __future__ import annotations
import asyncio
import json
import os
import sys
from pathlib import Path

SERVER_SCRIPT = Path(__file__).parent.parent / "scripts" / "run_server_mock.py"


async def _run_e2e() -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        env=dict(os.environ),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            tool_names = {t.name for t in tools_resp.tools}
            assert "build_get" in tool_names, f"build_get missing — got: {tool_names}"

            result = await session.call_tool(
                "build_get",
                {"target_node_id": "n4", "key_values": {"name": "eth0"}},
            )
            assert not result.isError, f"Tool error: {result.content}"
            payload = json.loads(result.content[0].text)
            xml = payload.get("xml", "")
            assert xml, f"build_get returned no XML: {payload}"
            assert "<source>" not in xml, "<source> must not appear in <get>"
            assert "eth0" in xml, "key value 'eth0' missing from filter"

            return xml


def test_mcp_build_get_e2e():
    xml = asyncio.run(_run_e2e())
    print(f"\n--- Generated XML ---\n{xml}--- End XML ---")
    assert "get" in xml
