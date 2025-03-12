from llm_planner.actor.operator import Operator
from llm_planner.message import Message

from contextlib import AsyncExitStack
from anthropic import Anthropic

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import mcp.types as types

import os


class MCPService(Operator):
    # call directly to a mcp tool
    def __init__(self):
        super().__init__()

        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

        self.sessions = []
        self.tools = []
        self.available_tools = {}

        self.is_connected = False
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.server_path = os.path.join(current_dir, '..', '..', 'service',
                                        'echo_server.py')

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if self.is_connected:
            return

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not is_python:
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command,
                                              args=[server_script_path],
                                              env=None)

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools += response.tools
        available_tools = {
            tool.name: {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools
        }

        self.available_tools.update(available_tools)

        self.is_connected = True

        print("\nConnected to server with tools:",
              [tool.name for tool in self.tools])

    async def process(self, sender_id, message: Message):
        await self.connect_to_server(self.server_path)

        query = message["content"]
        tool_use_id = message["tool_use_id"]
        messages = [{"role": "user", "content": query}]

        query = message["content"]
        tool_name = message["tool_name"]
        tool_args = message["tool_args"]

        tool = self.available_tools[tool_name]

        # Execute tool call
        result = await self.session.call_tool(tool_name, tool_args)
        print(f"result : {result.content}")

        messages = []

        messages.append({"role": "assistant", "content": []})

        messages.append({
            "role":
                "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result.content
            }]
        })

        ret = message.spawn()
        ret["content"] = messages
        return ret

    async def finalize(self) -> None:
        await self.exit_stack.aclose()
