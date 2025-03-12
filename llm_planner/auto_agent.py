from llm_planner.message import Message

from contextlib import AsyncExitStack
from anthropic import Anthropic

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.stdio import get_default_environment

from llm_planner.operators.mcp.MCPSettings import mcp_settings
from llm_planner.operators.mcp.MCPTool import MCPTool


class AutonomousAgent:
    # call mcp tools from multiple agents
    def __init__(self, tools=[], settings=mcp_settings):
        super().__init__()

        self.anthropic = Anthropic()

        self.sessions = []

        self.is_connected = False

        self.tools = tools
        self.settings = settings

        self.tool2desc = {}
        self.tool2session = {}
        self.tooldesc2session = {}
        self.tool2exit_stack = {}

    async def connect_to_server(self):
        """Connect to an MCP server
        """
        if self.is_connected:
            return

        for t in self.tools:
            ss = self.settings.servers[t]
            if ss.transport == "stdio":
                if ss.command is None or ss.args is None:
                    raise ValueError(f"Command and args are required: {t}")

                server_params = StdioServerParameters(
                    command=ss.command,
                    args=ss.args,
                    env={
                        **get_default_environment(),
                        **(ss.env or {})
                    },
                )

                self.tool2exit_stack[t] = AsyncExitStack()

                stdio_transport = await self.tool2exit_stack[
                    t].enter_async_context(stdio_client(server_params))
                stdio_tmp, write_tmp = stdio_transport
                self.tool2session[t] = await self.tool2exit_stack[
                    t].enter_async_context(ClientSession(stdio_tmp, write_tmp))

                await self.tool2session[t].initialize()
            else:
                assert False, f"URL is required for SSE transport {t}"

        print(f"End of tools init")

        for t in self.tools:
            response = await self.tool2session[t].list_tools()
            self.tool2desc[t] = response.tools

            for rt in self.tool2desc[t]:
                print(f"{t} : Connected to server with tools: {rt.name}")
                assert rt.name not in self.tooldesc2session, "can't duplicate tool names"
                self.tooldesc2session[rt.name] = self.tool2session[t]

        self.is_connected = True
        print(f"self.is_connected = True")

    async def process(self, message: Message):
        await self.connect_to_server()

        query = message["content"]
        """Process a query using Claude and available tools"""
        messages = [{"role": "user", "content": query}]

        available_tools = []
        for _, t_desc in self.tool2desc.items():
            for tool in t_desc:
                available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools)

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:

            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                tool = MCPTool(self.tooldesc2session[tool_name], tool_name)
                result = await tool.call(tool_args)

                final_text.append(
                    f"[Calling tool {tool_name} with args {tool_args}]")
                final_text.append(
                    f"[Calling tool {tool_name} result: {result.content}]")
                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role":
                        "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result.content
                    }]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools)

                if len(response.content) > 0:
                    final_text.append(response.content[0].text)

        ret = message.spawn()
        ret["content"] = "\n".join(final_text)
        print(f"[DEBUG]: {ret['content']}")

        return ret

    async def finalize(self) -> None:
        for t in self.tools:
            await self.tool2exit_stack[t].aclose()
