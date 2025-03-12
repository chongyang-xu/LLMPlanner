from llm_planner.actor.operator import Operator


class MCPTool(Operator):
    # call directly to a mcp tool
    def __init__(self, session, name):
        super().__init__()

        self.session = session
        self.tool_name = name

    async def call(self, tool_args):
        return await self.session.call_tool(self.tool_name, tool_args)
