import json
import time

from llm_planner.message import Message
from llm_planner.actor.system import System

from llm_planner.operators.mcp.MCPAgent import MCPAgent

###################
# run test
###################
mcp = MCPAgent(tools=["time", "fetch"])
msg = Message()
msg['content'] = 'could you tell me the content of https://raw.githubusercontent.com/lastmile-ai/mcp-agent/refs/heads/main/examples/mcp_hello_world/README.md ?'
mcp.send(mcp.id, msg)

msg2 = msg.spawn()
msg2['content'] = 'What time is it now in CEST ?'
mcp.send(mcp.id, msg2)

System.start()
