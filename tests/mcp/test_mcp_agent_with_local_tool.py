import json
import time

from llm_planner.message import Message
from llm_planner.actor.system import System

from llm_planner.operators.mcp.MCPAgent import MCPAgent

###################
# run test
###################
mcp = MCPAgent(tools=["echo"])
msg = Message()
msg['content'] = 'could you echo this message back? message: "assds".'
mcp.send(mcp.id, msg)

msg2 = msg.spawn()
msg2['content'] = 'Could you also tell me the sum of 1 + 1 ?'
mcp.send(mcp.id, msg2)

System.start()
