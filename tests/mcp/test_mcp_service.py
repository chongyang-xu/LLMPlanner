import json
import time

from llm_planner.message import Message
from llm_planner.actor.system import System

from llm_planner.operators.mcp.MCPService import MCPService

###################
# run test
###################
mcp = MCPService()
msg = Message()
msg['content'] = 'could you echo this message back? message: "assds".'
msg["tool_use_id"] = 1
msg["tool_name"] = 'echo_tool'
msg["tool_args"] = {'message': 'assds'}

mcp.send(mcp.id, msg)

System.start()
