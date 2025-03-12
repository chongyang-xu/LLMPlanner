from llm_planner.auto_agent import AutonomousAgent
from llm_planner.actor.system import System

from llm_planner.message import Message

aa = AutonomousAgent(tools=["fetch", "echo"])
system = System()

msg = Message()
msg['content'] = 'Could you tell me the result of 1 + 1 ?'

system.run_agent(aa, msg)
