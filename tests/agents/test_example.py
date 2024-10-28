from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.message import Message


class Printer(Agent):

    async def process(self, sender_id, message):
        content = message['content']
        print(content)


printer = Printer()
system = System()

msg = Message()
msg['content'] = 'start'
system.send(None, printer.id, msg)

system.start()
