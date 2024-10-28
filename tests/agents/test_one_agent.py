from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.message import Message


class Foo(Agent):

    async def process(self, sender_id, message):
        content = message['content']
        if content == 'start':
            print("Foo: Starting")
            msg = message.spawn()
            msg["content"] = "hello"
            self.send(self.id, msg)
        else:
            print(f"Foo: Received '{content}'")


system = System()
foo = Foo()

msg = Message()
msg['content'] = 'start'
system.send(None, foo.id, msg)

system.start()
