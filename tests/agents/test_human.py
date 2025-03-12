from llm_planner.message import Message
from llm_planner.actor.operator import Operator
from llm_planner.actor.system import System

from llm_planner.operators.Human import Human


class Foo(Operator):

    async def process(self, sender_id, message):
        content = message['content']
        if content == 'start':
            human = Human()
            msg = message.spawn()
            msg["content"] = 'Hi, how are you?'
            self.send(human.id, msg)
        else:
            print("---Human response---")
            print(message["response"])


system = System()
foo = Foo()

msg = Message()
msg['content'] = 'start'
system.send(None, foo.id, msg)

system.start()
