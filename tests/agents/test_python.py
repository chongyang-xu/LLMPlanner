from llm_planner.message import Message
from llm_planner.actor.agent import Agent
from llm_planner.actor.system import System

from llm_planner.agents.Python import Python


class Foo(Agent):

    async def process(self, sender_id, message):
        content = message['content']
        if content == 'start':
            pi = Python()
            msg = message.spawn()
            msg["content"] = "with open('test_python.txt', 'w') as f:\n\tf.write('test')\n\tprint('test print')"
            self.send(pi.id, msg)

        else:
            print("---response---")
            print(message["response"])


system = System()
foo = Foo()

msg = Message()
msg['content'] = 'start'
system.send(None, foo.id, msg)

system.start()
