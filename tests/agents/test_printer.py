from llm_planner.message import Message
from llm_planner.actor.operator import Operator
from llm_planner.actor.system import System

from llm_planner.operators.Printer import Printer


class Foo(Operator):

    async def process(self, sender_id, message):
        content = message['content']
        if content == 'start':
            printer = Printer()
            msg = message.spawn()
            msg["content"] = 'printer.txt'
            self.send(printer.id, msg)
            msg["content"] = 'ibm.png'
            self.send(printer.id, msg)

        else:
            print("---response---")
            print(message["pdf_file"])


system = System()
foo = Foo()

msg = Message()
msg['content'] = 'start'
system.send(None, foo.id, msg)

system.start()
