import asyncio

from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.message import Message


class ParentAgent(Agent):

    async def process(self, sender_id, message):
        content = message['content']
        if content == 'start':
            print("ParentAgent: Starting ChildAgent")
            child_Agent = ChildAgent()
            msg = Message()
            msg['content'] = 'Hello Child'
            self.send(child_Agent.id, msg)

        elif 'Processed' in content:
            print(f"ParentAgent: Received response '{content}' from ChildAgent")
            print("ParentAgent: Starting SecondAgent")
            second_Agent = SecondAgent()
            msg = Message()
            msg['content'] = content
            self.send(second_Agent.id, msg)
        else:
            print(f"ParentAgent: Received unknown message '{content}'")


class ChildAgent(Agent):

    async def process(self, sender_id, message):
        content = message['content']
        print(f"ChildAgent: Received '{content}' from {sender_id}")
        response = f"Processed '{content}'"
        msg = Message()
        msg['content'] = response
        self.send(sender_id, {'content': response})


class SecondAgent(Agent):

    async def process(self, sender_id, message):
        content = message['content']
        print(f"SecondAgent: Received '{content}' from {sender_id}")


system = System()
parent_agent = ParentAgent()
msg = Message()
msg['content'] = 'start'
system.send(None, parent_agent.id, msg)
system.start()
