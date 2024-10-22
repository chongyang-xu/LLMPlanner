import asyncio
from llm_planner.actor.agent import Agent
from llm_planner.message import Message


class Human(Agent):

    def __init__(self):
        super().__init__()

    async def async_input(self, prompt):
        # Use asyncio.to_thread to avoid blocking by running input() in a thread
        return await asyncio.to_thread(input, prompt)

    async def process(self, sender_id, message: Message):
        if message["content"] is not None:
            human_response = await self.async_input(
                f"Please response to this request:\n{message['content']}\n")
            # Prepare the response message
            msg = Message()
            msg['request_message'] = message
            msg['response'] = human_response

            self.send(sender_id, msg)
