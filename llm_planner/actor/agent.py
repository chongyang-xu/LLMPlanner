from .actor import Actor
from ..message import Message

from deepdiff import DeepHash  # for caching
from diskcache import Cache  # for caching

import asyncio

from enum import Enum


# Define an enumeration
class Dependency(Enum):
    STRICT = 1
    LAZY = 2


class Agent(Actor):

    def __init__(self,
                 return_value=False,
                 with_batching=False,
                 with_caching=False):
        super().__init__()
        self.return_value = return_value
        self.futures = {}
        self.with_batching = with_batching
        self.with_caching = with_caching

        if self.with_caching:
            cache_dir = f"/tmp/llm_planner/agent"
            self.cache = Cache(cache_dir)

        if self.with_batching:
            self.batch = []
            self.batch_size = 2

    def add_dependency(self,
                       agent: 'Agent',
                       dependency_type: Dependency = Dependency.STRICT):
        pass

    async def initialize(self) -> None:
        pass

    async def finalize(self) -> None:
        pass

    async def can_process(self, message) -> bool:
        # Logic to determine if the agent can process the message
        # based on its current state and dependencies
        return True

    async def on_receive(self, sender_id, message):
        if self.with_caching:
            if message['response'] is not None:  # response message
                assert message[
                    "content"] is None  # with request in the response message

                if self.with_batching:
                    await self.on_process_batch(sender_id, message)
                else:
                    await self.on_process(sender_id, message)

            else:  # request message
                assert message['content'] is not None
                content = message["content"]
                key = DeepHash(content)[content]

                entries = self.cache.get(key, [])

                if len(entries) > 0:
                    if entries[0] is not None:
                        msg = Message()
                        msg['request_message'] = message
                        msg['response'] = entries[0]
                        self.send(sender_id, msg)
                else:
                    if self.with_batching:
                        await self.on_process_batch(sender_id, message)
                    else:
                        await self.on_process(sender_id, message)
        else:
            if self.with_batching:
                await self.on_process_batch(sender_id, message)
            else:
                await self.on_process(sender_id, message)

    async def on_process(self, sender_id, message):
        await self.process(sender_id, message)

    async def on_process_batch(self, sender_id, message):
        if len(self.batch) < self.batch_size:
            self.batch.append((sender_id, message))

        if len(self.batch) >= self.batch_size:
            ids, msgs = zip(*self.batch)
            await self.process_batch(ids, msgs)
            self.batch.clear()

    async def process(self, sender_id, message):
        pass

    async def process_batch(self, sender_ids, messages):
        pass

    def send(self, recipient_id, message):
        super().send(recipient_id, message)

        if self.with_caching:
            if message['response'] is not None:
                assert message["request_message"] is not None

                request_content = message["request_message"]["content"]
                key = DeepHash(request_content)[request_content]

                # assert not cache miss with request_content
                entries = self.cache.get(key, [])
                if len(entries) == 0:
                    self.cache.set(key, [message["response"]])

        if not self.return_value:
            return None

        self.futures[message.id] = asyncio.Future()

        class FutureValue:

            def __init__(self, futures, idx):
                self.futures = futures
                self.idx = idx

            def value(self):
                return self.futures[self.idx]

        return FutureValue(self.futures, message.id)
