from .actor import Actor

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
        cache_hit = False
        if self.with_caching:
            prompt = message["prompt"]
            key = DeepHash(prompt)[prompt]
            entries = self.cache.get(key, [])

            if len(entries) > 0:
                cache_hit = True

                if self.return_value:
                    self.system.actors[sender_id].futures[
                        message.id].set_result(entries[0])
                else:
                    pass
                    #self.system.actors[sender_id].futures[
                    #    message.id].set_result(None)
                return

        assert not cache_hit

        if self.with_batching:
            await self.on_process_batch(sender_id, message)
        else:
            await self.on_process(sender_id, message)

    async def on_process(self, sender_id, message):
        ret = await self.process(sender_id, message)

        if self.with_caching:
            prompt = message["prompt"]
            key = DeepHash(prompt)[prompt]
            self.cache.set(key, [ret])

        if self.return_value:
            self.system.actors[sender_id].futures[message.id].set_result(ret)
        else:
            pass
            #if sender_id is not None:
            #    self.system.actors[sender_id].futures[
            #        message.id].set_result(None)

    async def on_process_batch(self, sender_ids, messages):
        if len(self.batch) < self.batch_size:
            self.batch.append((sender_id, message))

        if len(self.batch) >= self.batch_size:
            ids, msgs = zip(*self.batch)
            rets = await self.process_batch(ids, msgs)

            if self.with_caching:
                for i in range(len(ids)):
                    prompt = msgs[i]["prompt"]
                    key = DeepHash(prompt)[prompt]
                    self.cache.set(key, [rets[i]])

            if self.return_value:
                for i in range(len(ids)):
                    self.system.actors[ids[i]].futures[msgs[i].id].set_result(
                        rets[i])
            else:
                pass
                #for i in range(len(ids)):
                #    self.system.actors[ids[i]].futures[
                #        msgs[i].id].set_result(None)

            self.batch.clear()

    async def process(self, sender_id, message):
        pass

    async def process_batch(self, sender_ids, messages):
        pass

    def send(self, recipient_id, message):
        super().send(recipient_id, message)

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
