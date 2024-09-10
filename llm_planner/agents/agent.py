import pykka

from llm_planner.message import Message

from deepdiff import DeepHash  # for caching
from diskcache import Cache  # for caching
import uuid


class Agent(pykka.ThreadingActor):

    def __init__(self):
        super().__init__()

    def on_receive(self, message: Message):
        self.receive(message)

    def receive(self, message: Message):
        print(message)


class LLMAgent(pykka.ThreadingActor):

    def __init__(self):
        super().__init__()
        cache_dir = f"/tmp/llm_planner/{uuid.uuid1()}"
        self.cache = Cache(cache_dir)
        self.batch = []
        self.batch_size = 4

    def on_receive(self, message: Message):

        prompt = message["prompt"]

        # caching
        key = DeepHash(prompt)[prompt]
        entries = self.cache.get(key, [])

        msg = Message()
        if len(entries) > 0:
            msg["ret"] = entries
            return msg

        # batching
        if len(self.batch) < self.batch_size:
            self.batch.append(prompt)

        if len(self.batch) < self.batch_size:
            return Message(content={"ret": [""]})
        else:
            batch_msg = Message()
            batch_msg["prompt"] = self.batch
            ret_msg = self.receive(batch_msg)
            self.batch.clear()
            return ret_msg

    def receive(self, message: Message):
        print(message)


def agent_stop_all():
    pykka.ActorRegistry.stop_all(block=True)


AgentRef = pykka._ref.ActorRef
