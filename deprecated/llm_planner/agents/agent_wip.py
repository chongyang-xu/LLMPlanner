from enum import Enum

from llm_planner.message import Message

import time
import asyncio


# Define an enumeration
class Dependency(Enum):
    STRICT = 1
    LAZY = 2


ALL_AGENTS = {}


class Agent:

    def __init__(self, agent_name):
        self.name = agent_name

        self.dependencies = {}
        self.dependent_agents = {}

        self.message_queue = []
        self.futures = {}

        ALL_AGENTS[agent_name] = self

    def add_dependency(self,
                       agent: 'Agent',
                       dependency_type: Dependency = Dependency.STRICT):
        self.dependencies[agent.name] = dependency_type
        self.dependent_agents[agent.name] = agent

    async def process(self, message: Message) -> None:
        pass

    async def initialize(self) -> None:
        pass

    async def finalize(self) -> None:
        pass

    async def can_process(self, message: Message) -> bool:
        # Logic to determine if the agent can process the message
        # based on its current state and dependencies
        return True

    def ask(self, message: Message) -> asyncio.Future:
        self.message_queue.append(message)
        self.futures[message.id] = asyncio.Future()

        class FutureValue:

            def __init__(self, futures, idx):
                self.futures = futures
                self.idx = idx

            #def value(self):
            # return read_value(self.futures[self.idx])

            def value(self):
                return self.futures[self.idx]

        return FutureValue(self.futures, message.id)


async def async_start(batching):
    while True:
        for name, agent in ALL_AGENTS.items():
            #print(f"iter on {name}: {agent}")
            msg_q = agent.message_queue
            if len(msg_q) < 1:
                continue
            if await agent.can_process(msg_q[0]):
                if not batching:
                    start_time = time.time()
                    await agent.process(msg_q[0])
                    print(
                        f"{name}: process: {time.time() - start_time : .4f} sec"
                    )
                else:
                    assert False

                del msg_q[0]

        await asyncio.sleep(0)


def start_agents(entry_func):
    global MAIN_LOOP
    MAIN_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(MAIN_LOOP)

    # Schedule the periodic task
    task1 = MAIN_LOOP.create_task(entry_func())
    task2 = MAIN_LOOP.create_task(async_start(False))

    try:
        MAIN_LOOP.run_until_complete(task1)  # Run the event loop indefinitely
        MAIN_LOOP.run_until_complete(task2)  # Run the event loop indefinitely

    except KeyboardInterrupt:
        print("Stopping the loop.")
    finally:
        MAIN_LOOP.close()


def read_value(fut):
    return MAIN_LOOP.run_until_complete(fut)
