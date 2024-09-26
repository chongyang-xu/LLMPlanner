from llm_planner.agents.agent_wip import Agent, read_value
from llm_planner.message import Message

from llm_planner.service.hf_serve import HFServe

MINI_LLM_PATH = "/DS/dsg-ml/nobackup/cxu/weights/Qwen2-0.5B/"

import asyncio
from concurrent.futures import ThreadPoolExecutor

import time


class MiniLLM(Agent):

    def __init__(self, agent_name, max_token=16):
        super().__init__(agent_name)

        policy_param_ = {"model": MINI_LLM_PATH, "max_token": max_token}
        self.serve = HFServe(None, policy_param_)

        self.futures = {}

    async def process(self, message: Message) -> None:

        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(executor, self.blocking_serve,
                                          message)
            self.futures[message.id] = future

    def blocking_serve(self, message: Message):
        start = time.time()

        msg = Message()
        if isinstance(message["prompt"], list):
            msg["ret"] = self.serve.work_on(message["prompt"])
        else:
            msg["ret"] = self.serve.work_on([message["prompt"]])

        end = time.time()

        # print(f"blocking_serve: {end-start:.2f} sec")
        # print(f"blocking_serve: {msg}")
        return msg
