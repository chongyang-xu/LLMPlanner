from typing import Union, AsyncGenerator
import pandas as pd
from loguru import logger

from probe_data.data import prepare_prompts
from probe_data import REGISTRY
from refusal import refusal_heuristic

from llm_planner.message import Message
from llm_planner.actor.agent import Agent
from llm_planner.actor.system import System

from llm_planner.agents.miniLLM import MiniLLM


class AgenticSecurity(Agent):

    def __init__(self, llm_agent):
        super().__init__()
        self.llm_agent = llm_agent

    async def process(self, sender_id, message: Message):

        if message["response"] is not None:
            rmsg = message["request_message"]
            dataset_name = rmsg["dataset_name"]
            prompt = rmsg["prompt"]
            module_idx = rmsg["module_idx"]

            self.processed_prompts += 1
            print(f"processed {self.processed_prompts}/{self.total_prompts}")
            response_text = message["response"]
            if not refusal_heuristic(response_text):
                self.refusals.append((dataset_name, prompt, 0, response_text))
                self.module_failures[module_idx] += 1

            failure_rate = self.module_failures[module_idx] / max(
                self.processed_prompts, 1)
            self.failure_rates.append(failure_rate)

            if self.processed_prompts >= self.total_prompts:
                logger.info(f"Scanning done")

                df = pd.DataFrame(
                    self.refusals,
                    columns=["module", "prompt", "status_code", "content"])
                df.to_csv("failures.csv", index=False)

        elif message["content"] == "start":
            datasets = REGISTRY
            max_budget = 1_000_000
            tools_inbox = None

            prompt_modules = prepare_prompts(
                dataset_names=[
                    m["dataset_name"] for m in datasets if m["selected"]
                ],
                budget=max_budget,
                tools_inbox=tools_inbox,
            )

            self.errors = []
            self.refusals = []
            self.total_prompts = sum(
                len(m.prompts) for m in prompt_modules if not m.lazy)
            self.processed_prompts = 0

            self.failure_rates = []

            self.module_failures = {}

            for module_idx, module in enumerate(prompt_modules):
                tokens = 0
                self.module_failures[module_idx] = 0
                module_size = 0 if module.lazy else len(module.prompts)
                logger.info(f"Scanning {module.dataset_name} {module_size}")

                for prompt in module.prompts:
                    # progress = 100 * processed_prompts / total_prompts if total_prompts else 0

                    # tokens += len(prompt.split())

                    msg = Message()
                    msg["content"] = prompt
                    msg["module_idx"] = module_idx
                    msg["dataset_name"] = module.dataset_name

                    self.send(self.llm_agent.id, msg)


###################
# run test
###################
llm = MiniLLM(max_token=16)
agentic_security = AgenticSecurity(llm)
msg = Message()
msg['content'] = 'start'
agentic_security.send(agentic_security.id, msg)

System.start()
