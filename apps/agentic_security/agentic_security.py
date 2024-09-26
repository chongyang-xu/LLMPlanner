from typing import Union, AsyncGenerator
import pandas as pd
from loguru import logger

from probe_data.data import prepare_prompts
from probe_data import REGISTRY
from refusal import refusal_heuristic

from llm_planner.message import Message
from llm_planner.agents.miniLLM import MiniLLM

actor_ref = MiniLLM.start(max_token=16)

datasets = REGISTRY
max_budget = 1_000_000
tools_inbox = None

prompt_modules = prepare_prompts(
    dataset_names=[m["dataset_name"] for m in datasets if m["selected"]],
    budget=max_budget,
    tools_inbox=tools_inbox,
)

errors = []
refusals = []
total_prompts = sum(len(m.prompts) for m in prompt_modules if not m.lazy)
processed_prompts = 0

failure_rates = []

for module in prompt_modules:
    tokens = 0
    module_failures = 0
    module_size = 0 if module.lazy else len(module.prompts)
    logger.info(f"Scanning {module.dataset_name} {module_size}")

    for prompt in module.prompts:
        processed_prompts += 1
        progress = 100 * processed_prompts / total_prompts if total_prompts else 0

        tokens += len(prompt.split())

        try:
            msg = Message(prompt=prompt)
            answer = actor_ref.ask(msg)
            response_text = answer["ret"][0]

            tokens += len(response_text.split())

            if not refusal_heuristic(response_text):
                refusals.append((module.dataset_name, prompt, 0, response_text))
                module_failures += 1

        except Exception as e:
            logger.error(f"Request error: {e}")
            errors.append((module.dataset_name, prompt, str(e)))
            module_failures += 1
            continue

        failure_rate = module_failures / max(processed_prompts, 1)
        failure_rates.append(failure_rate)

logger.info(f"Scanning done")

df = pd.DataFrame(errors + refusals,
                  columns=["module", "prompt", "status_code", "content"])
df.to_csv("failures.csv", index=False)
