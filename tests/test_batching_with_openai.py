from typing import Dict, Any

from llm_planner.planner.policy import PolicySelector
from llm_planner.planner.orchestrator import Orchestrator

from llm_planner.logger import Logger

from llm_planner.data.coqa_repurpose import load_coqa_query_to_ingress

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a custom logger
logger = Logger('TestBatching')

policy_para: Dict[str, Any] = {
    "model": 'gpt-3.5-turbo',
    "use_cache22": True,
    "cache22" : {},
}

ps = PolicySelector(select="batching", policy_param_=policy_para)
orch = Orchestrator(ps)

ing = load_coqa_query_to_ingress(n_query=4, shuffle=False)
orch.inject_ingress(ing)

orch.run()

eg = orch.inject_egress()
for q in eg.queue:
    logger.info(
        f"qid:{q.qid:08d}, time: {q.egress_time - q.ingress_time:.3f} s, {q.response.strip()}"
    )
