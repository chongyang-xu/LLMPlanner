from llm_planner.planner.policy import PolicySelector
from llm_planner.planner.orchestrator import Orchestrator

from llm_planner.logger import Logger

from dataset import load_coqa_query_to_ingress

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a custom logger
logger = Logger('TestPlanner')

model_path = '/DS/dsg-ml/nobackup/cxu/weights/Meta-Llama-3-8B-Instruct/'
ps = PolicySelector(select="batching",
                    model_path=model_path,
                    use_grouper=True,
                    prefix_caching=True)
orch = Orchestrator(ps)

ing = load_coqa_query_to_ingress(n_query=4, shuffle=False)
orch.inject_ingress(ing)

orch.run()

eg = orch.inject_egress()
for q in eg.queue:
    logger.info(
        f"qid:{q.qid:08d}, time: {q.egress_time - q.ingress_time:.3f} s, {q.response.strip()}"
    )
