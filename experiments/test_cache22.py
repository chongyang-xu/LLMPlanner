from typing import Dict, Any

from llm_planner.planner.policy import PolicySelector
from llm_planner.planner.orchestrator import Orchestrator

from llm_planner.logger import Logger

from llm_planner.data.coqa_repurpose import load_coqa_query_to_ingress
from llm_planner.data.similar_questions import SimilarQuestions

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a custom logger
logger = Logger('TestCache22')

policy_para: Dict[str, Any] = {
    "model_path": '/DS/dsg-ml/nobackup/cxu/weights/Meta-Llama-3-8B/',
    "use_cache22": False,
    "cache22": {
        "cache_dir": "/tmp/cache22/"
    }
}

ps = PolicySelector(select="batching", policy_param_=policy_para)
orch = Orchestrator(ps)

TEST_DS = 'sim_q'  #'coqa', 'sim_q'
N_QUERY = 128
SHUFFLE = True

if TEST_DS == 'coqa':
    ing = load_coqa_query_to_ingress(n_query=N_QUERY, shuffle=SHUFFLE)
    orch.inject_ingress(ing)

    orch.run()

    # eg = orch.inject_egress()
    # for q in eg.queue:
    #     logger.info(
    #         f"qid:{q.qid:08d}, time: {q.egress_time - q.ingress_time:.3f} s, {q.response.strip()}"
    #     )

if TEST_DS == 'sim_q':
    ds = SimilarQuestions()
    ing = ds.to_ingress(n_query=N_QUERY, shuffle=SHUFFLE)
    orch.inject_ingress(ing)

    orch.run()

    # eg = orch.inject_egress()
    # for q in eg.queue:
    #     logger.info(
    #         f"qid:{q.qid:08d}, time: {q.egress_time - q.ingress_time:.3f} s, {q.response.strip()}"
    #     )
