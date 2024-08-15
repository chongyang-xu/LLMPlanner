import os
from typing import Dict, Any

from llm_planner.planner.policy import PolicySelector
from llm_planner.planner.orchestrator import Orchestrator
from llm_planner.planner.queue import Ingress

from llm_planner.logger import Logger

from llm_planner.query import Instruct, InstructQuery, Stop

from llm_planner.data.coqa_repurpose import load_coqa_query_to_ingress

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a custom logger
logger = Logger('TestLoRAFinetune')

ps = PolicySelector(select="instruct")
orch = Orchestrator(ps)

para: Dict[str, Any] = {
    "model": "/DS/dsg-ml/nobackup/cxu/weights/Meta-Llama-3-8B/",
    "lora_path": [
        "/tmp/lora_1/lora/",
        "/tmp/lora_2/lora/",
    ],
}

iq = InstructQuery(qid=0,
                   instruct_=Instruct.INFERENCE_LORA,
                   instruct_param_=para)
stop = Stop(qid=2)
ing = Ingress()

ing.enq(iq)
ing.enq(stop)

orch.inject_ingress(ing)

orch.run()

eg = orch.inject_egress()
for q in eg.queue:
    logger.info(f"qid:{q.qid:08d}, {q.response}")
