import os
from typing import Dict, Any

from llm_planner.planner.policy import PolicySelector
from llm_planner.planner.orchestrator import Orchestrator
from llm_planner.planner.queue import Ingress

from llm_planner.logger import Logger

from llm_planner.query import Instruct, InstructQuery, Stop
from llm_planner.data.dolly_15k_oai import Dolly15kOAI

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a custom logger
logger = Logger('TestLoRAFinetune')

ps = PolicySelector(select="instruct")
orch = Orchestrator(ps)

para: Dict[str, Any] = {
    "model_path": "/DS/dsg-ml/nobackup/cxu/weights/Meta-Llama-3-8B/",
    "lora_output_path": "/tmp/lora_3/",
    "max_steps": 3,
    "dataset": Dolly15kOAI()
}

iq = InstructQuery(qid=0,
                   instruct_=Instruct.FINETUNE_LORA,
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

# accelerate launch test_lora_finetune.py
