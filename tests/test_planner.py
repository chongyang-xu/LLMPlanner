from llm_planner.planner.policy import PolicySelector
from llm_planner.planner.orchestrator import Orchestrator
from llm_planner.query import Query

# select a policy/optimization to use

policy_para: Dict[str, Any] = {
    "model_path": '/DS/dsg-ml/nobackup/cxu/weights/Meta-Llama-3-8B/',
    "model_api":'gpt-3.5-turbo'
}

ps = PolicySelector(select="batching",policy_param_=policy_para)

# create a orchestrator with the policy
orch = Orchestrator(ps)

# prepare a query
q = Query(qid, query)

# submit the query to orchestrator
orch.inject_ingress([q])

# run the planner
orch.run()

# extract the response
eg = orch.inject_egress()