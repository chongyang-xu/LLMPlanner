# LLMPlanner

## About
LLMPlanner is a planner for serving natural language requests. LLMPlanner takes queries in natual language and return responses to users. It optimizes usage of multiple services to lower down the cost of generating responses for user queries. Find overview [here](./docs/overview.png).

## Quick start

### Requirements:
- Python >= 3.9

```bash
# using virtual environment is recommended
$ conda create -n env_name python=3.9
$ conda activate env_name
$ conda install pip

# install required packages
$ ./scripts/prep_cu118.sh # for packages with cuda-11.8
```

### Install
```bash
git clone https://github.com/mpi-dsg/LLMPlanner.git
cd LLMPlanner/
./scripts/wheel.sh
```

#### Reinstall llm_planner
Before using latest code inside directory llm_planner/, you need to reinstall llm_planner by
```
./scripts/wheel.sh
```
To intall third party modules, refer [here](./docs/third_party.md).


### Run an example
After llm_planner is installed, an 'app' that uses llm_planner can be started.

This is an example sketch of an app:
```python

policy_para: Dict[str, Any] = {
    "model": '/DS/dsg-ml/nobackup/cxu/weights/Meta-Llama-3-8B/',
}

# select a policy/optimization to use
ps = PolicySelector(select="batching", policy_param_=policy_para)

# create a orchestrator with the policy
orch = Orchestrator(ps)

# prepare a Message
q = Message(qid, Message)

# submit the Message to orchestrator
orch.inject_ingress([q])

# run the planner
orch.run()

# extract the response
eg = orch.inject_egress()
```

### an example:
```bash
cd test/
python test_policy_batching.py
```
