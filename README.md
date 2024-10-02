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
class Printer(Agent):

    async def process(self, sender_id, message):
        content = message['content']
        print(content)


printer = Printer()
system = System()

msg = Message()
msg['content'] = 'start'
system.send(None, printer.id, msg)

system.start()
```

### more examples:
```bash
ls tests/ apps/
```
