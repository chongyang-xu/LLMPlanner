# LLMPlanner

## About
LLMPlanner is a planner for serving natural language requests. LLMPlanner takes queries in natual language and return responses to users. It optimizes usage of multiple services to lower down the cost of generating responses for user queries.

## Third party services
LLMPlanner also brings in third party services for its usage.

### Download third_party submodules
```bash
# download submodules directly
git clone --recurse-submodules https://github.com/mpi-dsg/LLMPlanner.git

# or update submodule after downloading
cd LLMPlanner/
git submodule update --init --recursive
```

### cache22
cache22 is considered as a cached serving service in LLMPlanner: if a request has been processed by cach22, the cached results will be used, otherwise cache22 will call model's request method to generate new response.

- Install

```bash
cd third_party/cache22/
pip install -e .
```
