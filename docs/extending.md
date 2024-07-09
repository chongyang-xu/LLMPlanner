# LLMPlanner

## About
LLMPlanner is a planner for serving natural language requests. LLMPlanner takes queries in natual language and return responses to users. It optimizes usage of multiple services to lower down the cost of generating responses for user queries.

## Extending LLMPlanner

### Adding a new optimization

1. New optimizations can be added as a `policy` into folder `llm_planner/planner/policies/`. `batching.py` is an example of testing idea of Batching by implementing it in a `Grouper` operator.

2. `llm_planner/planner/policy.py` need to be extended to handle the new policy from parameter `select`

### Adding a new service

A new service can be added to `llm_planner/service/`, and can then be registered to `SERVICE_LIST` in `llm_planner/planner/policy.py`.
