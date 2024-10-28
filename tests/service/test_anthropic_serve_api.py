
from llm_planner.service.anthropic_serve_api import AnthropicServe_API


policy_param = {
    "model": "claude-3-5-sonnet-20241022",
    "max_token": 16,
}
serve = AnthropicServe_API(None, policy_param)

content = [
{
  "role": "user",
   "content": "Which LLMs are best?",
}]
r = serve.work_on([content])

print(r)
