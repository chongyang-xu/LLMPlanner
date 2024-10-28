
from llm_planner.service.openai_serve_api import OpenAIServe_API


policy_param = {
    # "model": "gpt-3.5-turbo",
    "model": "gpt-4o-2024-08-06",
    "max_token": 16,
}
serve = OpenAIServe_API(None, policy_param)

content = [
{
  "role": "user",
   "content": "Which LLMs are best?",
}]
r = serve.work_on([content])

print(r)
