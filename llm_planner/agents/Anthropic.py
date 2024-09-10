from llm_planner.agents.agent import LLMAgent

from llm_planner.message import Message
from llm_planner.service.anthropic_serve_api import AnthropicServe_API


class Anthropic(LLMAgent):

    def __init__(self):
        super().__init__()
        policy_param = {
            "model": "claude-3-5-sonnet-20240620",
            "max_token": 16,
        }
        self.impl = AnthropicServe_API(None, policy_param)

    def receive(self, message):
        msg = Message()
        if isinstance(message["prompt"], list):
            msg["ret"] = self.impl.work_on(message["prompt"])
        else:
            msg["ret"] = self.impl.work_on([message["prompt"]])

        return msg
