from llm_planner.agents.agent import LLMAgent

from llm_planner.message import Message
from llm_planner.service.openai_serve_api import OpenAIServe_API


class OpenAI(LLMAgent):

    def __init__(self, max_token=16):
        super().__init__()
        policy_param = {
            "model": "gpt-3.5-turbo",
            "max_token": max_token,
        }
        self.impl = OpenAIServe_API(None, policy_param)

    def receive(self, message):
        msg = Message()
        if isinstance(message["prompt"], list):
            msg["ret"] = self.impl.work_on(message["prompt"])
        else:
            msg["ret"] = self.impl.work_on([message["prompt"]])

        return msg
