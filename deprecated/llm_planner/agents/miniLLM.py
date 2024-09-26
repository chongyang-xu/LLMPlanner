from llm_planner.agents.agent import LLMAgent
from llm_planner.message import Message

from llm_planner.service.hf_serve import HFServe

MINI_LLM_PATH = "/DS/dsg-ml/nobackup/cxu/weights/Qwen2-0.5B/"


class MiniLLM(LLMAgent):

    def __init__(self, max_token=16):
        super().__init__()
        policy_param_ = {"model": MINI_LLM_PATH, "max_token": max_token}
        self.serve = HFServe(None, policy_param_)

    def receive(self, message: Message):
        msg = Message()
        if isinstance(message["prompt"], list):
            msg["ret"] = self.serve.work_on(message["prompt"])
        else:
            msg["ret"] = self.serve.work_on([message["prompt"]])

        return msg
