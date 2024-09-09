from llm_planner.agents.agent import LLMAgent
from llm_planner.message import Message
from llm_planner.service.hf_serve import HFServe

LLAMA3_8B_PATH = "/DS/dsg-ml/nobackup/cxu/weights/Meta-Llama-3-8B-Instruct/"


class Llama3_8B(LLMAgent):

    def __init__(self):
        super().__init__()
        policy_param_ = {"model": LLAMA3_8B_PATH, "max_token": 1024}
        self.serve = HFServe(None, policy_param_)

    def receive(self, message: Message):
        msg = Message()
        if isinstance(msg, list):
            msg["ret"] = self.serve.work_on(message["prompt"])
        else:
            msg["ret"] = self.serve.work_on([message["prompt"]])

        return msg
