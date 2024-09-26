from llm_planner.actor.agent import Agent

from llm_planner.message import Message

from llm_planner.service.hf_serve import HFServe

MINI_LLM_PATH = "/DS/dsg-ml/nobackup/cxu/weights/Qwen2-0.5B/"


class MiniLLM(Agent):

    def __init__(self,
                 max_token=16,
                 return_value=False,
                 with_batching=True,
                 with_caching=False):
        super().__init__(return_value=return_value,
                         with_batching=with_batching,
                         with_caching=with_caching)
        policy_param_ = {"model": MINI_LLM_PATH, "max_token": max_token}
        self.serve = HFServe(None, policy_param_)

    async def process(self, sender_id, message: Message):
        ret = self.serve.work_on([message["prompt"]])

        msg = Message()
        msg['request_msg_id'] = message.id
        msg['response'] = ret[0]

        self.send(sender_id, msg)

    async def process_batch(self, sender_ids, messages):
        ret = self.serve.work_on([msg["prompt"] for msg in messages])

        for i in range(len(ret)):
            msg = Message()
            msg['request_msg_id'] = messages[i].id
            msg['response'] = ret[i]
            self.send(sender_ids[i], msg)
