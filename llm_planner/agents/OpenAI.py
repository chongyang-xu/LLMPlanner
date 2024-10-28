from llm_planner.actor.agent import Agent

from llm_planner.message import Message
from llm_planner.service.openai_serve_api import OpenAIServe_API


class OpenAI(Agent):

    def __init__(self,
                 max_token=16,
                 return_value=False,
                 with_batching=False,
                 with_caching=False):
        super().__init__(return_value=return_value,
                         with_batching=with_batching,
                         with_caching=with_caching)
        policy_param = {
            "model": "gpt-3.5-turbo",
            "max_token": max_token,
        }
        self.serve = OpenAIServe_API(None, policy_param)

    async def process(self, sender_id, message: Message):
        if message["content"] is not None:
            r = self.serve.work_on([message["content"]])
            if r is not None:
                msg = message.spawn()
                msg['request_message'] = message
                msg['response'] = r[0]
                self.send(sender_id, msg)

    async def process_batch(self, sender_ids, messages):
        senders = []
        msgs = []

        for sid, msg in zip(sender_ids, messages):
            if msg['content'] is not None:
                senders.append(sid)
                msgs.append(msg)

        if len(msgs) > 0:
            wk = [m["content"] for m in msgs]
            rets = self.serve.work_on(wk)
            for i in range(len(rets)):
                msg = msgs[i].spawn()
                msg['request_message'] = msgs[i]
                msg['response'] = rets[i]
                self.send(senders[i], msg)
