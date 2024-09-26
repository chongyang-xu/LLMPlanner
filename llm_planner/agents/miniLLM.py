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
        if message['content'] is not None:
            r = self.serve.work_on([message["content"]])
            if r is not None:
                msg = Message()
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
                msg = Message()
                msg['request_message'] = msgs[i]
                msg['response'] = rets[i]
                self.send(senders[i], msg)



"""
    async def process(self, message: Message) -> None:

        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(executor, self.blocking_serve,
                                          message)
            self.futures[message.id] = future

    def blocking_serve(self, message: Message):
        start = time.time()

        msg = Message()
        if isinstance(message["prompt"], list):
            msg["ret"] = self.serve.work_on(message["prompt"])
        else:
            msg["ret"] = self.serve.work_on([message["prompt"]])

        end = time.time()

        # print(f"blocking_serve: {end-start:.2f} sec")
        # print(f"blocking_serve: {msg}")
        return msg
"""