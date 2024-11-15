from ..message import Message
from llm_planner.actor.agent import Agent
from llm_planner.actor.system import System

from llm_planner.agents.miniLLM import MiniLLM

import inspect


class Template:

    def __init__(self, llm=None):
        self.actors = []
        self.registered_agent = {}
        _llm_agent = MiniLLM(max_token=16,
                             return_value=False,
                             with_batching=False,
                             with_caching=False)
        self.registered_agent["mini_llm"] = _llm_agent

        self.sender_receiver = {}
        self.msg_to_reduce = {}  # map_id to factual number for reduce
        self.in_id = 0
        self.out_id = 0

        self.kv_store = {}

    def receiver_of(self, actor):
        return self.sender_receiver[actor.id]

    def input(self):

        class private_input(Agent):

            def __init__(self, template):
                super().__init__()
                self.template = template

            async def process(self, sender_id, message: Message):
                if message["content"] == "start":
                    iterable = message["data"]
                    for data in iterable:
                        msg = message.spawn()
                        msg["content"] = data
                        self.send(self.template.receiver_of(self), msg)

        pi = private_input(self)
        self.actors.append(pi)

        return self  # Return the same object for chaining

    def ask(self, llm_name):
        assert llm_name in self.registered_agent

        class private_ask(Agent):

            def __init__(self, template, llm_name):
                super().__init__()
                self.template = template
                self.llm_name = llm_name

            async def process(self, sender_id, message: Message):
                if message["response"] is not None:
                    message["content"] = message["response"]
                    del message["response"]

                    rmsg = message["request_message"]
                    if rmsg is not None:
                        if rmsg["map_id"] is not None:
                            message["map_id"] = rmsg["map_id"]

                    self.send(self.template.receiver_of(self), message)
                else:
                    receiver_id = self.template.registered_agent[
                        self.llm_name].id
                    self.send(receiver_id, message)

        pask = private_ask(self, llm_name)
        self.actors.append(pask)

        return self

    def map(self, map_function):

        class private_map(Agent):

            def __init__(self, template):
                super().__init__()
                self.template = template

            async def process(self, sender_id, message: Message):
                msgs = map_function(message)
                if isinstance(msgs, list):
                    self.template.msg_to_reduce[message.id] = len(msgs)
                    for m in msgs:
                        m["map_id"] = message.id
                        self.send(self.template.receiver_of(self), m)
                else:
                    msgs["map_id"] = message.id
                    self.template.msg_to_reduce[message.id] = 1
                    self.send(self.template.receiver_of(self), msgs)

        pm = private_map(self)
        self.actors.append(pm)

        return self

    def filter(self, filter_function):

        class private_filter(Agent):

            def __init__(self, template):
                super().__init__()
                self.template = template

            async def process(self, sender_id, message: Message):
                if filter_function(message):
                    self.send(self.template.receiver_of(self), message)
                else:
                    mid = message["map_id"]
                    if mid is not None and mid in self.template.msg_to_reduce:
                        self.template.msg_to_reduce[mid] -= 1

        pf = private_filter(self)
        self.actors.append(pf)

        return self

    def reduce(self, reduce_function):

        class private_reduce(Agent):

            def __init__(self, template):
                super().__init__()
                self.template = template
                self.req_msg = {}

            async def process(self, sender_id, message: Message):
                assert message["map_id"] is not None
                oid = message["map_id"]
                assert oid in self.template.msg_to_reduce.keys()
                reduce_count = self.template.msg_to_reduce[oid]

                if oid not in self.req_msg:
                    self.req_msg[oid] = [message]
                else:
                    self.req_msg[oid].append(message)

                if oid in self.req_msg and len(
                        self.req_msg[oid]) == reduce_count:
                    result_msg = reduce_function(self.req_msg[oid])
                    self.send(self.template.receiver_of(self), result_msg)
                    del self.req_msg[oid]

        pr = private_reduce(self)
        self.actors.append(pr)

        return self

    def repeat(self, times, sub_block):

        class private_repeat(Agent):

            def __init__(self, template):
                super().__init__()
                self.template = template
                self.times = times
                self.sub_block = sub_block

            def get_times(self, message: Message):
                if isinstance(self.times, int):
                    return self.times
                else:
                    assert callable(self.times)
                    return self.times(message)

            async def process(self, sender_id, message: Message):
                repeat_n = self.get_times(message)
                if sender_id == self.sub_block.out_id:
                    message.tid += 1
                    if message.tid < repeat_n:
                        self.send(self.sub_block.in_id, message)
                    else:
                        message.tid = 0
                        self.send(self.template.receiver_of(self), message)
                else:
                    if repeat_n > 0:
                        self.send(self.sub_block.in_id, message)

        prpt = private_repeat(self)
        sub_block.sender_receiver[sub_block.out_id] = prpt.id
        self.actors.append(prpt)
        return self

    def branch(self, condition, true_block, false_block):

        class private_branch(Agent):

            def __init__(self, template):
                super().__init__()
                self.template = template
                self.condition = condition
                self.true_block = true_block
                self.false_block = false_block

            async def process(self, sender_id, message: Message):

                if sender_id in [
                        self.true_block.out_id, self.false_block.out_id
                ]:
                    self.send(self.template.receiver_of(self), message)
                else:
                    if condition(message):
                        self.send(self.true_block.in_id, message)
                    else:
                        self.send(self.false_block.in_id, message)

        pb = private_branch(self)
        true_block.sender_receiver[true_block.out_id] = pb.id
        false_block.sender_receiver[false_block.out_id] = pb.id
        self.actors.append(pb)
        return self

    def print(self):

        class private_print(Agent):

            def __init__(self, template):
                super().__init__()
                self.template = template

            async def process(self, sender_id, message: Message):
                print(message["content"])
                print('-' * 20)

        pp = private_print(self)
        self.actors.append(pp)

        return self

    def done(self):
        self.in_id = self.actors[0].id
        self.out_id = self.actors[-1].id
        for i in range(len(self.actors) - 1):
            self.sender_receiver[self.actors[i].id] = self.actors[i + 1].id
        return self

    def start(self, iterable):
        msg = Message()
        msg['content'] = "start"
        msg['data'] = iterable
        self.actors[0].send(self.actors[0].id, msg)
        System.start()

    def get(self, key):
        if key in self.kv_store:
            return self.kv_store[key]
        else:
            return None

    def put(self, key, value):
        self.kv_store[key] = value


def find_instance_of(typename):
    frame = inspect.currentframe()
    while frame:
        local_self = frame.f_locals.get("self")
        if isinstance(local_self, typename):
            return local_self
        frame = frame.f_back
    assert False, f"can't find container of type {typename}"


def kv_get(key):
    template = find_instance_of(Template)
    return template.get(key)


def kv_put(key, value):
    template = find_instance_of(Template)
    template.put(key, value)
