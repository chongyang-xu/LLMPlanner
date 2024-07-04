import time

from .allocation import Allocation, AllocationType
from .queue import Queue, Ingress, Egress
from .policy import PolicySelector

from llm_planner.util import timing


class Orchestrator:

    def __init__(self, p_selector: PolicySelector):
        self.ingress = Ingress()
        self.egress = Egress()
        self.policy_selector = p_selector

    def inject_ingress(self, ing: Ingress):
        self.ingress = ing

    def inject_egress(self):
        return self.egress

    @timing
    def run(self):
        while True:
            if self.ingress.empty():
                continue

            user_input = self.ingress.deq()
            user_input.ingress_time = time.time()
            alloc = Allocation(AllocationType.USER_INPUT, [user_input],
                               self.policy_selector.canonizer())

            alloc_q = Queue()
            alloc_q.enq(alloc)

            if user_input.query == "$stop$":
                break

            while not alloc_q.empty():
                alloc = alloc_q.deq()
                ret = alloc.op(alloc)

                if ret is None:
                    continue

                ret_list = [ret] if not isinstance(ret, list) else ret

                for e in ret_list:
                    if e.type == AllocationType.NOP:
                        continue
                    if e.is_end:
                        assert len(e.q_list) == 1
                        e.q_list[0].egress_time = time.time()
                        self.egress.enq(e.q_list[0])
                    else:
                        alloc_q.enq(e)
