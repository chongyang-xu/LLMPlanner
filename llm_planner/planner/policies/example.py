from llm_planner.planner.allocation import AllocationType, Allocation

from llm_planner.planner.op import AbstractCanonizer
from llm_planner.planner.op import AbstractMapper
from llm_planner.planner.op import AbstractGrouper
from llm_planner.planner.op import AbstractRouter
from llm_planner.planner.op import AbstractReducer

from llm_planner.query import Query, PromptedQuery


class PartialQuery(Query):

    def __init__(self, qid: int, query: str, pidx: int, total: int):
        super().__init__(qid, query)
        self.pidx = pidx
        self.total = total


class Canonizer(AbstractCanonizer):

    def __init__(self, p_selector):
        super().__init__(p_selector)

    def __call__(self, alloc: Allocation):
        assert alloc.type == AllocationType.USER_INPUT
        assert len(alloc.q_list) == 1

        alloc.type = AllocationType.INPUT_DECOMPOSING

        # canonizing
        alloc.q_list[0].query = alloc.q_list[0].query.strip()

        alloc.op = self.policy_selector.mapper()
        return alloc


class Mapper(AbstractMapper):

    def __init__(self, p_selector):
        super().__init__(p_selector)

    def __call__(self, alloc: Allocation):
        assert alloc.type == AllocationType.INPUT_DECOMPOSING
        assert len(alloc.q_list) == 1

        q = alloc.q_list[0]
        decompose = q.query.split(",")

        ret = []
        for idx, e in enumerate(decompose):
            pq = PartialQuery(q.qid, e, pidx=idx, total=len(decompose))
            pq.ingress_time = q.ingress_time
            alloc = Allocation(AllocationType.QUERY_GROUPING, [pq],
                               self.policy_selector.grouper())
            ret.append(alloc)
        return ret


class Grouper(AbstractGrouper):

    def __init__(self, p_selector):
        super().__init__(p_selector)
        self.q_list = []
        self.emit_length = 1

    def __call__(self, alloc: Allocation):
        assert alloc.type == AllocationType.QUERY_GROUPING
        assert len(alloc.q_list) == 1

        if alloc.eager:
            alloc.type = AllocationType.QUERY_ROUTING
            alloc.op = self.policy_selector.router()
            return alloc
        else:
            self.q_list.append(alloc.q_list[0])
            if len(self.q_list) >= self.emit_length:
                alloc = Allocation(AllocationType.QUERY_ROUTING, self.q_list,
                                   self.policy_selector.router())
                self.q_list = []
                return alloc
            else:
                alloc = Allocation(AllocationType.NOP, None,
                                   self.policy_selector.nop())
                return alloc


class Router(AbstractRouter):

    def __init__(self, p_selector):
        super().__init__(p_selector)

    def __call__(self, alloc: Allocation):
        assert alloc.type == AllocationType.QUERY_ROUTING
        assert len(alloc.q_list) >= 1

        #-------------------------
        # all route to llm service
        #-------------------------
        alloc.type = AllocationType.QERTY_SERVING
        r_list = self.policy_selector.services[
            'llm_planner.service.VLLMServe'].work_on(alloc.q_list)
        alloc.q_list = r_list

        alloc.type = AllocationType.QUERY_REDUCING
        alloc.op = self.policy_selector.reducer()
        return alloc


class Reducer(AbstractReducer):

    def __init__(self, p_selector):
        super().__init__(p_selector)
        self.dict = {}

    def __call__(self, alloc: Allocation):
        assert alloc.type == AllocationType.QUERY_REDUCING
        assert len(alloc.q_list) >= 1

        ret = []

        for q in alloc.q_list:
            ret_q = None
            if isinstance(q, PartialQuery):
                resp = None
                if q.qid in self.dict:
                    self.dict[q.qid].append(q.response)
                    if len(self.dict[q.qid]) == q.total:
                        resp = ",".join(self.dict[q.qid])
                elif q.total == 1:
                    resp = q.response
                else:
                    self.dict[q.qid] = [q.response]

                if resp is not None:
                    if isinstance(q, PromptedQuery):
                        ret_q = PromptedQuery(q.qid, q.prompt, q.query)
                    else:
                        ret_q = Query(q.qid, q.query)
                    ret_q.response = resp
                    ret_q.ingress_time = q.ingress_time
            else:
                ret_q = q

            if ret_q is not None:
                alloc = Allocation(AllocationType.USER_RESPONSE, [ret_q],
                                   self.policy_selector.nop(),
                                   is_end=True)
                ret.append(alloc)

        if len(ret) == 0:
            alloc = Allocation(AllocationType.NOP,
                               None,
                               self.policy_selector.nop(),
                               is_end=True)
            ret.append(alloc)
        return ret
