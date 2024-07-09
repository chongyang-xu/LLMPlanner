from llm_planner.planner.allocation import AllocationType, Allocation

from llm_planner.planner.op import AbstractCanonizer, AbstractGrouper

from .example import Router as EX_Router
from .example import Reducer as EX_Reducer

Router = EX_Router
Reducer = EX_Reducer


class Canonizer(AbstractCanonizer):

    def __init__(self, p_selector):
        super().__init__(p_selector)

    def __call__(self, alloc: Allocation):
        assert alloc.type == AllocationType.USER_INPUT
        assert len(alloc.q_list) == 1

        alloc.type = AllocationType.QUERY_GROUPING

        # canonizing
        alloc.q_list[0].query = alloc.q_list[0].query.strip()

        alloc.op = self.policy_selector.grouper()
        return alloc


class Grouper(AbstractGrouper):

    def __init__(self, p_selector):
        super().__init__(p_selector)
        self.q_list = []
        self.emit_length = 4
        self.enable = True

    def __call__(self, alloc: Allocation):
        assert alloc.type == AllocationType.QUERY_GROUPING
        assert len(alloc.q_list) == 1

        if not self.enable:
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
