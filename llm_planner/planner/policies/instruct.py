from llm_planner.planner.allocation import AllocationType, Allocation

from llm_planner.planner.op import AbstractCanonizer, AbstractRouter, AbstractReducer

from llm_planner.query import InstructQuery, Instruct


class Canonizer(AbstractCanonizer):

    def __init__(self, p_selector):
        super().__init__(p_selector)

    def __call__(self, alloc: Allocation):
        assert alloc.type == AllocationType.USER_INPUT
        assert len(alloc.q_list) == 1

        alloc.type = AllocationType.QUERY_ROUTING

        # canonizing
        alloc.q_list[0].query = alloc.q_list[0].query.strip()

        alloc.op = self.policy_selector.router()
        return alloc


class Router(AbstractRouter):

    def __init__(self, p_selector):
        super().__init__(p_selector)

    def __call__(self, alloc: Allocation):
        assert alloc.type == AllocationType.QUERY_ROUTING
        assert len(alloc.q_list) == 1

        alloc.type = AllocationType.QERTY_SERVING

        q = alloc.q_list[0]
        if not isinstance(q, InstructQuery):
            assert False, "Only InstructQuery is handled"

        if q.instruct == Instruct.TRAIN:
            r_list = self.policy_selector.services[
                "llm_planner.service.HFTrain"].work_on(alloc.q_list)
        elif q.instruct == Instruct.FINETUNE_FULL:
            r_list = self.policy_selector.services[
                "llm_planner.service.HFFullParameterFinetune"].work_on(
                    alloc.q_list)
        elif q.instruct == Instruct.FINETUNE_LORA:
            r_list = self.policy_selector.services[
                "llm_planner.service.HFLoRAFinetune"].work_on(alloc.q_list)
        elif q.instruct == Instruct.INFERENCE_LORA:
            r_list = self.policy_selector.services[
                "llm_planner.service.HFLoRAServe"].work_on(alloc.q_list)
        else:
            assert False, "Not supported Instruct"

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
        assert len(alloc.q_list) == 1

        alloc.type == AllocationType.USER_RESPONSE
        alloc.op = self.policy_selector.nop()
        alloc.is_end = True
        return alloc
