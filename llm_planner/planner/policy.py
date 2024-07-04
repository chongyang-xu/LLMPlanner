from .policies import example
from .policies import batching

from .op import NOP

from llm_planner.service.llm_service import SingleLLM


class PolicySelector:

    def __init__(self,
                 select="example",
                 use_grouper=True,
                 model_path=None,
                 tp=1,
                 prefix_caching=False):
        if select == "example":
            self.canonizer_ = example.Canonizer(self)
            self.mapper_ = example.Mapper(self)
            self.grouper_ = example.Grouper(self)
            self.use_grouper_ = True
            self.router_ = example.Router(self)
            self.reducer_ = example.Reducer(self)
            self.nop_ = NOP(self)

            self.llm_service_ = SingleLLM(self,
                                          model_path=model_path,
                                          dtype='half',
                                          tp=tp,
                                          prefix_caching=prefix_caching)
        elif select == "batching":
            self.canonizer_ = batching.Canonizer(self)
            self.mapper_ = NOP(self)
            self.grouper_ = batching.Grouper(self)
            self.use_grouper_ = use_grouper
            self.router_ = batching.Router(self)
            self.reducer_ = batching.Reducer(self)
            self.nop_ = NOP(self)

            self.llm_service_ = SingleLLM(self,
                                          model_path=model_path,
                                          dtype='half',
                                          tp=tp,
                                          prefix_caching=prefix_caching)
        else:
            assert False

    def canonizer(self):
        return self.canonizer_

    def mapper(self):
        return self.mapper_

    def grouper(self):
        return self.grouper_

    def use_grouper(self):
        return self.use_grouper_

    def router(self):
        return self.router_

    def reducer(self):
        return self.reducer_

    def nop(self):
        return self.nop_

    def llm_service(self):
        return self.llm_service_
