from typing import Dict, Any

from .policies import example
from .policies import batching
from .policies import instruct

from .op import NOP

from llm_planner.service.hf_serve import HFServe
from llm_planner.service.openai_serve_api import OpenAIServe_API
from llm_planner.service.anthropic_serve_api import AnthropicServe_API
from llm_planner.service.hf_serve import HFServe
from llm_planner.service.hf_train import HFTrain
from llm_planner.service.hf_finetune import HFFullParameterFinetune
from llm_planner.service.hf_finetune import HFLoRAFinetune
from llm_planner.service.hf_serve import HFLoRAServe

from llm_planner.service.cache22 import CachedServing22

SERVICE_LIST = {
    "llm_planner.service.HFServe": HFServe,
    "llm_planner.service.OpenAIServe_API": OpenAIServe_API,
    "llm_planner.service.AnthropicServe_API": AnthropicServe_API,
    "llm_planner.service.Cache22": CachedServing22,
    "llm_planner.service.HFTrain": HFTrain,
    "llm_planner.service.HFFullParameterFinetune": HFFullParameterFinetune,
    "llm_planner.service.HFLoRAFinetune": HFLoRAFinetune,
    "llm_planner.service.HFLoRAServe": HFLoRAServe,
}


class PolicySelector:

    def __init__(self, select="example", policy_param_: Dict[str, Any] = {}):

        self.use_cache22 = policy_param_.get('use_cache22', False)

        self.model = policy_param_.get('model', None)

        self.services = {}
        for k, Class in SERVICE_LIST.items():
            self.services[k] = Class(self, policy_param_)

        if select == "example":
            self.canonizer_ = example.Canonizer(self)
            self.mapper_ = example.Mapper(self)
            self.grouper_ = example.Grouper(self)
            self.grouper_.enable = policy_param_.get('use_grouper', True)
            self.router_ = example.Router(self)
            self.reducer_ = example.Reducer(self)
            self.nop_ = NOP(self)

        elif select == "batching":
            self.canonizer_ = batching.Canonizer(self)
            self.mapper_ = NOP(self)
            self.grouper_ = batching.Grouper(self)
            self.grouper_.enable = policy_param_.get('use_grouper', True)
            self.router_ = batching.Router(self)
            self.reducer_ = batching.Reducer(self)
            self.nop_ = NOP(self)

        elif select == "instruct":
            self.canonizer_ = instruct.Canonizer(self)
            self.mapper_ = NOP(self)
            self.grouper_ = NOP(self)
            self.router_ = instruct.Router(self)
            self.reducer_ = instruct.Reducer(self)
            self.nop_ = NOP(self)
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
