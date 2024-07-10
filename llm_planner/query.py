from enum import Enum
from typing import Dict, Any


class Query:

    def __init__(self, qid: int, query: str):
        self.qid = qid
        self.query: str = query
        self.response = None
        self.ingress_time = 0.0
        self.egress_time = 0.0

    def __repr__(self):
        return f"{self.query}"

    def from_json(self, json_str: str):
        pass

    def to_json(self):
        pass


class PromptedQuery(Query):

    def __init__(self, qid: int, prompt: str, query: str):
        super().__init__(qid, query)
        self.prompt = prompt

    def __repr__(self):
        return f"{self.prompt}\n{self.query}"


class Instruct(Enum):
    FINETUNE_FULL = "FinetuneFullParameter"
    FINETUNE_LORA = "FinetuneLoRA"
    TRAIN = "Train"
    INFERENCE_LORA = "InferenceLoRA"


class InstructQuery(Query):

    def __init__(self,
                 qid: int,
                 instruct_: Instruct,
                 instruct_param_: Dict[str, Any],
                 query_="<InstructedQuery>"):
        super().__init__(qid, query_)

        self.instruct: Instruct = instruct_
        self.instruct_param: Dict[str, Any] = instruct_param_
        self.param_check(self.instruct_param)
        self.print_params()

    def print_params(self):
        print('-' * 50)
        print(f"qid: {self.qid}, Instruct: {self.instruct}")
        print('-' * 50)

    def __repr__(self):
        return f"{self.query}"

    def param_spec(self, instruct_: Instruct):
        pass

    def param_check(self, param: Dict[str, Any]):
        if self.instruct in (Instruct.TRAIN, Instruct.FINETUNE_FULL):
            assert "model_path" in param
        elif self.instruct == Instruct.FINETUNE_LORA:
            assert "model_path" in param
        elif self.instruct == Instruct.INFERENCE_LORA:
            assert "model_path" in param


class Stop(Query):

    def __init__(self, qid: int = -1):
        super().__init__(qid, "$stop$")
