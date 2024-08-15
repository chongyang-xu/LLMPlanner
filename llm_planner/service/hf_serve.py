from typing import List, Dict, Any

from llm_planner.util import timing
from llm_planner.query import Query
from llm_planner.service.service import SingleLLMServe

from llm_planner.logger import Logger

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Create a custom logger
logger = Logger('HuggingFaceLogger')


class HFServe(SingleLLMServe):

    def __init__(self, p_selector, policy_param_: Dict[str, Any]):

        super().__init__()
        self.p_selector = p_selector
        self.init_done = False
        ##########################
        #
        # setting parameters
        #
        ##########################
        self.model = policy_param_.get('model',
                                       "/DS/dsg-ml/nobackup/cxu/weights/gpt2/")
        self.max_token = policy_param_.get('max_token', 16)

    def init_service(self):
        if self.init_done:
            return

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map="auto").eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.init_done = True

    @timing
    def work_on(self, q_list: List[Query]):
        self.init_service()

        batch = [str(q) for q in q_list]

        tokenized_batch = self.tokenizer(
            batch,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=False,
            padding=True,
        )

        outputs = self.model.generate(**tokenized_batch,
                                      max_new_tokens=self.max_token,
                                      do_sample=False)
        ret = self.tokenizer.batch_decode(outputs)

        for idx, o in enumerate(ret):
            q_list[idx].response = o

        return q_list


class HFLoRAServe(SingleLLMServe):

    def __init__(self, p_selector, policy_param_: Dict[str, Any]):

        super().__init__()
        self.p_selector = p_selector
        self.init_done = False
        ##########################
        #
        # setting parameters
        #
        ##########################
        self.model = policy_param_.get('model',
                                       "/DS/dsg-ml/nobackup/cxu/weights/gpt2/")
        self.max_token = policy_param_.get('max_token', 16)

    def init_service(self, inst_param: Dict[str, Any]):
        if self.init_done:
            return

        assert "lora_path" in inst_param
        self.model = inst_param.get('model', self.model)
        self.lora_path = inst_param.get('lora_path', None)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map="auto").eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = PeftModel.from_pretrained(self.base_model,
                                               self.lora_path[0],
                                               adapter_name="adapter_0")

        adapters = []
        weights = []
        for idx, lora in enumerate(self.lora_path):
            if idx == 0:
                continue
            _ = self.model.load_adapter(lora, adapter_name=f"adapter_{idx}")
            adapters.append(f"adapter_{idx}")
            weights.append(1.0)

        self.model.add_weighted_adapter(adapters,
                                        weights,
                                        "merge",
                                        combination_type="ties",
                                        density=0.2)

        self.model.set_adapter("merge")

        self.init_done = True

    @timing
    def work_on(self, q_list: List[Query]):
        self.init_service()

        batch = [str(q) for q in q_list]

        tokenized_batch = self.tokenizer(
            batch,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=False,
            padding=True,
        )

        outputs = self.model.generate(**tokenized_batch,
                                      max_new_tokens=self.max_token,
                                      do_sample=False)
        ret = self.tokenizer.batch_decode(outputs)

        for idx, o in enumerate(ret):
            q_list[idx].response = o

        return q_list
