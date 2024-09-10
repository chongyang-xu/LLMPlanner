from typing import List, Dict, Any
import os

from llm_planner.util import timing
from llm_planner.message import Message
from llm_planner.service.service import SingleLLMServe

from llm_planner.logger import Logger

from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.model_str = policy_param_.get(
            'model', "/DS/dsg-ml/nobackup/cxu/weights/gpt2/")
        self.max_token = policy_param_.get('max_token', 16)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def init_service(self):
        if self.init_done:
            return

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_str, device_map="auto").eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.init_done = True

    @timing
    def work_on(self, q_list: List[str]):
        self.init_service()

        batch = [q for q in q_list]
        tokenized_batch = self.tokenizer(
            batch,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=False,
            padding=True,
        ).to('cuda')

        outputs = self.model.generate(**tokenized_batch,
                                      max_new_tokens=self.max_token,
                                      do_sample=False)

        decoded_outputs = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=
            True,  # Removes special tokens like [PAD], <s>, </s>
            clean_up_tokenization_spaces=
            True  # Removes extra spaces in the output)
        )
        # print(f"decoded_outputs={decoded_outputs}")
        new_outputs = [
            out_row[len(in_row):].strip()
            for out_row, in_row in zip(decoded_outputs, batch)
        ]

        # print(f"new_outputs={new_outputs}")
        return new_outputs

        # for idx, o in enumerate(ret):
        #    q_list[idx].response = o

        # return q_list
