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

    def formatting_content(self, input_var):
        if isinstance(input_var, list):
            if isinstance(input_var[0], dict):
                return self.tokenizer.apply_chat_template(
                    input_var, tokenize=False, add_generation_prompt=True)
            else:
                assert False, "a single request with multiple text chunk is unexpected"
        assert isinstance(input_var, str)
        return input_var

    @timing
    def work_on(self, q_list: List[str]):
        self.init_service()

        batch = [self.formatting_content(q) for q in q_list]
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

        new_outputs = [
            out_row[len(in_row):]
            for out_row, in_row in zip(outputs, tokenized_batch["input_ids"])
        ]

        decoded_outputs = self.tokenizer.batch_decode(
            new_outputs,
            skip_special_tokens=
            True,  # Removes special tokens like [PAD], <s>, </s>
            clean_up_tokenization_spaces=
            True  # Removes extra spaces in the output),
        )
        return decoded_outputs
