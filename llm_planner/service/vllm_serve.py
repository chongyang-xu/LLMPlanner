from typing import List, Dict, Any

from llm_planner.query import Query
from llm_planner.service.service import SingleLLMServe
from llm_planner.util import timing, test_timing, is_typed_dict, get_gpu_name
from llm_planner.logger import Logger

# Create a custom logger
logger = Logger('VLLMLogger')

from vllm import LLM as VLLM, SamplingParams as VLLMSamplingParams
from vllm.inputs import TokensPrompt

from transformers import AutoTokenizer


class VLLMServe(SingleLLMServe):

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
        self.prefix_caching = policy_param_.get('prefix_caching', False)
        self.tp_size = policy_param_.get('tp_size', 1)
        self.data_type = policy_param_.get('data_type', 'auto')

        gname = get_gpu_name()
        if gname == 'Volta' and self.data_type == 'auto':
            self.data_type = 'half'
            logger.info("Overwriting data_type to half.")
        self.max_token = policy_param_.get('max_token', 16)

    def init_service(self):
        if self.init_done:
            return

        self.model = VLLM(model=self.model_str,
                          trust_remote_code=True,
                          enable_prefix_caching=self.prefix_caching,
                          dtype=self.data_type,
                          tensor_parallel_size=self.tp_size)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.init_done = True

    @timing
    def work_on(self, q_list: List[Query]):
        self.init_service()

        sampling_param = VLLMSamplingParams(temperature=0,
                                            max_tokens=self.max_token)
        batch = [str(q) for q in q_list]
        outputs = self.model.generate(batch, sampling_param)

        for idx, out in enumerate(outputs):
            q_list[idx].response = out.outputs[0].text

        return q_list

    @timing
    def test_query(self, q_list: List[Query], max_tokens):
        sampling_param = VLLMSamplingParams(temperature=0,
                                            max_tokens=max_tokens)

        batch = [str(q) for q in q_list]
        outputs = self.model.generate(batch, sampling_param)

        for idx, out in enumerate(outputs):
            q_list[idx].response = out.outputs[0].text

        return q_list

    @test_timing
    def test_tids(self, t_list: List[TokensPrompt], max_tokens, tag=""):
        sampling_param = VLLMSamplingParams(temperature=0,
                                            max_tokens=max_tokens)

        request_output = self.model.generate(t_list,
                                             sampling_param,
                                             use_tqdm=False)

        rets = []
        for o in request_output:
            rets.append(o.outputs[0].text)
        return rets

    def token(self, q_list: List[Query], seq_token_len=16):
        tokenized_input = []
        for q in q_list:
            out = self.tokenizer(str(q), return_tensors="pt", padding=True)
            input_ids = out["input_ids"].squeeze().tolist()
            tokens_prompt = TokensPrompt(
                prompt_token_ids=input_ids[:seq_token_len])
            tokenized_input.append(tokens_prompt)
        return tokenized_input

    def detoken(self, token_ids):
        ret_text = []
        for output in token_ids:
            if is_typed_dict(output) or isinstance(output, dict):
                if 'input_ids' in output:
                    output = output['input_ids']
                if 'prompt_token_ids' in output:
                    output = output['prompt_token_ids']

            output_text = self.tokenizer.decode(output,
                                                skip_special_tokens=True)
            ret_text.append(output_text)
        return ret_text
