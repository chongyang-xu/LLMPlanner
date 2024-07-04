from typing import List

from llm_planner.query import Query
from llm_planner.service.service import Service
from llm_planner.util import timing, test_timing, is_typed_dict
from llm_planner.logger import Logger

# Create a custom logger
logger = Logger('SingleLLMLogger')

from vllm import LLM as VLLM, SamplingParams as VLLMSamplingParams
from vllm.inputs import TokensPrompt

from transformers import AutoTokenizer


class SingleLLM(Service):
    # from llm_planner.planner.orchestrator import Orchestrator
    def __init__(self,
                 p_selector,
                 model_path=None,
                 dtype='auto',
                 tp=1,
                 prefix_caching=False):
        logger.info(f"model_path={model_path}, dtype={dtype}")

        super().__init__()
        self.p_selector = p_selector

        if model_path is None:
            model_path = "/DS/dsg-ml/nobackup/cxu/weights/Qwen2-0.5B/"
        self.model = VLLM(model=model_path,
                          trust_remote_code=True,
                          enable_prefix_caching=prefix_caching,
                          dtype=dtype,
                          tensor_parallel_size=tp)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @timing
    def work_on(self, q_list: List[Query], max_tokens=16):
        sampling_param = VLLMSamplingParams(temperature=0,
                                            max_tokens=max_tokens)
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
