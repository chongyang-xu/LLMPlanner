from typing import List, Dict, Any

from llm_planner.util import timing
from llm_planner.query import Query
from llm_planner.service.service import SingleLLMServe

from llm_planner.logger import Logger

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import openai
import os 

from dotenv import load_dotenv
# Create a custom logger
logger = Logger('HuggingFaceLogger')


class OpenAIServe_API(SingleLLMServe):

    def __init__(self, p_selector, policy_param_: Dict[str, Any]):

        super().__init__()
        load_dotenv()

        # Now you can access the variables
        self.api_key = os.getenv('OPENAI_API_KEY')

        self.p_selector = p_selector
        self.init_done = False
        ##########################
        #
        # setting parameters
        #
        ##########################
        self.model_path = policy_param_.get(
            'model_api', "gpt-3.5-turbo")
        self.max_token = policy_param_.get('max_token', 16)

    def init_service(self):

        if self.init_done:
            return

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_path, device_map="auto").eval()

        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self.tokenizer.padding_side = "left"
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        self.init_done = True

    @timing
    def work_on(self, q_list: List[Query]):
        self.init_service()
        openai.api_key = self.api_key
        
        batch = [str(q) for q in q_list]

        response = openai.Completion.create(
            engine="gpt-3.5-turbo", 
            prompt=batch,
            max_tokens=self.max_token,  
            stop=None,
            temperature=0.7
        )

        generated_text = response['choices'][0]['text'].strip()
        responses.append(generated_text)

        for idx, o in enumerate(responses):
            q_list[idx].response = o

        return q_list

        return q_list


