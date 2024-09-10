from typing import List, Dict, Any

from llm_planner.util import timing
from llm_planner.message import Message
from llm_planner.service.service import SingleLLMServe

from llm_planner.logger import Logger

from openai import OpenAI
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
        self.model_str = policy_param_.get('model', "gpt-3.5-turbo")
        self.max_token = policy_param_.get('max_token', 16)

    def init_service(self):

        if self.init_done:
            return

        self.client = OpenAI()

        self.init_done = True

    @timing
    def work_on(self, q_list: List[str]):
        self.init_service()
        responses = []
        for q in q_list:
            b = str(q)
            completion = self.client.chat.completions.create(
                model=self.model_str,
                messages=[{
                    "role": "system",
                    "content": "You are a helpful assistant."
                }, {
                    "role": "user",
                    "content": b,
                }],
                max_tokens=self.max_token,
                stop=None,
                temperature=0.7)
            res = completion.choices[0].message.content
            responses.append(res)

        for idx, o in enumerate(responses):
            q_list[idx].response = o

        return q_list
