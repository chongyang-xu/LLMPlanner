from typing import List, Dict, Any

from llm_planner.util import timing
from llm_planner.message import Message
from llm_planner.service.service import SingleLLMServe

from llm_planner.logger import Logger

import anthropic

import os

# Create a custom logger
logger = Logger('HuggingFaceLogger')


class AnthropicServe_API(SingleLLMServe):

    def __init__(self, p_selector, policy_param_: Dict[str, Any]):

        super().__init__()

        # Now you can access the variables
        self.api_key = os.getenv('ANTHROPIC_API_KEY')

        self.p_selector = p_selector
        self.init_done = False
        ##########################
        #
        # setting parameters
        #
        ##########################
        self.model_str = policy_param_.get('model',
                                           "claude-3-5-sonnet-20240620")
        self.max_token = policy_param_.get('max_token', 16)

    def init_service(self):

        if self.init_done:
            return

        self.client = anthropic.Anthropic()

        self.init_done = True

    @timing
    def work_on(self, q_list: List[str]):
        self.init_service()
        responses = []
        for q in q_list:
            # b = str(q)

            response = self.client.beta.prompt_caching.messages.create(
                model=self.model_str,
                max_tokens=self.max_token,
                system=[
                    {
                        "type":
                            "text",
                        "text":
                            "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n",
                        #"cache_control": {
                        #    "type": "ephemeral"
                        #}
                    },
                ],
                messages=q
                #[{
                #    "role": "user",
                #    "content": b
                #}],
            )
            responses.append(response.content[0].text)

        return responses

        #for idx, o in enumerate(responses):
        #    q_list[idx].response = o

        #return q_list
