from llm_planner.agents.miniLLM import MiniLLM

from llm_planner.message import Message
from llm_planner.agents.agent import Agent, AgentRef, agent_stop_all

import os, re

import logging
import signal

import pykka.debug

logging.basicConfig(level=logging.DEBUG)
signal.signal(signal.SIGUSR1, pykka.debug.log_thread_tracebacks)


def printf(input):
    print("-" * 50)
    print(input)
    print("-" * 50)


class SotAgent(Agent):

    def __init__(self, llm_agent: AgentRef):
        super().__init__()
        self.llm_agent = llm_agent
        self.PROMPT_ROLE_SWITCH_STR = "[ROLESWITCHING assistant:]"
        self.OUTLINE_PROMPT = """
You're an organizer responsible for only giving the skeleton (not the full content) for answering the question
Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question.
Instead of writing a full sentence, each skeleton point should be very short with only 3~5 words.
Generally, the skeleton should have 3~10 points.

Here are two questions:
<example>
Question:
What are the typical types of Chinese dishes?
Skeleton:
1. Dumplings.
2. Noodles.
3. Dim Sum.
4. Hot Pot.
5. Wonton.
6. Ma Po Tofu.
7. Char Siu.
8. Fried Rice.

Question:
What are some practical tips for individuals to reduce their carbon emissions?
Skeleton:
1. Energy conservation.
2. Efficient transportation.
3. Home energy efficiency.
4. Reduce water consumption.
5. Sustainable diet.
6. Sustainable travel.
</example>

Now, please provide the skeleton for the following question.
{request}
Skeleton:[ROLESWITCHING assistant:]
"""

        self.POINT_PROMPT = """
You're responsible for continuing the writing of one and only one point in the overall answer to the following question.

{request}

The skeleton of the answer is

{outline}

Continue and only continue the writing of point {point}. 
Write it **very shortly** in 1~2 sentence and do not continue with other points!
[ROLESWITCHING assistant:]
{point}. {point_outline}
"""

    def on_receive(self, message: Message):
        request = message["prompt"]
        splits = self.OUTLINE_PROMPT.split(self.PROMPT_ROLE_SWITCH_STR)

        partial_answer = splits[1]
        prompt = f"User:\n {splits[0].format(request=request)}\n\n Assistat:\n{partial_answer}"
        msg = Message(prompt=prompt)
        ret = self.llm_agent.ask(msg)
        #print(ret["ret"][0])

        outline = ret["ret"][0]

        # printf(f"outline : {outline}")

        # Extract points.
        re_result = re.findall(r"(\d+)\.\s?([\s\S]+?)(?=\n|\n*$)", outline)
        if len(re_result) > 0:
            points, point_outlines = zip(*re_result)
        else:
            points, point_outlines = [], []
        assert len(points) == len(point_outlines)

        num_points = len(points)
        if num_points > 0:
            # Filter to get unique point indexes
            points_filtered = []
            point_outlines_filtered = []
            points_set = set([])
            for i in range(num_points):
                if points[i] not in points_set:
                    points_set.add(points[i])
                    points_filtered.append(points[i])
                    point_outlines_filtered.append(point_outlines[i])
            points = points_filtered
            point_outlines = point_outlines_filtered

        num_points = len(points)
        messages = []
        for point, point_outline in zip(points, point_outlines):
            point_prompt = self.POINT_PROMPT.format(
                request=request,
                point=point,
                outline=outline,
                point_outline=point_outline,
            )
            point_prompt = point_prompt.replace(self.PROMPT_ROLE_SWITCH_STR, "")

            # printf(f"point_prompt : {point_prompt}")

            msg = Message(prompt=point_prompt)
            ret = self.llm_agent.ask(msg)
            printf(ret["ret"][0])


###################
# run test
###################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

minillm_ref = MiniLLM.start(max_token=64)
sot = SotAgent.start(minillm_ref)

begin = Message(prompt="What are the great large language models?")

sot.tell(begin)

# agent_stop_all()
