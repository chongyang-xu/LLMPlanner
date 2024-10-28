from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.agents.miniLLM import MiniLLM
from llm_planner.agents.Llama3_8B import Llama3_8B

import os, re


class SotAgent(Agent):

    def __init__(self):
        super().__init__()
        self.points = {}
        self.num_points = 0
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
{point}. {point_outline}:

"""

    async def process(self, sender_id, message: Message):

        if message["response"] is not None:
            rmsg = message["request_message"]
            if rmsg["outline"] is not None:
                outline = message["response"]
                print(f"outline : {outline}")
                # Extract points.
                re_result = re.findall(r"(\d+)\.\s?([\s\S]+?)(?=\n|\n*$)",
                                       outline)
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
                self.num_points = num_points
                messages = []
                request = rmsg["origin_req"]
                for point, point_outline in zip(points, point_outlines):
                    # print(f"{point}@{point_outline}")
                    point_prompt = self.POINT_PROMPT.format(
                        request=request,
                        point=point,
                        outline=outline,
                        point_outline=point_outline,
                    )
                    point_prompt = point_prompt.replace(
                        self.PROMPT_ROLE_SWITCH_STR, "")

                    # printf(f"point_prompt : {point_prompt}")

                    msg = message.spawn()
                    msg["content"] = point_prompt
                    msg["point"] = point
                    msg["point_outline"] = point_outline
                    # print(f"point prompt: {point_prompt}\n{'-'*10}")

                    self.send(self.minillm.id, msg)

            elif rmsg["point"] is not None:
                pid = rmsg["point"]
                self.points[pid] = f"{pid}. " + rmsg[
                    "point_outline"] + "\n    " + message["response"]

                if len(self.points) == self.num_points:
                    for idx, val in self.points.items():
                        print(f"{val}")

        elif message["content"] is not None:
            self.minillm = MiniLLM(max_token=16,
                                   return_value=True,
                                   with_batching=False,
                                   with_caching=False)

            request = message["content"]
            splits = self.OUTLINE_PROMPT.split(self.PROMPT_ROLE_SWITCH_STR)
            partial_answer = splits[1].strip()
            # print(f"partial_answer={partial_answer}")
            prompt = f"User:\n {splits[0].format(request=request)}\n{partial_answer}"
            # print(f"outline prompt: {prompt}\n{'-'*10}")
            msg = message.spawn()
            msg["content"] = prompt
            msg["outline"] = True
            msg["origin_req"] = request
            self.send(self.minillm.id, msg)


###################
# run test
###################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sot = SotAgent()
msg = Message()
msg['content'] = "What are the best large language models?"
sot.send(sot.id, msg)

System.start()
