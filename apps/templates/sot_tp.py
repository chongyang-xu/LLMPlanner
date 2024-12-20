from llm_planner.message import Message
from llm_planner.templates.template import Template

import re

OUTLINE_PROMPT = """
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

POINT_PROMPT = """
You're responsible for continuing the writing of one and only one point in the overall answer to the following question.

{request}

The skeleton of the answer is

{outline}

Continue and only continue the writing of point {point}. 
Write it **very shortly** in 1~2 sentence and do not continue with other points!
[ROLESWITCHING assistant:]
{point}. {point_outline}:

"""

PROMPT_ROLE_SWITCH_STR = "[ROLESWITCHING assistant:]"


def outline_prompt(message: Message):
    splits = OUTLINE_PROMPT.split(PROMPT_ROLE_SWITCH_STR)
    partial_answer = splits[1].strip()
    # print(f"partial_answer={partial_answer}")

    request = message["content"]
    prompt = f"User:\n {splits[0].format(request=request)}\n{partial_answer}"
    # print(f"outline prompt: {prompt}\n{'-'*10}")
    msg = message.spawn()
    msg["content"] = prompt
    msg["outline"] = True
    msg["origin_req"] = request
    return msg


def split_outline(message: Message):
    rmsg = message["request_message"]
    assert rmsg["outline"] is not None

    outline = message["content"]
    print(f"outline:\n{outline}")
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
    request = rmsg["origin_req"]
    for point, point_outline in zip(points, point_outlines):
        # print(f"{point}@{point_outline}")
        point_prompt = POINT_PROMPT.format(
            request=request,
            point=point,
            outline=outline,
            point_outline=point_outline,
        )
        point_prompt = point_prompt.replace(PROMPT_ROLE_SWITCH_STR, "")

        # printf(f"point_prompt : {point_prompt}")

        msg = message.spawn()
        msg["content"] = point_prompt
        msg["point"] = point
        msg["point_outline"] = point_outline
        messages.append(msg)
    return messages


def assemble(msgs):
    ret = msgs[0].spawn()
    ret["content"] = ""

    points = {}
    for m in msgs:
        rmsg = m["request_message"]
        assert rmsg["point"] is not None

        pid = rmsg["point"]
        points[
            pid] = f"{pid}. " + rmsg["point_outline"] + "\n    " + m["content"]

    points = dict(sorted(points.items()))
    for idx, val in points.items():
        ret["content"] += f"{val}\n"
    return ret


sot = Template().input().map(outline_prompt).ask("mini_llm").map(
    split_outline).ask("mini_llm").reduce(assemble).print()

sot.start(["What are the best large language models?"])
