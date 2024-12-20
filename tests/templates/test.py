from llm_planner.message import Message

from llm_planner.templates.template import Template

import random


def update(msg):
    content = msg["content"]
    msgs = []
    for i in range(2):
        m = msg.spawn()
        m["content"] = content + " " + str(i)
        msgs.append(m)
    return msgs


test = Template().input().print()

test.start(["test"])
